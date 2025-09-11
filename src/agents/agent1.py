#!/usr/bin/env python3
"""
Agent1 - 질의 분석 및 검증 에이전트

역할:
1. 사용자 입력 → 비속어 검출 (즉시 중단)
2. 질의 분석 (변수 추출 + 의도 분류)
3. 고정 변수 유무 체크
4. 부족한 변수 → 재질문을 통해 변수 채우기
5. 선택 변수 완성 → 정책 및 비속어 최종 검사
6. 위반 시 안내 후 처음부터 다시 시작
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent  # src/agents/ -> src/ -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import GEMINI_API_KEY

# __init__.py를 거치지 않고 직접 import
import importlib.util

# variable_extractor 직접 로드
spec = importlib.util.spec_from_file_location("variable_extractor", project_root / "src" / "tools" / "variable_extractor.py")
variable_extractor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variable_extractor_module)
VariableExtractor = variable_extractor_module.VariableExtractor

# intent_classifier 직접 로드  
spec = importlib.util.spec_from_file_location("intent_classifier", project_root / "src" / "tools" / "intent_classifier.py")
intent_classifier_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intent_classifier_module)
IntentClassifier = intent_classifier_module.IntentClassifier


class ConversationState:
    """대화 상태 관리 클래스"""
    
    # 필수 변수 정의 (클래스 상수)
    MANDATORY_VARIABLES = ['누가 (To/Recipient)', '무엇을 (What/Subject)', '어떻게 (How/Method)']
    
    def __init__(self):
        self.variables = {
            '누가 (To/Recipient)': '없음',
            '무엇을 (What/Subject)': '없음',
            '어떻게 (How/Method)': '없음',
            '언제 (When/Time)': '없음',
            '어디서 (Where/Place)': '없음',
            '왜 (Why/Reason)': '없음'
        }
        
    def update_variables(self, new_variables: Dict[str, str]):
        """변수 업데이트 (기존 값이 '없음'일 때만)"""
        for key, value in new_variables.items():
            if key in self.variables and value != '없음' and value.strip():
                if self.variables[key] == '없음':
                    self.variables[key] = value
                        
    def get_confirmed_variables(self) -> Dict[str, str]:
        """확정된 변수들만 반환"""
        return {k: v for k, v in self.variables.items() if v != '없음'}
        
    def get_missing_variables(self) -> List[str]:
        """누락된 변수들 반환"""
        return [var for var in self.MANDATORY_VARIABLES if self.variables.get(var, '없음') == '없음']
        
    def check_mandatory_variables(self) -> Dict[str, Any]:
        """
        필수 변수 완성도 체크
        
        Returns:
            필수 변수 체크 결과
        """
        missing_mandatory = []
        
        for var in self.MANDATORY_VARIABLES:
            value = self.variables.get(var, '없음')
            # '없음'으로 시작하거나 빈 값이면 미완성으로 처리
            if value == '없음' or value.startswith('없음') or not value.strip():
                missing_mandatory.append(var)
        
        return {
            'is_complete': len(missing_mandatory) == 0,
            'missing_mandatory': missing_mandatory,
            'total_mandatory': len(self.MANDATORY_VARIABLES),
            'completed_mandatory': len(self.MANDATORY_VARIABLES) - len(missing_mandatory)
        }
    
    def ai_judge_completeness(self, user_input: str, intent: Dict[str, Any], ai_model) -> Dict[str, Any]:
        """
        AI 기반 변수 완성도 판단
        
        Args:
            user_input: 원본 사용자 입력
            intent: 의도 분류 결과
            ai_model: Gemini AI 모델
            
        Returns:
            AI 판단 결과
        """
        confirmed_vars = self.get_confirmed_variables()
        
        # AI 판단 프롬프트 생성
        prompt = f"""
        사용자가 카카오 알림톡 템플릿을 만들려고 합니다. 매우 엄격하게 판단해주세요.
        
        [사용자 원본 요청]
        {user_input}
        
        [의도 분류]
        의도: {intent.get('intent', '불명')}
        신뢰도: {intent.get('confidence', 0.0)}
        
        [현재 파악된 변수]
        {chr(10).join([f"- {k}: {v}" for k, v in confirmed_vars.items()]) if confirmed_vars else "- 아직 파악된 변수 없음"}
        
        [미완성 변수]
        {chr(10).join([f"- {k}: 없음" for k, v in self.variables.items() if v == '없음'])}
        
        ⚠️ 매우 엄격한 판단 기준:
        1. 누가 (To/Recipient): 반드시 명확해야 함 - "친구", "고객", "회원" 등
        2. 무엇을 (What/Subject): 구체적인 내용 필요 - "점심", "회의", "이벤트" 등  
        3. 어떻게 (How/Method): 알림/안내/확인 등 방식이 있어야 함
        4. 언제 (When/Time): 대부분의 알림톡에서 중요함
        5. 어디서 (Where/Place): 만남/방문이 관련된 경우 필수
        6. 왜 (Why/Reason): 복잡한 경우 이유가 필요할 수 있음
        
        ❌ 불완성 판단 기준:
        - 핵심 정보 3개(누가, 무엇을, 어떻게) 중 하나라도 추론하기 어려우면 INCOMPLETE
        - 알림톡 수신자가 혼란스러울 수 있으면 INCOMPLETE  
        - 실제 비즈니스에서 사용하기엔 정보 부족하면 INCOMPLETE
        
        응답 형식:
        완성도: [COMPLETE/INCOMPLETE]
        필요한_추가변수: [변수명1, 변수명2] (없으면 없음)
        이유: [구체적인 부족한 이유]
        """
        
        try:
            response = ai_model.generate_content(prompt)
            result = self._parse_completion_response(response.text, 'needed_variables')
            return result
        except Exception as e:
            print(f"AI 판단 중 오류: {e}")
            # 폴백: 기본 필수 변수 체크
            mandatory_result = self.check_mandatory_variables()
            return {
                'is_complete': mandatory_result['is_complete'],
                'needed_variables': mandatory_result['missing_mandatory'],
                'reasoning': 'AI 판단 실패로 기본 필수 변수 체크 사용'
            }
    
    def _parse_completion_response(self, response: str, variables_key: str = 'needed_variables') -> Dict[str, Any]:
        """AI 완성도 응답 파싱 (통합 버전)"""
        lines = response.strip().split('\n')
        result = {
            'is_complete': False,
            variables_key: [],
            'reasoning': 'AI 응답 파싱 실패'
        }
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if '완성도' in key:
                    result['is_complete'] = 'COMPLETE' in value.upper()
                elif ('필요한' in key or '부족한' in key) and '변수' in key:
                    if '없음' not in value and value.strip():
                        variables = [v.strip() for v in value.replace('[', '').replace(']', '').split(',')]
                        result[variables_key] = [v for v in variables if v and v != '없음']
                elif '이유' in key:
                    result['reasoning'] = value
        
        return result


class Agent1:
    """
    질의 분석 및 검증 에이전트
    """
    
    # 변수 질문 템플릿 (클래스 상수)
    VARIABLE_QUESTIONS = {
        '누가 (To/Recipient)': "누구에게 보낼 메시지인가요? (예: 고객님, 회원님, 특정 고객층)",
        '무엇을 (What/Subject)': "무엇에 대한 내용인가요? (예: 주문 확인, 이벤트 안내, 시스템 점검)",
        '어떻게 (How/Method)': "어떤 방식으로 안내하시겠어요? (예: 알림 메시지, 확인 요청, 정보 제공)",
        '언제 (When/Time)': "언제와 관련된 내용인가요? (예: 특정 날짜, 시간, 기간)",
        '어디서 (Where/Place)': "어느 장소와 관련된 내용인가요? (예: 매장, 온라인, 특정 위치)",
        '왜 (Why/Reason)': "왜 이 메시지를 보내야 하나요? (예: 안내 목적, 확인 요청, 마케팅)"
    }
    
    def __init__(self, api_key: str = None):
        """
        Agent1 초기화
        
        Args:
            api_key: Gemini API 키
        """
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
        
        # 도구들 초기화
        self.variable_extractor = VariableExtractor(self.api_key)
        self.intent_classifier = IntentClassifier(self.api_key)
        
        # 정책 문서 로드
        self.policy_content = self._load_policy_document()
        
        # 비속어 키워드 로드
        self.profanity_keywords = self._load_profanity_keywords()
        
        # 대화 상태 초기화
        self.conversation_state = None
        
        print("Agent1 초기화 완료")
    
    def _load_policy_document(self) -> str:
        """정책 문서 로드"""
        try:
            policy_path = project_root / "data" / "cleaned_alrimtalk.md"
            if policy_path.exists():
                with open(policy_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                print("정책 문서를 찾을 수 없습니다.")
                return ""
        except Exception as e:
            print(f"정책 문서 로드 실패: {e}")
            return ""
    
    def _load_profanity_keywords(self) -> set:
        """비속어 키워드 로드"""
        try:
            keyword_path = project_root / "predata" / "cleaned_blacklist_keyword.txt"
            if keyword_path.exists():
                with open(keyword_path, 'r', encoding='utf-8') as f:
                    keywords = set()
                    for line in f:
                        keyword = line.strip()
                        if keyword:
                            keywords.add(keyword.lower())
                    print(f"비속어 키워드 {len(keywords)}개 로드 완료")
                    return keywords
            else:
                print("비속어 키워드 파일을 찾을 수 없습니다.")
                return set()
        except Exception as e:
            print(f"비속어 키워드 로드 실패: {e}")
            return set()
    
    def check_initial_profanity(self, text: str) -> bool:
        """
        초기 비속어 검출 (blacklist_keyword 파일 기반)
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            True: 비속어 검출됨, False: 정상
        """
        if not self.profanity_keywords:
            return False
            
        text_lower = text.lower()
        
        # 공백 제거해서도 체크
        text_no_space = text_lower.replace(" ", "")
        
        for keyword in self.profanity_keywords:
            if keyword in text_lower or keyword in text_no_space:
                print(f"비속어 검출: '{keyword}'")
                return True
        
        return False
    
    def analyze_query(self, user_input: str) -> Dict[str, Any]:
        """
        질의 분석 (변수 추출 + 의도 분류)
        
        Args:
            user_input: 사용자 입력
            
        Returns:
            분석 결과 딕셔너리
        """
        print("질의 분석 시작...")
        
        # 1. 변수 추출
        variables = self.variable_extractor.extract_variables(user_input)
        
        # 2. 의도 분류
        intent_result = self.intent_classifier.classify_intent(user_input, variables)
        
        # 3. 변수 유효성 검증
        validation = self.variable_extractor.validate_variables(variables)
        mandatory_check = self.variable_extractor.check_mandatory_variables(variables)
        
        analysis_result = {
            'user_input': user_input,
            'variables': variables,
            'intent': intent_result,
            'validation': validation,
            'mandatory_check': mandatory_check,
            'missing_variables': self.variable_extractor.get_missing_variables(variables)
        }
        
        print(f"분석 완료 - 의도: {intent_result['intent']}, 완성도: {mandatory_check['completeness_score']:.1%}")
        return analysis_result
    
    
    def generate_contextual_reask(self, conversation_state: ConversationState, missing_variables: List[str]) -> str:
        """
        컨텍스트를 고려한 재질문 생성
        
        Args:
            conversation_state: 현재 대화 상태
            missing_variables: 부족한 변수 리스트
            
        Returns:
            컨텍스트 어웨어 재질문 문자열
        """
        if not missing_variables:
            return ""
        
        # 확정된 정보 표시
        confirmed_info = conversation_state.get_confirmed_variables()
        
        result = "**현재까지 파악된 정보:**\n"
        if confirmed_info:
            for key, value in confirmed_info.items():
                result += f"  [확정] {key}: {value}\n"
        else:
            result += "  (아직 확정된 정보가 없습니다)\n"
        
        result += "\n**추가로 필요한 정보:**\n"
        
        # 변수명 매핑 (AI 응답과 실제 변수명 매칭)
        variable_mapping = {
            '누가': '누가 (To/Recipient)',
            '무엇을': '무엇을 (What/Subject)',  
            '어떻게': '어떻게 (How/Method)',
            '언제': '언제 (When/Time)',
            '어디서': '어디서 (Where/Place)',
            '왜': '왜 (Why/Reason)'
        }
        
        for i, var in enumerate(missing_variables, 1):
            # 매핑된 변수명으로 변환
            mapped_var = variable_mapping.get(var, var)
            
            if mapped_var in self.VARIABLE_QUESTIONS:
                result += f"  {i}. {self.VARIABLE_QUESTIONS[mapped_var]}\n"
            elif var in self.VARIABLE_QUESTIONS:
                result += f"  {i}. {self.VARIABLE_QUESTIONS[var]}\n"
        
        result += "\n**안내:** 한 번에 여러 정보를 콤마(,)로 구분해서 입력해도 됩니다."
        
        return result
    
    def determine_needed_variables_by_intent(self, user_input: str, intent: str, confidence: float, current_variables: Dict[str, str]) -> Dict[str, Any]:
        """
        의도에 따른 필요 변수 AI 판단
        
        Args:
            user_input: 원본 사용자 입력
            intent: 분류된 의도
            confidence: 의도 분류 신뢰도
            current_variables: 현재 변수 상태
            
        Returns:
            필요 변수 판단 결과
        """
        confirmed_vars = {k: v for k, v in current_variables.items() if not (v == '없음' or v.startswith('없음') or not v.strip())}
        missing_vars = {k: v for k, v in current_variables.items() if (v == '없음' or v.startswith('없음') or not v.strip())}
        
        prompt = f"""
        카카오 알림톡 템플릿을 만들기 위해 사용자 의도에 따라 필요한 변수들을 판단해주세요.

        [사용자 입력]
        "{user_input}"
        
        [분류된 의도]
        의도: {intent}
        신뢰도: {confidence:.2f}
        
        [현재 확정된 변수]
        {chr(10).join([f"- {k}: {v}" for k, v in confirmed_vars.items()]) if confirmed_vars else "- 없음"}
        
        [현재 누락된 변수]
        {chr(10).join([f"- {k}" for k in missing_vars.keys()]) if missing_vars else "- 없음"}
        
        알림톡 의도별 필요 변수 가이드:
        - 예약알림/리마인드: 누가, 무엇을, 언제, 어디서 (필수), 어떻게 (중요)
        - 주문확인: 누가, 무엇을, 어떻게 (필수), 언제 (중요)
        - 신청접수: 누가, 무엇을, 어떻게 (필수), 언제, 왜 (중요)
        - 이벤트안내: 누가, 무엇을, 어떻게, 언제 (필수), 어디서, 왜 (중요)
        - 시스템알림: 누가, 무엇을, 어떻게 (필수), 언제 (중요)
        
        판단 기준:
        1. 해당 의도에서 반드시 필요한 변수들이 모두 채워졌는가?
        2. 알림톡 수신자가 이해하기에 충분한 정보인가?
        3. 실제 비즈니스에서 사용 가능한 완성도인가?
        
        응답 형식:
        완성도: [COMPLETE/INCOMPLETE]
        부족한_변수: [변수명1, 변수명2] (없으면 "없음")
        이유: [구체적인 이유 설명]
        """
        
        try:
            response = self.variable_extractor.model.generate_content(prompt)
            result = self._parse_completion_response(response.text, 'missing_variables')
            return result
        except Exception as e:
            print(f"의도별 변수 판단 중 오류: {e}")
            # 폴백: 기본 필수 변수 체크
            basic_check = self.conversation_state.check_mandatory_variables()
            return {
                'is_complete': basic_check['is_complete'],
                'missing_variables': basic_check['missing_mandatory'],
                'reasoning': 'AI 판단 실패로 기본 필수 변수 체크 사용'
            }
    
    
    def check_policy_compliance(self, text: str, variables: Dict[str, str]) -> Dict[str, Any]:
        """
        정책 준수 여부 확인 (message_yuisahang 문서 기반)
        
        Args:
            text: 검사할 텍스트
            variables: 추출된 변수들
            
        Returns:
            정책 준수 결과
        """
        violations = []
        text_lower = text.lower()
        
        if not self.policy_content:
            return {
                'is_compliant': True,
                'violations': ["정책 문서를 로드할 수 없어 검사를 건너뜁니다."],
                'risk_level': 'UNKNOWN'
            }
        
        policy_lower = self.policy_content.lower()
        
        # 1. 광고성 내용 검사
        ad_keywords = ["할인", "이벤트", "무료", "특가", "프로모션", "쿠폰"]
        has_ad_content = any(keyword in text_lower for keyword in ad_keywords)
        
        if has_ad_content:
            if "[광고]" not in text and "광고" in policy_lower:
                violations.append("광고성 내용이지만 [광고] 표기가 없습니다")
        
        # 2. message_yuisahang 문서의 금지사항 검사
        # 청소년 유해 정보 검사
        youth_harmful = ["주류", "전자담배", "성인", "19세"]
        if any(keyword in text_lower for keyword in youth_harmful):
            if "청소년" in policy_lower and "유해" in policy_lower:
                violations.append("청소년 유해 정보 관련 - 연령 인증 필요")
        
        # 금융 관련 제한사항
        financial_keywords = ["결제", "송금", "납부", "대출", "투자", "주식"]
        if any(keyword in text_lower for keyword in financial_keywords):
            if "금융" in policy_lower and "제한" in policy_lower:
                violations.append("금융 관련 내용 - 정책 검토 필요")
        
        # 개인정보 관련
        if "개인정보" in text_lower or "정보 수집" in text_lower:
            if "개인정보" in policy_lower and "동의" in policy_lower:
                violations.append("개인정보 수집 시 동의 절차 필요")
        
        # 스팸성 표현 검사
        spam_keywords = ["긴급", "마지막", "즉시", "빨리", "한정", "선착순"]
        spam_count = sum(1 for keyword in spam_keywords if keyword in text_lower)
        if spam_count >= 2:
            violations.append("스팸성 표현 과다 사용")
        
        # 위험도 계산
        risk_level = "HIGH" if len(violations) >= 2 else ("MEDIUM" if violations else "LOW")
        
        return {
            'is_compliant': len(violations) == 0,
            'violations': violations,
            'risk_level': risk_level
        }
    
    
    def process_query(self, user_input: str, is_follow_up: bool = False) -> Dict[str, Any]:
        """
        메인 처리 로직 (개선된 컨텍스트 관리)
        
        Args:
            user_input: 사용자 입력
            is_follow_up: 재질문 후 추가 입력인지 여부
            
        Returns:
            처리 결과
        """
        print(f"\nAgent1 처리 시작: '{user_input[:50]}...'")
        
        # 대화 상태 초기화 (첫 입력인 경우)
        if not is_follow_up or self.conversation_state is None:
            self.conversation_state = ConversationState()
        
        # 1. 초기 비속어 검출
        if self.check_initial_profanity(user_input):
            self.conversation_state = None  # 상태 초기화
            return {
                'status': 'error',
                'error_type': 'profanity',
                'message': "비속어가 검출되었습니다. 프롬프트를 다시 작성해주세요.",
                'restart_required': True
            }
        
        # 2. 질의 분석 (새로운 입력만)
        analysis_result = self.analyze_query(user_input)
        new_variables = analysis_result['variables']
        
        # 3. 변수 업데이트 (점진적)
        if is_follow_up:
            # 추가 입력에 대한 더 스마트한 변수 매핑
            self.smart_variable_update(user_input, new_variables)
        else:
            # 첫 입력인 경우 기본 업데이트
            self.conversation_state.update_variables(new_variables)
        
        # 4. 의도 기반 필요 변수 판단
        current_variables = self.conversation_state.variables
        intent = analysis_result['intent']['intent']
        confidence = analysis_result['intent']['confidence']
        
        needed_vars_result = self.determine_needed_variables_by_intent(
            user_input, intent, confidence, current_variables
        )
        
        # 필요한 변수가 부족하면 재질문
        if not needed_vars_result['is_complete']:
            missing_vars = needed_vars_result['missing_variables']
            reask_question = self.generate_contextual_reask(self.conversation_state, missing_vars)
            
            return {
                'status': 'reask_required',
                'message': reask_question,
                'analysis': analysis_result,
                'missing_variables': missing_vars,
                'reasoning': needed_vars_result['reasoning']
            }
        
        # 7. AI도 완성이라고 판단하면 정책 및 비속어 검사
        confirmed_vars = self.conversation_state.get_confirmed_variables()
        combined_text = " ".join([v for v in confirmed_vars.values()])
        
        # 정책 준수 확인
        policy_result = self.check_policy_compliance(combined_text, current_variables)
        
        # 최종 비속어 검사
        has_profanity = self.check_initial_profanity(combined_text)
        
        # 8. 위반사항이 있으면 안내 후 재시작
        if not policy_result['is_compliant']:
            violation_msg = "\n".join([f"• {v}" for v in policy_result['violations']])
            self.conversation_state = None  # 상태 초기화
            return {
                'status': 'policy_violation',
                'message': f"정책 위반이 감지되었습니다:\n\n{violation_msg}\n\n프롬프트를 다시 작성해주세요.",
                'violations': policy_result['violations'],
                'restart_required': True
            }
        
        if has_profanity:
            self.conversation_state = None  # 상태 초기화
            return {
                'status': 'profanity_violation',
                'message': "비속어가 감지되었습니다. 프롬프트를 다시 작성해주세요.",
                'restart_required': True
            }
        
        # 9. 모든 검사 통과 - 성공 결과 반환
        return {
            'status': 'success',
            'message': "모든 검사를 통과했습니다. 템플릿 생성이 가능합니다.",
            'analysis': analysis_result,
            'variables': current_variables,
            'intent': analysis_result['intent'],
            'policy_result': policy_result,
            'selected_variables': confirmed_vars
        }
        
    def smart_variable_update(self, user_input: str, new_variables: Dict[str, str]):
        """
        하이브리드 변수 매핑: 규칙 기반 + AI 보조
        
        Args:
            user_input: 사용자 입력
            new_variables: 새로 추출된 변수들
        """
        print(f"하이브리드 매핑 시작: '{user_input}'")
        
        missing_vars = self.conversation_state.get_missing_variables()
        if not missing_vars:
            self.conversation_state.update_variables(new_variables)
            return
        
        # 1단계: 규칙 기반 매핑
        rule_mapping = self._rule_based_mapping(user_input, missing_vars)
        print(f"규칙 매핑 결과: {rule_mapping}")
        
        # 2단계: 규칙으로 해결 안 된 것만 AI 처리
        remaining_vars = []
        for var in missing_vars:
            if var in rule_mapping:
                self.conversation_state.variables[var] = rule_mapping[var]
                print(f"규칙 적용: {var} = {rule_mapping[var]}")
            else:
                remaining_vars.append(var)
        
        # 3단계: AI 보조 (남은 변수만)
        if remaining_vars:
            print(f"AI 보조 처리 대상: {remaining_vars}")
            ai_mapping = self._ai_assisted_mapping(user_input, remaining_vars)
            
            for var in remaining_vars:
                if var in ai_mapping:
                    self.conversation_state.variables[var] = ai_mapping[var]
                    print(f"AI 적용: {var} = {ai_mapping[var]}")
        
        # 4단계: LLM 추출 결과도 반영 (빈 값만)
        self.conversation_state.update_variables(new_variables)
        print(f"최종 상태: {self.conversation_state.get_confirmed_variables()}")
    
    def _rule_based_mapping(self, user_input: str, missing_vars: List[str]) -> Dict[str, str]:
        """규칙 기반 변수 매핑"""
        result = {}
        
        # 명확한 키워드 패턴
        EXACT_PATTERNS = {
            '누가 (To/Recipient)': ['신청자', '회원', '고객', '참가자', '구매자', '사용자', '등록자'],
            '어떻게 (How/Method)': ['알림', '안내', '공지', '메시지', '통지', '발송', '전달', '메세지'],
            '어디서 (Where/Place)': ['온라인', '오프라인', '카페', '회의실', '매장', '센터', '학원', '강의실'],
            '언제 (When/Time)': ['오늘', '내일', '모레', '다음주', '시간', '날짜'],
            '왜 (Why/Reason)': ['때문', '위해', '목적', '이유로']
        }
        
        # 컨텍스트 조합 인식
        CONTEXT_COMBINATIONS = {
            ('부트캠프', '신청자'): '부트캠프 신청자',
            ('설명회', '참가자'): '설명회 참가자', 
            ('강의', '수강생'): '강의 수강생',
            ('이벤트', '참여자'): '이벤트 참여자',
            ('세미나', '참석자'): '세미나 참석자'
        }
        
        input_words = user_input.lower().replace(',', ' ').replace('.', ' ').split()
        
        # 1. 컨텍스트 조합 우선 체크
        for (context, target), full_name in CONTEXT_COMBINATIONS.items():
            if context in user_input and target in user_input and '누가 (To/Recipient)' in missing_vars:
                result['누가 (To/Recipient)'] = full_name
                break
        
        # 2. 개별 패턴 매핑
        for var_name in missing_vars:
            if var_name in result:  # 이미 매핑됨
                continue
                
            if var_name in EXACT_PATTERNS:
                keywords = EXACT_PATTERNS[var_name]
                for word in input_words:
                    if any(keyword in word for keyword in keywords):
                        # 전체 단어나 조합 찾기
                        if var_name == '누가 (To/Recipient)':
                            # 앞뒤 단어와 조합해서 더 의미있는 표현 만들기
                            word_index = input_words.index(word)
                            if word_index > 0:
                                result[var_name] = f"{input_words[word_index-1]} {word}"
                            else:
                                result[var_name] = word
                        else:
                            result[var_name] = word
                        break
        
        return result
    
    def _ai_assisted_mapping(self, user_input: str, remaining_vars: List[str]) -> Dict[str, str]:
        """AI 보조 매핑 (규칙 실패 시만)"""
        if not remaining_vars:
            return {}
        
        prompt = f"""
        규칙 매핑이 실패한 변수들을 AI로 처리합니다.
        
        입력: "{user_input}"
        남은 변수: {remaining_vars}
        
        각 변수에 대해 가장 적절한 값을 매핑하세요.
        
        응답 형식:
        {chr(10).join([f'{var}: [값 또는 "없음"]' for var in remaining_vars])}
        """
        
        try:
            response = self.variable_extractor.model.generate_content(prompt)
            return self._parse_ai_mapping(response.text)
        except Exception as e:
            print(f"AI 보조 매핑 오류: {e}")
            return {}
    
    def _parse_ai_mapping(self, response: str) -> Dict[str, str]:
        """AI 매핑 응답 파싱"""
        result = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                # "누가 (To/Recipient): 백엔드 부트캠프 신청자들" 형태 파싱
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # 변수명 정규화
                    if '누가' in key:
                        key = '누가 (To/Recipient)'
                    elif '무엇을' in key:
                        key = '무엇을 (What/Subject)'
                    elif '어떻게' in key:
                        key = '어떻게 (How/Method)'
                    elif '언제' in key:
                        key = '언제 (When/Time)'
                    elif '어디서' in key:
                        key = '어디서 (Where/Place)'
                    elif '왜' in key:
                        key = '왜 (Why/Reason)'
                    
                    if value and value != '없음':
                        result[key] = value
        
        return result
    
    def interactive_session(self):
        """대화형 세션 실행 (독립 실행용)"""
        print("Agent1 - 질의 분석 및 검증 시스템")
        print("=" * 50)
        print("알림톡 템플릿을 위한 내용을 입력해주세요.")
        print("종료하려면 'quit', 'exit', '종료'를 입력하세요.\n")
        
        while True:
            try:
                user_input = input("내용을 입력하세요: ").strip()
                if user_input.lower() in ['quit', 'exit', '종료']:
                    print("Agent1을 종료합니다.")
                    break
                if not user_input:
                    continue
                
                # 재질문 루프
                is_first_input = True
                current_input = user_input
                
                while True:
                    result = self.process_query(current_input, is_follow_up=not is_first_input)
                    
                    if result.get('restart_required') or result['status'] in ['error', 'policy_violation', 'profanity_violation']:
                        print(f"\n{result['message']}\n처음부터 다시 시작합니다.\n")
                        break
                    
                    elif result['status'] == 'reask_required':
                        print(f"\n{result['message']}\n")
                        additional_input = input("추가 정보를 입력하세요: ").strip()
                        if additional_input:
                            current_input = additional_input
                            is_first_input = False
                        else:
                            break
                    
                    elif result['status'] == 'success':
                        print(f"\n{result['message']}\n")
                        print("분석 결과:")
                        print(f"  의도: {result['intent']['intent']}")
                        print(f"  신뢰도: {result['intent']['confidence']:.2f}")
                        print("\n선택된 변수:")
                        for key, value in result['selected_variables'].items():
                            print(f"  • {key}: {value}")
                        print(f"\n정책 준수: {'통과' if result['policy_result']['is_compliant'] else '위반'}")
                        print("-" * 50)
                        break
                
            except KeyboardInterrupt:
                print("\n\n사용자가 중단했습니다.")
                break
            except Exception as e:
                print(f"오류: {e}")


def main():
    """메인 실행 함수"""
    try:
        agent = Agent1()
        agent.interactive_session()
    except Exception as e:
        print(f"Agent1 초기화 실패: {e}")


if __name__ == "__main__":
    main()