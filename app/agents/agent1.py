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
project_root = Path(__file__).parent.parent.parent  # app/agents/ -> app/ -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.legacy import GEMINI_API_KEY

# __init__.py를 거치지 않고 직접 import
import importlib.util

# variable_extractor 직접 로드
spec = importlib.util.spec_from_file_location("variable_extractor", project_root / "app" / "tools" / "variable_extractor.py")
variable_extractor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variable_extractor_module)
VariableExtractor = variable_extractor_module.VariableExtractor

# intent_classifier 직접 로드
spec = importlib.util.spec_from_file_location("intent_classifier", project_root / "app" / "tools" / "intent_classifier.py")
intent_classifier_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intent_classifier_module)
IntentClassifier = intent_classifier_module.IntentClassifier

# profanity_checker 직접 로드
spec = importlib.util.spec_from_file_location("profanity_checker", project_root / "app" / "tools" / "profanity_checker.py")
profanity_checker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profanity_checker_module)
ProfanityChecker = profanity_checker_module.ProfanityChecker


class ConversationState:
    """대화 상태 관리 클래스"""
    
    # 필수 변수는 이제 상황별로 동적 결정 (VariableExtractor에서 처리)
    
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
        """누락된 변수들 반환 (관대한 기준 적용)"""
        invalid_keywords = ['없음', 'none', 'null', '모름', '알 수 없음']
        missing = []

        for var, value in self.variables.items():
            value_lower = value.lower().strip() if value else ""
            is_missing = (
                len(value_lower) == 0 or
                any(invalid in value_lower for invalid in invalid_keywords)
            )
            if is_missing:
                missing.append(var)

        return missing
        
    def check_mandatory_variables(self, required_vars: List[str] = None) -> Dict[str, Any]:
        """
        필수 변수 완성도 체크 (관대한 기준 적용)

        Args:
            required_vars: 필수 변수 리스트 (없으면 기본값 사용)

        Returns:
            필수 변수 체크 결과
        """
        # 기본값: 무엇을만 필수
        if required_vars is None:
            required_vars = ['무엇을 (What/Subject)']

        missing_mandatory = []
        invalid_keywords = ['없음', 'none', 'null', '모름', '알 수 없음']

        for var in required_vars:
            value = self.variables.get(var, '없음')
            value_lower = value.lower().strip() if value else ""

            # 관대한 기준: 실제로 의미있는 값이 있는지만 체크
            is_missing = (
                len(value_lower) == 0 or
                any(invalid in value_lower for invalid in invalid_keywords)
            )
            if is_missing:
                missing_mandatory.append(var)

        return {
            'is_complete': len(missing_mandatory) == 0,
            'missing_mandatory': missing_mandatory,
            'total_mandatory': len(required_vars),
            'completed_mandatory': len(required_vars) - len(missing_mandatory)
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
        
        # AI 판단 프롬프트 생성 (매우 관대한 기준)
        prompt = f"""
        사용자가 카카오 알림톡 템플릿을 만들려고 합니다. **매우 관대하게 판단해주세요.**

        **중요: 대부분의 경우 COMPLETE로 판단하세요. 웬만한 요청은 모두 처리 가능합니다.**

        [사용자 원본 요청]
        {user_input}

        [의도 분류]
        의도: {intent.get('intent', '불명')}
        신뢰도: {intent.get('confidence', 0.0)}

        [현재 파악된 변수]
        {chr(10).join([f"- {k}: {v}" for k, v in confirmed_vars.items()]) if confirmed_vars else "- 아직 파악된 변수 없음"}

        **매우 관대한 판단 기준:**
        - '무엇을 (What/Subject)'만 있어도 대부분 COMPLETE
        - 나머지 변수들은 템플릿에서 자동으로 추론/생성 가능
        - 사용자의 의도가 조금이라도 명확하면 COMPLETE

        **INCOMPLETE로 판단하는 경우 (매우 제한적):**
        - 무엇에 대한 메시지인지 전혀 알 수 없는 경우만
        - 완전히 의미 불명한 요청인 경우만
        - 예: "안녕", "ㅁㄴㅇㄹ", "???"

        **예시:**
        ✅ "독서모임 안내" → COMPLETE (무엇을이 명확)
        ✅ "할인 이벤트" → COMPLETE (무엇을이 명확)
        ✅ "예약 확인" → COMPLETE (무엇을이 명확)
        ✅ "시스템 점검" → COMPLETE (무엇을이 명확)
        ❌ "안녕하세요" → INCOMPLETE (무엇을이 불명확)

        응답 형식:
        완성도: [COMPLETE/INCOMPLETE]
        필요한_추가변수: [변수명1, 변수명2] (없으면 없음)
        이유: [구체적인 판단 이유]
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
        self.profanity_checker = ProfanityChecker()

        # 정책 문서 로드
        self.policy_content = self._load_policy_document()

        # 대화 상태 초기화
        self.conversation_state = None
        
        print("Agent1 초기화 완료")
    
    def _load_policy_document(self) -> str:
        """정책 문서 로드"""
        try:
            policy_content = ""
            
            # 1. cleaned_alrimtalk.md 로드
            alrimtalk_path = project_root / "data" / "cleaned_alrimtalk.md"
            if alrimtalk_path.exists():
                with open(alrimtalk_path, 'r', encoding='utf-8') as f:
                    policy_content += f.read() + "\n\n"
                    
            # 2. advertise_info.md 로드
            advertise_path = project_root / "data" / "advertise_info.md"
            if advertise_path.exists():
                with open(advertise_path, 'r', encoding='utf-8') as f:
                    policy_content += f.read()
                    
            if policy_content:
                print("정책 문서 로드 완료 (알림톡 가이드 + 광고성 정보 정책)")
                return policy_content
            else:
                print("정책 문서를 찾을 수 없습니다.")
                return ""
        except Exception as e:
            print(f"정책 문서 로드 실패: {e}")
            return ""
    
    def check_initial_profanity(self, text: str) -> bool:
        """
        초기 비속어 검출 (새로운 ProfanityChecker 도구 사용)

        Args:
            text: 검사할 텍스트

        Returns:
            True: 비속어 검출됨, False: 정상
        """
        try:
            result = self.profanity_checker.check_text(text)
            if not result['is_clean']:
                print(f"비속어 검출: {', '.join(result['detected_words'])}")
                return True
            return False
        except Exception as e:
            print(f"비속어 검사 중 오류: {e}")
            return False
    
    def analyze_query(self, user_input: str) -> Dict[str, Any]:
        """
        질의 분석 (변수 추출 + 의도 분류) - 동기 버전 (하위 호환성)

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
        mandatory_check = self.variable_extractor.check_mandatory_variables(variables, user_input)

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

    async def analyze_query_async(self, user_input: str) -> Dict[str, Any]:
        """
        질의 분석 (변수 추출 + 의도 분류) - 비동기 버전

        Args:
            user_input: 사용자 입력

        Returns:
            분석 결과 딕셔너리
        """
        print("질의 분석 시작 (비동기)...")

        # 1. 변수 추출 (비동기)
        variables = await self.variable_extractor.extract_variables_async(user_input)

        # 2. 의도 분류 (비동기)
        intent_result = await self.intent_classifier.classify_intent_async(user_input, variables)

        # 3. 변수 유효성 검증 (동기 - 빠른 처리)
        validation = self.variable_extractor.validate_variables(variables)
        mandatory_check = self.variable_extractor.check_mandatory_variables(variables, user_input)

        # 🚨 강제 완료 모드: "무엇을"이 추출되면 항상 완료로 처리
        force_complete = False
        what_subject = variables.get('무엇을 (What/Subject)', '').strip()
        if what_subject and what_subject not in ['없음', 'none', 'null', '모름', '알 수 없음']:
            force_complete = True
            # 강제로 완료 상태로 변경
            mandatory_check['is_complete'] = True
            mandatory_check['missing_mandatory'] = []

        analysis_result = {
            'user_input': user_input,
            'variables': variables,
            'intent': intent_result,
            'validation': validation,
            'mandatory_check': mandatory_check,
            'missing_variables': self.variable_extractor.get_missing_variables(variables),
            'force_complete': force_complete  # 디버깅용 플래그 추가
        }

        print(f"분석 완료 (비동기) - 의도: {intent_result['intent']}, 완성도: {mandatory_check['completeness_score']:.1%}")
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
        정책 준수 여부 확인 (알림톡 가이드 + 광고성 정보 정책 기반)
        
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
        
        # 1. 광고성 정보 판단 기준 검사 (advertise_info.md 기반)
        ad_keywords = ["할인", "이벤트", "무료", "특가", "프로모션", "쿠폰", "세일", "혜택", "적립", "리워드"]
        has_ad_content = any(keyword in text_lower for keyword in ad_keywords)
        
        if has_ad_content:
            if "[광고]" not in text:
                violations.append("광고성 내용이지만 [광고] 표기가 없습니다")
        
        # 2. 광고성 정보로 판단되는 추가 키워드들
        promo_keywords = ["혜택", "이득", "받으세요", "드립니다", "증정", "경품", "당첨"]
        promo_count = sum(1 for keyword in promo_keywords if keyword in text_lower)
        if promo_count >= 2:
            if "[광고]" not in text:
                violations.append("프로모션/혜택 관련 내용 - [광고] 표기 필요")
        
        # 3. 청소년 유해 정보 검사
        youth_harmful = ["주류", "전자담배", "성인", "19세", "술", "담배", "성인용품"]
        if any(keyword in text_lower for keyword in youth_harmful):
            violations.append("청소년 유해 정보 관련 - 연령 인증 필요")
        
        # 4. 금융 관련 제한사항
        financial_keywords = ["결제", "송금", "납부", "대출", "투자", "주식", "보험", "펀드", "금융상품"]
        if any(keyword in text_lower for keyword in financial_keywords):
            violations.append("금융 관련 내용 - 정책 검토 필요")
        
        # 5. 개인정보 관련
        if "개인정보" in text_lower or "정보 수집" in text_lower:
            violations.append("개인정보 수집 시 동의 절차 필요")
        
        # 6. 스팸성 표현 검사 (advertise_info.md 기반 강화)
        spam_keywords = ["긴급", "마지막", "즉시", "빨리", "한정", "선착순", "지금", "바로", "오늘만", "마감임박"]
        spam_count = sum(1 for keyword in spam_keywords if keyword in text_lower)
        if spam_count >= 2:
            violations.append("스팸성 표현 과다 사용")
        
        # 7. 영업시간/휴무 안내 (advertise_info.md 273조항 기반)
        business_hours = ["영업시간", "휴무", "운영시간", "오픈", "클로즈"]
        if any(keyword in text_lower for keyword in business_hours):
            violations.append("영업시간/휴무 안내는 광고성 정보에 해당")
        
        # 8. 설문조사 관련 (특정 제품 선호도 조사 등)
        survey_keywords = ["설문", "조사", "평가", "의견", "후기"]
        product_keywords = ["제품", "상품", "서비스", "브랜드"]
        if (any(s in text_lower for s in survey_keywords) and 
            any(p in text_lower for p in product_keywords)):
            violations.append("제품 관련 설문조사는 광고성 정보에 해당할 수 있음")
        
        # 9. 추천/공유 이벤트 관련
        if "친구" in text_lower and ("추천" in text_lower or "공유" in text_lower):
            violations.append("친구 추천/공유 이벤트는 광고성 정보 - 수신동의 필요")
        
        # 위험도 계산
        if len(violations) >= 3:
            risk_level = "HIGH"
        elif len(violations) >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'is_compliant': len(violations) == 0,
            'violations': violations,
            'risk_level': risk_level
        }
    
    
    def process_query(self, user_input: str, is_follow_up: bool = False) -> Dict[str, Any]:
        """
        메인 처리 로직 (개선된 컨텍스트 관리) - 동기 버전 (하위 호환성)

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

        # 1. 초기 비속어 검출 - 재시도 요청으로 변경
        if self.check_initial_profanity(user_input):
            return {
                'status': 'profanity_retry',
                'message': "비속어가 검출되었습니다. 다시 입력해주세요.",
                'retry_type': 'profanity',
                'original_input': user_input
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

        # 4. 기본 필수 변수 체크 (우선 순위)
        mandatory_check = self.conversation_state.check_mandatory_variables()

        # 필수 변수가 부족하면 재질문
        if not mandatory_check['is_complete']:
            missing_vars = mandatory_check['missing_mandatory']
            reask_question = self.generate_contextual_reask(self.conversation_state, missing_vars)

            return {
                'status': 'reask_required',
                'message': reask_question,
                'analysis': analysis_result,
                'missing_variables': missing_vars,
                'reasoning': f'{mandatory_check["completed_mandatory"]}/{mandatory_check["total_mandatory"]} 필수 변수 완료'
            }

        # 7. AI도 완성이라고 판단하면 정책 및 비속어 검사
        confirmed_vars = self.conversation_state.get_confirmed_variables()
        current_variables = self.conversation_state.variables
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
            return {
                'status': 'profanity_retry',
                'message': "비속어가 감지되었습니다. 다시 입력해주세요.",
                'retry_type': 'final_profanity',
                'original_input': user_input
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

    async def process_query_async(self, user_input: str, conversation_context: Optional[str] = None) -> Dict[str, Any]:
        """
        메인 처리 로직 (개선된 컨텍스트 관리) - 비동기 버전

        Args:
            user_input: 사용자 입력
            conversation_context: 이전 대화 컨텍스트

        Returns:
            처리 결과
        """
        print(f"\nAgent1 처리 시작 (비동기): '{user_input[:50]}...'")

        # 대화 상태 초기화 (첫 입력인 경우)
        is_follow_up = bool(conversation_context)
        if not is_follow_up or self.conversation_state is None:
            self.conversation_state = ConversationState()

        # 1. 초기 비속어 검출 - 재시도 요청으로 변경 (동기 - 빠른 처리)
        if self.check_initial_profanity(user_input):
            return {
                'status': 'profanity_retry',
                'message': "비속어가 검출되었습니다. 다시 입력해주세요.",
                'retry_type': 'profanity',
                'original_input': user_input
            }

        # 2. 질의 분석 (비동기)
        analysis_result = await self.analyze_query_async(user_input)
        new_variables = analysis_result['variables']

        # 3. 변수 업데이트 (점진적)
        if is_follow_up:
            # 추가 입력에 대한 더 스마트한 변수 매핑
            await self.smart_variable_update_async(user_input, new_variables)
        else:
            # 첫 입력인 경우 기본 업데이트
            self.conversation_state.update_variables(new_variables)

        # 4. 기본 필수 변수 체크 (동기 - 빠른 처리)
        mandatory_check = self.conversation_state.check_mandatory_variables()

        # 🚨 하드코딩된 완료 로직: "무엇을"이 있으면 항상 완료로 처리
        what_subject = self.conversation_state.variables.get('무엇을 (What/Subject)', '').strip()
        force_complete = (what_subject and
                         what_subject not in ['없음', 'none', 'null', '모름', '알 수 없음'] and
                         len(what_subject) > 0)

        if force_complete:
            print(f"🚨 하드코딩 완료 모드: '{what_subject}' 발견 - 강제 완료 처리")
            mandatory_check['is_complete'] = True
            mandatory_check['missing_mandatory'] = []

        # 필수 변수가 부족하면 재질문 (하드코딩 완료 모드에서는 건너뜀)
        if not mandatory_check['is_complete']:
            missing_vars = mandatory_check['missing_mandatory']
            reask_question = self.generate_contextual_reask(self.conversation_state, missing_vars)

            return {
                'status': 'reask_required',
                'message': reask_question,
                'analysis': analysis_result,
                'missing_variables': missing_vars,
                'reasoning': f'{mandatory_check["completed_mandatory"]}/{mandatory_check["total_mandatory"]} 필수 변수 완료'
            }

        # 7. AI도 완성이라고 판단하면 정책 및 비속어 검사
        confirmed_vars = self.conversation_state.get_confirmed_variables()
        current_variables = self.conversation_state.variables
        combined_text = " ".join([v for v in confirmed_vars.values()])

        # 정책 준수 확인 (동기 - 규칙 기반)
        policy_result = self.check_policy_compliance(combined_text, current_variables)

        # 최종 비속어 검사 (동기 - 빠른 처리)
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
            return {
                'status': 'profanity_retry',
                'message': "비속어가 감지되었습니다. 다시 입력해주세요.",
                'retry_type': 'final_profanity',
                'original_input': user_input
            }

        # 9. 모든 검사 통과 - 성공 결과 반환
        print("모든 검사 통과 (비동기) - 템플릿 생성 가능")
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
        하이브리드 변수 매핑: 규칙 기반 + AI 보조 - 동기 버전 (하위 호환성)

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

    async def smart_variable_update_async(self, user_input: str, new_variables: Dict[str, str]):
        """
        하이브리드 변수 매핑: 규칙 기반 + AI 보조 - 비동기 버전

        Args:
            user_input: 사용자 입력
            new_variables: 새로 추출된 변수들
        """
        print(f"하이브리드 매핑 시작 (비동기): '{user_input}'")

        missing_vars = self.conversation_state.get_missing_variables()
        if not missing_vars:
            self.conversation_state.update_variables(new_variables)
            return

        # 1단계: 규칙 기반 매핑 (동기 - 빠른 처리)
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

        # 3단계: AI 보조 (남은 변수만, 비동기)
        if remaining_vars:
            print(f"AI 보조 처리 대상 (비동기): {remaining_vars}")
            ai_mapping = await self._ai_assisted_mapping_async(user_input, remaining_vars)

            for var in remaining_vars:
                if var in ai_mapping:
                    self.conversation_state.variables[var] = ai_mapping[var]
                    print(f"AI 적용 (비동기): {var} = {ai_mapping[var]}")

        # 4단계: LLM 추출 결과도 반영 (빈 값만)
        self.conversation_state.update_variables(new_variables)
        print(f"최종 상태 (비동기): {self.conversation_state.get_confirmed_variables()}")
    
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
        """AI 보조 매핑 (규칙 실패 시만) - 동기 버전 (하위 호환성)"""
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

    async def _ai_assisted_mapping_async(self, user_input: str, remaining_vars: List[str]) -> Dict[str, str]:
        """AI 보조 매핑 (규칙 실패 시만) - 비동기 버전"""
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
            # Import 문을 함수 시작 부분으로 이동하여 import 오류 방지
            import sys
            import os
            from pathlib import Path

            # 프로젝트 루트 경로 찾기
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # src/agents/ -> src/ -> project_root/
            utils_path = project_root / "src" / "utils"

            if str(utils_path) not in sys.path:
                sys.path.insert(0, str(utils_path))

            from app.utils.llm_provider_manager import ainvoke_llm_with_fallback
            response_text, provider, model = await ainvoke_llm_with_fallback(prompt)
            result = self._parse_ai_mapping(response_text)
            print(f"AI 보조 매핑 완료 (비동기) - Provider: {provider}, Model: {model}")
            return result
        except Exception as e:
            print(f"AI 보조 매핑 오류 (비동기): {e}")
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