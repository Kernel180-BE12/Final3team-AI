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


class Agent1:
    """
    질의 분석 및 검증 에이전트
    """
    
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
        
        # 필수 변수 정의
        self.mandatory_variables = [
            '누가 (To/Recipient)', 
            '무엇을 (What/Subject)', 
            '어떻게 (How/Method)'
        ]
        
        print("Agent1 초기화 완료")
    
    def _load_policy_document(self) -> str:
        """정책 문서 로드"""
        try:
            policy_path = project_root / "data" / "message_yuisahang.md"
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
    
    def check_mandatory_variables(self, variables: Dict[str, str]) -> Dict[str, Any]:
        """
        고정 변수 유무 체크
        
        Args:
            variables: 추출된 변수들
            
        Returns:
            필수 변수 체크 결과
        """
        missing_mandatory = []
        
        for var in self.mandatory_variables:
            if variables.get(var, '없음') == '없음' or not variables.get(var, '').strip():
                missing_mandatory.append(var)
        
        return {
            'is_complete': len(missing_mandatory) == 0,
            'missing_mandatory': missing_mandatory,
            'total_mandatory': len(self.mandatory_variables),
            'completed_mandatory': len(self.mandatory_variables) - len(missing_mandatory)
        }
    
    def generate_reask_question(self, missing_variables: List[str]) -> str:
        """
        부족한 변수에 대한 재질문 생성
        
        Args:
            missing_variables: 부족한 변수 리스트
            
        Returns:
            재질문 문자열
        """
        if not missing_variables:
            return ""
        
        var_questions = {
            '누가 (To/Recipient)': "누구에게 보낼 메시지인가요? (예: 고객님, 회원님, 특정 고객층)",
            '무엇을 (What/Subject)': "무엇에 대한 내용인가요? (예: 주문 확인, 이벤트 안내, 시스템 점검)",
            '어떻게 (How/Method)': "어떤 방식으로 안내하시겠어요? (예: 알림 메시지, 확인 요청, 정보 제공)",
            '언제 (When/Time)': "언제와 관련된 내용인가요? (예: 특정 날짜, 시간, 기간)",
            '어디서 (Where/Place)': "어느 장소와 관련된 내용인가요? (예: 매장, 온라인, 특정 위치)",
            '왜 (Why/Reason)': "왜 이 메시지를 보내야 하나요? (예: 안내 목적, 확인 요청, 마케팅)"
        }
        
        questions = []
        for var in missing_variables:
            if var in var_questions:
                questions.append(f"{var_questions[var]}")
        
        if questions:
            return f"추가 정보가 필요합니다:\n\n" + "\n".join(questions) + "\n\n위 정보를 추가로 입력해주세요."
        
        return "추가 정보를 입력해주세요."
    
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
    
    def check_final_profanity(self, text: str) -> bool:
        """
        최종 비속어 검사 (정책 검사와 함께 수행)
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            True: 비속어 검출됨, False: 정상
        """
        return self.check_initial_profanity(text)
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """
        메인 처리 로직
        
        Args:
            user_input: 사용자 입력
            
        Returns:
            처리 결과
        """
        print(f"\nAgent1 처리 시작: '{user_input[:50]}...'")
        
        # 1. 초기 비속어 검출
        if self.check_initial_profanity(user_input):
            return {
                'status': 'error',
                'error_type': 'profanity',
                'message': "비속어가 검출되었습니다. 프롬프트를 다시 작성해주세요.",
                'restart_required': True
            }
        
        # 2. 질의 분석
        analysis_result = self.analyze_query(user_input)
        variables = analysis_result['variables']
        
        # 3. 고정 변수 체크
        mandatory_check = self.check_mandatory_variables(variables)
        
        # 4. 부족한 변수가 있으면 재질문
        if not mandatory_check['is_complete']:
            missing_vars = mandatory_check['missing_mandatory']
            reask_question = self.generate_reask_question(missing_vars)
            
            return {
                'status': 'reask_required',
                'message': reask_question,
                'analysis': analysis_result,
                'missing_variables': missing_vars,
                'completed_variables': mandatory_check['completed_mandatory'],
                'total_variables': mandatory_check['total_mandatory']
            }
        
        # 5. 선택변수가 모두 찬 경우 정책 및 비속어 검사
        combined_text = user_input + " " + " ".join([v for v in variables.values() if v != '없음'])
        
        # 정책 준수 확인
        policy_result = self.check_policy_compliance(combined_text, variables)
        
        # 최종 비속어 검사
        has_profanity = self.check_final_profanity(combined_text)
        
        # 6. 위반사항이 있으면 안내 후 재시작
        if not policy_result['is_compliant']:
            violation_msg = "\n".join([f"• {v}" for v in policy_result['violations']])
            return {
                'status': 'policy_violation',
                'message': f"정책 위반이 감지되었습니다:\n\n{violation_msg}\n\n프롬프트를 다시 작성해주세요.",
                'violations': policy_result['violations'],
                'restart_required': True
            }
        
        if has_profanity:
            return {
                'status': 'profanity_violation',
                'message': "비속어가 감지되었습니다. 프롬프트를 다시 작성해주세요.",
                'restart_required': True
            }
        
        # 7. 모든 검사 통과 - 성공 결과 반환
        return {
            'status': 'success',
            'message': "모든 검사를 통과했습니다. 템플릿 생성이 가능합니다.",
            'analysis': analysis_result,
            'variables': variables,
            'intent': analysis_result['intent'],
            'policy_result': policy_result,
            'selected_variables': {k: v for k, v in variables.items() if v != '없음'}
        }
    
    def interactive_session(self):
        """
        대화형 세션 실행 (독립 실행용)
        """
        print("Agent1 - 질의 분석 및 검증 시스템")
        print("=" * 50)
        print("알림톡 템플릿을 위한 내용을 입력해주세요.")
        print("종료하려면 'quit', 'exit', '종료'를 입력하세요.\n")
        
        while True:
            try:
                # 사용자 입력 받기
                user_input = input("내용을 입력하세요: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    print("Agent1을 종료합니다.")
                    break
                
                if not user_input:
                    print("입력이 비어있습니다.\n")
                    continue
                
                # 재질문 루프
                current_input = user_input
                variables_dict = {}
                
                while True:
                    result = self.process_query(current_input)
                    
                    if result['status'] == 'error' and result.get('restart_required'):
                        print(f"\n{result['message']}\n")
                        print("처음부터 다시 시작합니다.\n")
                        break
                    
                    elif result['status'] in ['policy_violation', 'profanity_violation']:
                        print(f"\n{result['message']}\n")
                        print("처음부터 다시 시작합니다.\n")
                        break
                    
                    elif result['status'] == 'reask_required':
                        print(f"\n{result['message']}\n")
                        
                        # 현재까지 완성된 변수 정보 표시
                        completed = result['completed_variables']
                        total = result['total_variables']
                        print(f"진행 상황: {completed}/{total} 필수 변수 완성\n")
                        
                        # 추가 정보 입력 받기
                        additional_input = input("추가 정보를 입력하세요: ").strip()
                        
                        if additional_input:
                            # 기존 입력과 추가 입력을 합쳐서 다시 처리
                            current_input = current_input + " " + additional_input
                        else:
                            print("추가 정보가 없습니다.\n")
                            break
                    
                    elif result['status'] == 'success':
                        print(f"\n{result['message']}\n")
                        
                        # 결과 정보 출력
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
                print(f"오류가 발생했습니다: {e}")


def main():
    """메인 실행 함수"""
    try:
        agent = Agent1()
        agent.interactive_session()
    except Exception as e:
        print(f"Agent1 초기화 실패: {e}")


if __name__ == "__main__":
    main()