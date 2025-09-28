#!/usr/bin/env python3
"""
AI 레벨 입력 검증 시스템
Issue 5 해결: 무의미한 입력 템플릿 생성 문제 - 3단계 AI 레벨 검증 강화
"""
from typing import Dict, Tuple, List, Optional
from enum import Enum
import re

class AIValidationError(Enum):
    """AI 검증 오류 타입"""
    MEANINGLESS_INPUT = "MEANINGLESS_INPUT"  # AI가 해석할 수 없는 입력
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"  # 정보 부족
    LOW_INTENT_CONFIDENCE = "LOW_INTENT_CONFIDENCE"  # 의도 파악 불가
    NO_EXTRACTABLE_VARIABLES = "NO_EXTRACTABLE_VARIABLES"  # 추출 가능한 변수 없음
    TEMPLATE_GENERATION_IMPOSSIBLE = "TEMPLATE_GENERATION_IMPOSSIBLE"  # 템플릿 생성 불가
    VALID = "VALID"

class AIInputValidator:
    """AI 레벨 입력 검증 클래스"""

    def __init__(self):
        """AI 검증기 초기화"""
        # AI 검증 임계값들
        self.min_intent_confidence = 0.4  # 최소 의도 신뢰도 40%
        self.min_completeness_score = 0.15  # 최소 완성도 15%
        self.min_meaningful_variables = 1  # 최소 의미있는 변수 개수
        self.min_template_quality = 0.3  # 최소 템플릿 품질 점수

        # 5W1H 필수 변수 목록
        self.essential_variables = [
            '무엇을', '누구에게', '언제', '어디서', '왜', '어떻게'
        ]

        # 알림톡 도메인 핵심 키워드
        self.domain_keywords = {
            '예약': ['예약', '방문', '접수', '신청', '등록'],
            '확인': ['확인', '체크', '검토', '점검'],
            '안내': ['안내', '알림', '공지', '통지', '전달'],
            '결제': ['결제', '납부', '수납', '청구', '정산'],
            '배송': ['배송', '택배', '발송', '도착', '출발'],
            '변경': ['변경', '수정', '조정', '취소'],
            '혜택': ['할인', '쿠폰', '혜택', '특가', '포인트']
        }

        print("AI 레벨 입력 검증 시스템 초기화 완료")

    def extract_basic_information(self, user_input: str) -> Dict[str, str]:
        """
        기본적인 정보 추출 (AI 없이 규칙 기반)
        Agent1 호출 전에 미리 확인할 수 있는 정보들
        """
        info = {}

        # 1. 시간/날짜 표현
        time_patterns = [
            r'(\d{1,2}시)', r'(\d{1,2}:\d{2})', r'(오전|오후)',
            r'(내일|오늘|어제)', r'(\d{1,2}월\s?\d{1,2}일)', r'(주말|평일)'
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, user_input)
            if matches:
                info['언제'] = ' '.join([match if isinstance(match, str) else match[0] for match in matches])
                break

        # 2. 장소 표현
        place_patterns = [
            r'(카페|식당|병원|학원|회사|학교|사무실)',
            r'(\w+점)', r'(\w+센터)', r'(\w+병원)', r'(\w+학원)'
        ]
        for pattern in place_patterns:
            matches = re.findall(pattern, user_input)
            if matches:
                info['어디서'] = ' '.join(matches)
                break

        # 3. 행동/목적 표현
        action_patterns = [
            r'(예약|방문|확인|안내|결제|주문|배송|취소|변경)',
            r'(만들어|생성|작성|보내|발송)'
        ]
        for pattern in action_patterns:
            matches = re.findall(pattern, user_input)
            if matches:
                info['무엇을'] = ' '.join(matches)
                break

        # 4. 대상 표현
        target_patterns = [
            r'(고객|회원|님|분|씨|선생님|사장님)',
            r'(\w+님)', r'(\w+회원)'
        ]
        for pattern in target_patterns:
            matches = re.findall(pattern, user_input)
            if matches:
                info['누구에게'] = ' '.join(matches)
                break

        # 5. 연락처 정보
        contact_patterns = [
            r'(01[0-9]-?\d{3,4}-?\d{4})',  # 전화번호
            r'(\w+@\w+\.\w+)'  # 이메일
        ]
        for pattern in contact_patterns:
            matches = re.findall(pattern, user_input)
            if matches:
                info['어떻게'] = '연락처'
                break

        return info

    def calculate_domain_relevance(self, user_input: str) -> float:
        """알림톡 도메인과의 관련성 점수 계산"""
        if not user_input:
            return 0.0

        total_keywords = 0
        matched_keywords = 0

        for category, keywords in self.domain_keywords.items():
            total_keywords += len(keywords)
            for keyword in keywords:
                if keyword in user_input:
                    matched_keywords += 1

        return matched_keywords / total_keywords if total_keywords > 0 else 0.0

    def analyze_input_complexity(self, user_input: str) -> Dict:
        """입력 복잡도 분석"""
        if not user_input:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0.0,
                'complexity_score': 0.0
            }

        # 단어 수 계산
        words = user_input.split()
        word_count = len(words)

        # 문장 수 계산
        sentences = re.split(r'[.!?]', user_input)
        sentence_count = len([s for s in sentences if s.strip()])

        # 평균 단어 길이
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

        # 복잡도 점수 계산 (0.0 ~ 1.0)
        complexity_score = min((word_count * 0.1 + sentence_count * 0.2 + avg_word_length * 0.1) / 3, 1.0)

        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'complexity_score': complexity_score
        }

    def validate_pre_ai_analysis(self, user_input: str) -> Tuple[bool, AIValidationError, str]:
        """
        AI 분석 전 사전 검증 (빠른 차단)
        """
        if not user_input or not user_input.strip():
            return False, AIValidationError.MEANINGLESS_INPUT, "입력이 비어있습니다."

        # 1. 기본 정보 추출
        basic_info = self.extract_basic_information(user_input)
        meaningful_info_count = len([v for v in basic_info.values() if v and v.strip()])

        # 2. 도메인 관련성 확인
        domain_relevance = self.calculate_domain_relevance(user_input)

        # 3. 입력 복잡도 확인
        complexity = self.analyze_input_complexity(user_input)

        # 4. 종합 판단
        # 의미있는 정보가 하나도 없고, 도메인 관련성도 없고, 복잡도도 낮으면 차단
        if (meaningful_info_count == 0 and
            domain_relevance < 0.1 and
            complexity['complexity_score'] < 0.2):
            return False, AIValidationError.MEANINGLESS_INPUT, "해석 가능한 정보가 없습니다. 구체적인 알림톡 내용을 입력해주세요."

        # 사전 검증 통과
        return True, AIValidationError.VALID, "사전 검증 통과"

    def validate_agent1_result(self, agent1_result: Dict, user_input: str) -> Tuple[bool, AIValidationError, str]:
        """
        Agent1 분석 결과 검증
        """
        if not agent1_result:
            return False, AIValidationError.MEANINGLESS_INPUT, "Agent1 분석 결과가 없습니다."

        # 1. 의도 신뢰도 검증
        intent_info = agent1_result.get('intent', {})
        intent_confidence = intent_info.get('confidence', 0.0)

        if intent_confidence < self.min_intent_confidence:
            return False, AIValidationError.LOW_INTENT_CONFIDENCE, f"의도 파악이 어렵습니다 (신뢰도: {intent_confidence:.1%}). 더 구체적으로 입력해주세요."

        # 2. 변수 추출 결과 검증
        variables = agent1_result.get('variables', {})
        meaningful_variables = 0

        for var_key, var_value in variables.items():
            if var_value and var_value != '없음' and len(str(var_value).strip()) > 1:
                meaningful_variables += 1

        if meaningful_variables < self.min_meaningful_variables:
            return False, AIValidationError.NO_EXTRACTABLE_VARIABLES, f"추출 가능한 정보가 부족합니다 ({meaningful_variables}개). 더 자세히 입력해주세요."

        # 3. 완성도 점수 검증
        completeness_info = agent1_result.get('mandatory_check', {})
        completeness_score = completeness_info.get('completeness_score', 0.0)

        if completeness_score < self.min_completeness_score:
            return False, AIValidationError.INSUFFICIENT_INFORMATION, f"정보가 부족합니다 (완성도: {completeness_score:.1%}). 5W1H 정보를 더 포함해주세요."

        # 4. 검증 상태 확인
        validation_info = agent1_result.get('validation', {})
        validation_status = validation_info.get('status', 'unknown')

        if validation_status == 'profanity_detected':
            return False, AIValidationError.MEANINGLESS_INPUT, "부적절한 내용이 감지되었습니다."

        # Agent1 검증 통과
        return True, AIValidationError.VALID, "Agent1 분석 결과 검증 통과"

    def validate_template_generation_feasibility(self, agent1_result: Dict) -> Tuple[bool, AIValidationError, str]:
        """
        템플릿 생성 가능성 검증
        """
        # 1. 필수 정보 확인
        variables = agent1_result.get('variables', {})
        intent = agent1_result.get('intent', {}).get('intent', '기타')

        # 2. 기타 의도이면서 정보가 거의 없으면 차단
        if intent == '기타':
            meaningful_vars = len([v for v in variables.values() if v and v != '없음'])
            if meaningful_vars <= 1:
                return False, AIValidationError.TEMPLATE_GENERATION_IMPOSSIBLE, "템플릿 생성에 필요한 정보가 부족합니다. 목적과 내용을 명확히 해주세요."

        # 3. 핵심 변수 중 하나라도 있는지 확인
        essential_found = False
        for var_key, var_value in variables.items():
            if var_key in ['무엇을', '누구에게'] and var_value and var_value != '없음':
                essential_found = True
                break

        if not essential_found:
            return False, AIValidationError.TEMPLATE_GENERATION_IMPOSSIBLE, "알림톡 생성에 필요한 핵심 정보('무엇을', '누구에게')가 없습니다."

        return True, AIValidationError.VALID, "템플릿 생성 가능"

    def comprehensive_ai_validation(self, user_input: str, agent1_result: Optional[Dict] = None) -> Tuple[bool, AIValidationError, str]:
        """
        종합적인 AI 레벨 검증
        """
        # 1. 사전 검증 (AI 분석 전)
        pre_valid, pre_error, pre_message = self.validate_pre_ai_analysis(user_input)
        if not pre_valid:
            return False, pre_error, pre_message

        # 2. Agent1 결과가 있으면 검증
        if agent1_result:
            # Agent1 결과 검증
            agent1_valid, agent1_error, agent1_message = self.validate_agent1_result(agent1_result, user_input)
            if not agent1_valid:
                return False, agent1_error, agent1_message

            # 템플릿 생성 가능성 검증
            template_valid, template_error, template_message = self.validate_template_generation_feasibility(agent1_result)
            if not template_valid:
                return False, template_error, template_message

        # 모든 검증 통과
        return True, AIValidationError.VALID, "AI 레벨 검증 통과"

    def get_ai_validation_suggestions(self, error_type: AIValidationError, user_input: str = "") -> str:
        """AI 검증 오류에 따른 개선 제안"""
        suggestions = {
            AIValidationError.MEANINGLESS_INPUT: "알림톡 목적을 명확히 해주세요. 예: '예약 확인', '할인 안내', '배송 알림' 등",
            AIValidationError.LOW_INTENT_CONFIDENCE: "더 구체적으로 입력해주세요. 예: '카페 예약 확인 메시지를 만들어주세요'",
            AIValidationError.NO_EXTRACTABLE_VARIABLES: "5W1H 정보를 포함해주세요: 무엇을, 누구에게, 언제, 어디서, 왜, 어떻게",
            AIValidationError.INSUFFICIENT_INFORMATION: "더 자세한 정보를 포함해주세요. 예: '내일 오후 2시 카페 예약 확인'",
            AIValidationError.TEMPLATE_GENERATION_IMPOSSIBLE: "알림톡의 목적과 대상을 명확히 해주세요."
        }
        return suggestions.get(error_type, "올바른 입력으로 다시 시도해주세요.")

    def analyze_input_quality(self, user_input: str, agent1_result: Optional[Dict] = None) -> Dict:
        """입력 품질 상세 분석"""
        analysis = {
            'basic_info': self.extract_basic_information(user_input),
            'domain_relevance': self.calculate_domain_relevance(user_input),
            'complexity': self.analyze_input_complexity(user_input),
            'pre_validation': {}
        }

        # 사전 검증 결과
        pre_valid, pre_error, pre_message = self.validate_pre_ai_analysis(user_input)
        analysis['pre_validation'] = {
            'is_valid': pre_valid,
            'error_type': pre_error.value,
            'message': pre_message
        }

        # Agent1 결과가 있으면 추가 분석
        if agent1_result:
            agent1_valid, agent1_error, agent1_message = self.validate_agent1_result(agent1_result, user_input)
            template_valid, template_error, template_message = self.validate_template_generation_feasibility(agent1_result)

            analysis['agent1_validation'] = {
                'is_valid': agent1_valid,
                'error_type': agent1_error.value,
                'message': agent1_message
            }

            analysis['template_validation'] = {
                'is_valid': template_valid,
                'error_type': template_error.value,
                'message': template_message
            }

            # 종합 검증
            final_valid, final_error, final_message = self.comprehensive_ai_validation(user_input, agent1_result)
            analysis['final_validation'] = {
                'is_valid': final_valid,
                'error_type': final_error.value,
                'message': final_message
            }

        return analysis


# 전역 인스턴스 (싱글톤)
_ai_validator = None

def get_ai_validator() -> AIInputValidator:
    """싱글톤 AI 검증기 반환"""
    global _ai_validator
    if _ai_validator is None:
        _ai_validator = AIInputValidator()
    return _ai_validator

# 편의 함수들
def validate_ai_input(user_input: str, agent1_result: Optional[Dict] = None) -> Tuple[bool, AIValidationError, str]:
    """AI 입력 검증 편의 함수"""
    validator = get_ai_validator()
    return validator.comprehensive_ai_validation(user_input, agent1_result)

def validate_pre_ai(user_input: str) -> Tuple[bool, AIValidationError, str]:
    """AI 분석 전 사전 검증 편의 함수"""
    validator = get_ai_validator()
    return validator.validate_pre_ai_analysis(user_input)

def is_ai_interpretable(user_input: str) -> bool:
    """AI 해석 가능성 간단 확인"""
    valid, _, _ = validate_pre_ai(user_input)
    return valid


if __name__ == "__main__":
    # 테스트 코드
    print("AI 레벨 입력 검증 시스템 테스트")
    print("=" * 50)

    validator = AIInputValidator()

    # 다양한 AI 해석 가능성 테스트 케이스
    test_cases = [
        # 완전히 무의미한 입력들 (차단되어야 함)
        "ff",
        "asdfasdfiasj",
        "ㅁㄴㅇㄹ",
        "aaaaaa",
        "1234567",

        # 전화번호/영어 이름만 (차단되어야 함)
        "010-1234-5678",
        "John Smith",
        "010-1234-5678 John Smith",

        # 정보가 부족하지만 일부 해석 가능 (경고)
        "예약",
        "안내",
        "John Smith 예약",

        # 적절한 수준의 정보 (통과해야 함)
        "카페 예약 확인",
        "할인 쿠폰 안내",
        "내일 병원 예약 확인 안내",
        "고객님께 배송 알림 메시지",
        "회원님 포인트 적립 안내",

        # 복잡하고 상세한 정보 (통과해야 함)
        "내일 오후 2시 강남역 카페에서 동창회 모임이 있다고 알림톡 보내줘",
        "주문하신 상품이 내일 도착한다고 고객님께 배송 안내 메시지 작성해주세요"
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i:2d}. 테스트: '{test_input}'")

        # 상세 분석
        analysis = validator.analyze_input_quality(test_input)

        print(f"    기본정보: {list(analysis['basic_info'].keys())}")
        print(f"    도메인관련성: {analysis['domain_relevance']:.2f}")
        print(f"    복잡도점수: {analysis['complexity']['complexity_score']:.2f}")

        # 사전 검증 결과
        pre_val = analysis['pre_validation']
        status = "통과" if pre_val['is_valid'] else "차단"
        print(f"    사전검증: {status} - {pre_val['message']}")

        if not pre_val['is_valid']:
            error_type = AIValidationError(pre_val['error_type'])
            suggestion = validator.get_ai_validation_suggestions(error_type, test_input)
            print(f"    제안: {suggestion}")

    print(f"\nAI 레벨 입력 검증 시스템 테스트 완료")