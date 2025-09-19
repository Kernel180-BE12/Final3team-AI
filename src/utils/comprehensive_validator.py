#!/usr/bin/env python3
"""
통합 입력 검증 시스템
Issue 5 해결: 무의미한 입력 템플릿 생성 문제 - 3단계 통합 검증
"""
from typing import Tuple, Dict, Optional
from enum import Enum

# 3단계 검증 모듈 import
from .language_detector import get_language_detector, ValidationError as LangError
from .korean_validator import get_korean_validator, KoreanValidationError
from .ai_input_validator import get_ai_validator, AIValidationError

class ValidationStage(Enum):
    """검증 단계"""
    STAGE_1_LANGUAGE = "STAGE_1_LANGUAGE"
    STAGE_2_KOREAN_QUALITY = "STAGE_2_KOREAN_QUALITY"
    STAGE_3_AI_LEVEL = "STAGE_3_AI_LEVEL"
    ALL_STAGES_PASSED = "ALL_STAGES_PASSED"

class ComprehensiveValidationResult:
    """종합 검증 결과"""
    def __init__(self):
        self.is_valid = False
        self.failed_stage = None
        self.error_code = ""
        self.error_message = ""
        self.suggestion = ""
        self.stage_results = {}

class ComprehensiveValidator:
    """3단계 통합 입력 검증 시스템"""

    def __init__(self):
        """통합 검증기 초기화"""
        self.language_detector = get_language_detector()
        self.korean_validator = get_korean_validator()
        self.ai_validator = get_ai_validator()

        print("🛡️ 3단계 통합 입력 검증 시스템 초기화 완료")

    def validate_input_comprehensive(self, user_input: str, agent1_result: Optional[Dict] = None) -> ComprehensiveValidationResult:
        """
        3단계 종합 입력 검증 실행

        Args:
            user_input: 사용자 입력
            agent1_result: Agent1 분석 결과 (선택적)

        Returns:
            ComprehensiveValidationResult: 종합 검증 결과
        """
        result = ComprehensiveValidationResult()

        # 1단계: 언어 감지 및 기본 검증
        print(f"🔍 1단계 검증: 언어 감지 및 기본 검증")
        stage1_valid, stage1_error, stage1_message = self.language_detector.validate_language_input(user_input)

        result.stage_results['stage1'] = {
            'valid': stage1_valid,
            'error_type': stage1_error.value,
            'message': stage1_message
        }

        if not stage1_valid:
            result.is_valid = False
            result.failed_stage = ValidationStage.STAGE_1_LANGUAGE
            result.error_code = stage1_error.value
            result.error_message = stage1_message
            result.suggestion = self.language_detector.get_validation_suggestion(user_input, stage1_error)
            return result

        print(f"✅ 1단계 통과: {stage1_message}")

        # 2단계: 한국어 품질 검증
        print(f"🔍 2단계 검증: 한국어 품질 검증")
        stage2_valid, stage2_error, stage2_message = self.korean_validator.validate_korean_quality(user_input)

        result.stage_results['stage2'] = {
            'valid': stage2_valid,
            'error_type': stage2_error.value,
            'message': stage2_message
        }

        if not stage2_valid:
            result.is_valid = False
            result.failed_stage = ValidationStage.STAGE_2_KOREAN_QUALITY
            result.error_code = stage2_error.value
            result.error_message = stage2_message
            result.suggestion = self.korean_validator.get_quality_suggestions(user_input, stage2_error)
            return result

        print(f"✅ 2단계 통과: {stage2_message}")

        # 3단계: AI 레벨 검증
        print(f"🔍 3단계 검증: AI 레벨 검증")
        stage3_valid, stage3_error, stage3_message = self.ai_validator.comprehensive_ai_validation(user_input, agent1_result)

        result.stage_results['stage3'] = {
            'valid': stage3_valid,
            'error_type': stage3_error.value,
            'message': stage3_message
        }

        if not stage3_valid:
            result.is_valid = False
            result.failed_stage = ValidationStage.STAGE_3_AI_LEVEL
            result.error_code = stage3_error.value
            result.error_message = stage3_message
            result.suggestion = self.ai_validator.get_ai_validation_suggestions(stage3_error, user_input)
            return result

        print(f"✅ 3단계 통과: {stage3_message}")

        # 모든 단계 통과
        result.is_valid = True
        result.failed_stage = ValidationStage.ALL_STAGES_PASSED
        result.error_code = "VALID"
        result.error_message = "모든 검증 단계를 통과했습니다."
        result.suggestion = ""

        return result

    def validate_quick(self, user_input: str) -> Tuple[bool, str, str]:
        """
        빠른 검증 (Agent1 분석 전)
        server.py에서 초기 차단용

        Returns:
            Tuple[bool, str, str]: (유효여부, 에러코드, 에러메시지)
        """
        # 1단계만 실행
        stage1_valid, stage1_error, stage1_message = self.language_detector.validate_language_input(user_input)
        if not stage1_valid:
            return False, stage1_error.value, stage1_message

        # 2단계 실행
        stage2_valid, stage2_error, stage2_message = self.korean_validator.validate_korean_quality(user_input)
        if not stage2_valid:
            return False, stage2_error.value, stage2_message

        # AI 레벨 사전 검증만 실행 (빠른 차단)
        pre_ai_valid, pre_ai_error, pre_ai_message = self.ai_validator.validate_pre_ai_analysis(user_input)
        if not pre_ai_valid:
            return False, pre_ai_error.value, pre_ai_message

        return True, "VALID", "빠른 검증 통과"

    def get_validation_statistics(self, user_input: str) -> Dict:
        """입력에 대한 상세 검증 통계"""
        stats = {
            'input_length': len(user_input),
            'language_stats': self.language_detector.get_text_statistics(user_input),
            'korean_analysis': self.korean_validator.analyze_korean_text(user_input),
            'ai_analysis': self.ai_validator.analyze_input_quality(user_input)
        }
        return stats

    def create_error_response_for_api(self, validation_result: ComprehensiveValidationResult) -> Dict:
        """
        API 응답용 에러 객체 생성 (server.py 호환)
        """
        error_response = {
            "error": {
                "code": validation_result.error_code,
                "message": validation_result.error_message
            },
            "validation_details": {
                "failed_stage": validation_result.failed_stage.value if validation_result.failed_stage else None,
                "suggestion": validation_result.suggestion,
                "stage_results": validation_result.stage_results
            },
            "timestamp": None  # server.py에서 설정
        }
        return error_response


# 전역 인스턴스 (싱글톤)
_comprehensive_validator = None

def get_comprehensive_validator() -> ComprehensiveValidator:
    """싱글톤 통합 검증기 반환"""
    global _comprehensive_validator
    if _comprehensive_validator is None:
        _comprehensive_validator = ComprehensiveValidator()
    return _comprehensive_validator

# 편의 함수들 (server.py에서 사용)
def validate_input_quick(user_input: str) -> Tuple[bool, str, str]:
    """빠른 입력 검증 (server.py용)"""
    validator = get_comprehensive_validator()
    return validator.validate_quick(user_input)

def validate_input_full(user_input: str, agent1_result: Optional[Dict] = None) -> ComprehensiveValidationResult:
    """전체 입력 검증"""
    validator = get_comprehensive_validator()
    return validator.validate_input_comprehensive(user_input, agent1_result)

def is_input_valid(user_input: str) -> bool:
    """간단한 입력 유효성 확인"""
    valid, _, _ = validate_input_quick(user_input)
    return valid


# server.py 호환성을 위한 함수 (기존 is_meaningful_text 대체)
def is_meaningful_text_advanced(user_input: str) -> bool:
    """
    기존 is_meaningful_text 함수의 고급 버전
    server.py에서 직접 사용 가능
    """
    return is_input_valid(user_input)


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 3단계 통합 입력 검증 시스템 테스트")
    print("=" * 60)

    validator = ComprehensiveValidator()

    # Issue 5에서 언급한 모든 문제 케이스들
    test_cases = [
        # 완전히 무의미한 입력들 (1단계에서 차단)
        ("ff", "영어만 입력"),
        ("asdfasdfiasj", "무의미한 영어"),
        ("1234567", "숫자만 입력"),
        ("!!!@@@###", "특수문자만"),

        # 한국어 품질 문제 (2단계에서 차단)
        ("ㅁㄴㅇㄹ", "자음/모음만"),
        ("ㅋㅋㅋㅋㅋ", "무의미한 패턴"),
        ("안녕ㅎ세요", "불완전한 한글"),

        # 전화번호/영어 이름 문제 (1단계에서 차단)
        ("010-1234-5678", "전화번호만"),
        ("John Smith", "영어 이름만"),
        ("010-1234-5678 John Smith", "한국어 비율 부족"),

        # AI 해석 어려운 케이스 (3단계에서 차단)
        ("예약", "정보 부족"),
        ("안내", "의미 모호"),

        # 경계선 케이스들 (일부는 통과, 일부는 차단)
        ("John Smith님 예약 확인", "혼합 - 한국어 포함"),
        ("010-1234-5678 연락처 안내", "혼합 - 목적 포함"),

        # 정상 케이스들 (모든 단계 통과)
        ("카페 예약 확인 메시지 만들어줘", "정상 입력"),
        ("할인 쿠폰 발급 안내", "정상 입력"),
        ("내일 오후 2시 병원 예약 확인", "상세 정보 포함"),
        ("고객님께 배송 완료 알림 보내드리겠습니다", "완전한 문장"),
    ]

    for i, (test_input, description) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. [{description}] '{test_input}'")

        # 통합 검증 실행
        result = validator.validate_input_comprehensive(test_input)

        # 결과 출력
        if result.is_valid:
            print(f"    🎉 전체 통과: {result.error_message}")
        else:
            print(f"    ❌ 차단됨 [{result.failed_stage.value}]: {result.error_message}")
            print(f"    💡 제안: {result.suggestion}")

        # 단계별 상세 결과
        for stage, stage_result in result.stage_results.items():
            status = "✅" if stage_result['valid'] else "❌"
            print(f"        {stage}: {status} {stage_result['message']}")

    print(f"\n🔧 3단계 통합 입력 검증 시스템 테스트 완료")
    print(f"🎯 Issue 5 무의미한 입력 템플릿 생성 문제 해결!")