#!/usr/bin/env python3
"""
언어 감지 및 입력 검증 시스템
Issue 5 해결: 무의미한 입력 템플릿 생성 문제 - 1단계 언어 감지
"""
import re
from typing import Tuple, Dict
from enum import Enum

class ValidationError(Enum):
    """입력 검증 오류 타입"""
    ENGLISH_ONLY = "ENGLISH_ONLY"
    NUMBERS_ONLY = "NUMBERS_ONLY"
    SPECIAL_CHARS_ONLY = "SPECIAL_CHARS_ONLY"
    LOW_KOREAN_RATIO = "LOW_KOREAN_RATIO"
    REPEATED_PATTERN = "REPEATED_PATTERN"
    TOO_SHORT = "TOO_SHORT"
    PHONE_NUMBER_ONLY = "PHONE_NUMBER_ONLY"
    VALID = "VALID"

class LanguageDetector:
    """언어 감지 및 입력 검증 클래스"""

    def __init__(self):
        """언어 감지기 초기화"""
        # 한국어 최소 비율 임계값 (30%)
        self.korean_ratio_threshold = 0.3

        # 최소 입력 길이
        self.min_input_length = 2

        # 반복 패턴 감지 설정
        self.max_repeated_ratio = 0.7  # 70% 이상 같은 문자면 반복으로 판정

        print("언어 감지 시스템 초기화 완료")

    def detect_korean_ratio(self, text: str) -> float:
        """한국어 문자 비율 계산"""
        if not text or not text.strip():
            return 0.0

        # 공백 제거하고 한글 문자 개수 계산
        text_no_space = re.sub(r'\s+', '', text)
        if len(text_no_space) == 0:
            return 0.0

        korean_chars = len(re.findall(r'[가-힣]', text_no_space))
        total_chars = len(text_no_space)

        return korean_chars / total_chars if total_chars > 0 else 0.0

    def is_english_only(self, text: str) -> bool:
        """영어만 입력인지 확인"""
        text_clean = re.sub(r'[\s\-\.]', '', text.strip())
        if not text_clean:
            return False
        return re.match(r'^[a-zA-Z]+$', text_clean) is not None

    def is_numbers_only(self, text: str) -> bool:
        """숫자만 입력인지 확인"""
        text_clean = re.sub(r'[\s\-\.]', '', text.strip())
        if not text_clean:
            return False
        return re.match(r'^[\d]+$', text_clean) is not None

    def is_special_chars_only(self, text: str) -> bool:
        """특수문자만 입력인지 확인"""
        text_clean = re.sub(r'\s+', '', text.strip())
        if not text_clean:
            return False
        return re.match(r'^[^\w가-힣]+$', text_clean) is not None

    def is_phone_number_only(self, text: str) -> bool:
        """전화번호만 입력인지 확인"""
        text_clean = text.strip()
        # 한국 전화번호 패턴들
        phone_patterns = [
            r'^01[0-9]-?\d{3,4}-?\d{4}$',  # 휴대폰
            r'^0[2-9][0-9]?-?\d{3,4}-?\d{4}$',  # 지역번호
            r'^\d{2,3}-?\d{3,4}-?\d{4}$',  # 일반적인 패턴
        ]

        for pattern in phone_patterns:
            if re.match(pattern, text_clean):
                return True
        return False

    def detect_repeated_pattern(self, text: str) -> bool:
        """반복 패턴 감지 (aaaaaa, 111111 등)"""
        text_clean = re.sub(r'\s+', '', text.strip())
        if len(text_clean) < 3:
            return False

        # 유니크한 문자 개수 vs 전체 길이 비율
        unique_chars = len(set(text_clean))
        total_chars = len(text_clean)

        # 문자 종류가 너무 적으면 반복 패턴으로 판정
        if unique_chars <= 2 and total_chars > 3:
            return True

        # 같은 문자가 70% 이상이면 반복
        char_counts = {}
        for char in text_clean:
            char_counts[char] = char_counts.get(char, 0) + 1

        max_count = max(char_counts.values())
        if max_count / total_chars > self.max_repeated_ratio:
            return True

        return False

    def get_text_statistics(self, text: str) -> Dict:
        """텍스트 통계 정보 반환"""
        if not text:
            return {
                'korean_ratio': 0.0,
                'english_ratio': 0.0,
                'number_ratio': 0.0,
                'special_ratio': 0.0,
                'length': 0,
                'unique_chars': 0
            }

        text_no_space = re.sub(r'\s+', '', text)
        total_len = len(text_no_space)

        if total_len == 0:
            return {
                'korean_ratio': 0.0,
                'english_ratio': 0.0,
                'number_ratio': 0.0,
                'special_ratio': 0.0,
                'length': 0,
                'unique_chars': 0
            }

        korean_count = len(re.findall(r'[가-힣]', text_no_space))
        english_count = len(re.findall(r'[a-zA-Z]', text_no_space))
        number_count = len(re.findall(r'[\d]', text_no_space))
        special_count = total_len - korean_count - english_count - number_count

        return {
            'korean_ratio': korean_count / total_len,
            'english_ratio': english_count / total_len,
            'number_ratio': number_count / total_len,
            'special_ratio': special_count / total_len,
            'length': total_len,
            'unique_chars': len(set(text_no_space))
        }

    def validate_language_input(self, text: str) -> Tuple[bool, ValidationError, str]:
        """
        종합적인 언어 입력 검증

        Returns:
            Tuple[bool, ValidationError, str]: (유효여부, 오류타입, 오류메시지)
        """
        if not text or not text.strip():
            return False, ValidationError.TOO_SHORT, "입력이 비어있습니다."

        text_clean = text.strip()

        # 1. 길이 검증
        if len(text_clean) < self.min_input_length:
            return False, ValidationError.TOO_SHORT, f"최소 {self.min_input_length}자 이상 입력해주세요."

        # 2. 전화번호만 입력 차단
        if self.is_phone_number_only(text_clean):
            return False, ValidationError.PHONE_NUMBER_ONLY, "전화번호만으로는 알림톡을 생성할 수 없습니다. 목적을 함께 입력해주세요."

        # 3. 영어만 입력 차단
        if self.is_english_only(text_clean):
            return False, ValidationError.ENGLISH_ONLY, "한국어로 입력해주세요. 영어만으로는 알림톡을 생성할 수 없습니다."

        # 4. 숫자만 입력 차단
        if self.is_numbers_only(text_clean):
            return False, ValidationError.NUMBERS_ONLY, "숫자만으로는 알림톡을 생성할 수 없습니다. 구체적인 내용을 입력해주세요."

        # 5. 특수문자만 입력 차단
        if self.is_special_chars_only(text_clean):
            return False, ValidationError.SPECIAL_CHARS_ONLY, "특수문자만으로는 알림톡을 생성할 수 없습니다."

        # 6. 반복 패턴 차단
        if self.detect_repeated_pattern(text_clean):
            return False, ValidationError.REPEATED_PATTERN, "반복되는 문자나 패턴은 사용할 수 없습니다. 의미있는 내용을 입력해주세요."

        # 7. 한국어 비율 검증 (가장 중요)
        korean_ratio = self.detect_korean_ratio(text_clean)
        if korean_ratio < self.korean_ratio_threshold:
            return False, ValidationError.LOW_KOREAN_RATIO, f"한국어 비율이 {korean_ratio:.1%}로 너무 낮습니다. 최소 {self.korean_ratio_threshold:.0%} 이상의 한국어를 포함해주세요."

        # 모든 검증 통과
        return True, ValidationError.VALID, "유효한 입력입니다."

    def get_validation_suggestion(self, text: str, error_type: ValidationError) -> str:
        """검증 오류에 따른 개선 제안"""
        suggestions = {
            ValidationError.ENGLISH_ONLY: "예시: 'John Smith 예약 확인' → '존 스미스님 예약 확인 안내'",
            ValidationError.NUMBERS_ONLY: "예시: '010-1234-5678' → '고객님께 연락처 확인 안내'",
            ValidationError.PHONE_NUMBER_ONLY: "예시: '010-1234-5678' → '예약 확인 연락드릴 예정입니다'",
            ValidationError.LOW_KOREAN_RATIO: "예시: 'John Smith 010-1234-5678' → '고객님 예약 확인 안내'",
            ValidationError.REPEATED_PATTERN: "예시: 'aaaa' → '안내 메시지 작성해주세요'",
            ValidationError.SPECIAL_CHARS_ONLY: "예시: '!!!' → '중요 공지사항 안내'",
            ValidationError.TOO_SHORT: "더 구체적인 내용을 입력해주세요."
        }
        return suggestions.get(error_type, "올바른 한국어로 다시 입력해주세요.")


# 전역 인스턴스 (싱글톤)
_language_detector = None

def get_language_detector() -> LanguageDetector:
    """싱글톤 언어 감지기 반환"""
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
    return _language_detector

# 편의 함수들
def validate_input_language(text: str) -> Tuple[bool, ValidationError, str]:
    """입력 언어 검증 편의 함수"""
    detector = get_language_detector()
    return detector.validate_language_input(text)

def get_korean_ratio(text: str) -> float:
    """한국어 비율 계산 편의 함수"""
    detector = get_language_detector()
    return detector.detect_korean_ratio(text)

def is_valid_korean_input(text: str) -> bool:
    """간단한 한국어 입력 유효성 확인"""
    valid, _, _ = validate_input_language(text)
    return valid


if __name__ == "__main__":
    # 테스트 코드
    print("언어 감지 시스템 테스트")
    print("=" * 50)

    detector = LanguageDetector()

    # Issue 5에서 언급한 문제 입력들 + 전화번호/영어 이름 케이스
    test_cases = [
        # 기존 무의미한 입력들
        "ff",
        "asdfasdfiasj",
        "ㅁㄴㅇㄹ",
        "aaaaaa",
        "1234567",
        "!!!@@@###",

        # 전화번호 + 영어 이름 케이스들
        "010-1234-5678",
        "John Smith",
        "010-1234-5678 John Smith",
        "John Smith 010-1234-5678",
        "My name is John Smith and my phone is 010-1234-5678",

        # 혼합 케이스들 (일부는 통과해야 함)
        "010-1234-5678 David Kim 예약",
        "예약 John Smith 010-1234-5678",
        "김철수님 010-1234-5678 연락드리겠습니다",

        # 정상적인 입력들 (통과해야 함)
        "카페 예약 확인 메시지 만들어줘",
        "고객님께 할인 쿠폰 발급 안내"
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i:2d}. 테스트: '{test_input}'")

        # 통계 정보
        stats = detector.get_text_statistics(test_input)
        print(f"    통계: 한글{stats['korean_ratio']:.1%} 영어{stats['english_ratio']:.1%} 숫자{stats['number_ratio']:.1%}")

        # 검증 결과
        is_valid, error_type, message = detector.validate_language_input(test_input)
        status = "통과" if is_valid else "차단"
        print(f"    결과: {status} - {message}")

        if not is_valid:
            suggestion = detector.get_validation_suggestion(test_input, error_type)
            print(f"    제안: {suggestion}")

    print(f"\n언어 감지 시스템 테스트 완료")