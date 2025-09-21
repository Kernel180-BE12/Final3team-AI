#!/usr/bin/env python3
"""
한국어 품질 검증 시스템 (KoNLPy 기반)
Issue 5 해결: 무의미한 입력 템플릿 생성 문제 - 2단계 한국어 품질 검증
"""
import re
from typing import Tuple, List, Dict
from enum import Enum

class KoreanValidationError(Enum):
    """한국어 검증 오류 타입"""
    JAMO_ONLY = "JAMO_ONLY"  # 자음/모음만 있음
    NO_MEANINGFUL_WORDS = "NO_MEANINGFUL_WORDS"  # 의미있는 단어 없음
    LOW_WORD_RATIO = "LOW_WORD_RATIO"  # 단어 비율 너무 낮음
    NONSENSE_PATTERN = "NONSENSE_PATTERN"  # 무의미한 패턴
    INCOMPLETE_WORDS = "INCOMPLETE_WORDS"  # 불완전한 단어들
    VALID = "VALID"

class KoreanValidator:
    """KoNLPy 기반 한국어 품질 검증 클래스"""

    def __init__(self):
        """한국어 검증기 초기화"""
        self.okt = None
        self._init_morphological_analyzer()
        self._init_korean_dictionaries()

        # 검증 임계값들
        self.min_meaningful_word_ratio = 0.3  # 최소 의미있는 단어 비율 30%
        self.min_word_count = 1  # 최소 의미있는 단어 개수
        self.max_jamo_ratio = 0.5  # 최대 자음/모음 비율 50%

        print("🔍 한국어 품질 검증 시스템 초기화 완료")

    def _init_morphological_analyzer(self):
        """형태소 분석기 초기화 (지연 로딩)"""
        try:
            from konlpy.tag import Okt
            self.okt = Okt()
            print("✅ KoNLPy 형태소 분석기 로드 완료")
        except ImportError:
            print("⚠️ KoNLPy가 설치되지 않았습니다. 기본 검증 모드로 동작")
            self.okt = None
        except Exception as e:
            print(f"⚠️ 형태소 분석기 초기화 실패: {e}")
            self.okt = None

    def _init_korean_dictionaries(self):
        """한국어 사전 및 패턴 초기화"""
        # 기본 한국어 단어 사전 (알림톡 도메인 특화)
        self.basic_korean_words = {
            # 알림톡 핵심 키워드
            '예약', '확인', '안내', '알림', '공지', '변경', '취소', '완료',
            '결제', '주문', '배송', '발송', '도착', '출발', '접수',
            '회원', '가입', '등록', '신청', '참가', '참여', '이용',
            '할인', '혜택', '쿠폰', '포인트', '적립', '사용', '만료',
            '이벤트', '행사', '모임', '만남', '방문', '시간', '장소',
            '고객', '님', '분', '씨', '선생님', '사장님', '회원님',

            # 기본 명사들
            '카페', '식당', '병원', '학원', '회사', '학교', '집', '사무실',
            '상품', '서비스', '제품', '물품', '상황', '문제', '해결',
            '연락', '전화', '메시지', '문자', '이메일', '홈페이지',
            '오늘', '내일', '어제', '시간', '날짜', '주말', '평일',

            # 기본 동사/형용사 어간
            '보내', '받', '드리', '주', '해', '되', '있', '없', '같', '다른',
            '좋', '나쁘', '중요', '필요', '가능', '불가능', '편리', '안전'
        }

        # 무의미한 패턴들
        self.nonsense_patterns = [
            r'^ㅋ+$',  # ㅋㅋㅋ
            r'^ㅎ+$',  # ㅎㅎㅎ
            r'^ㅠ+$',  # ㅠㅠㅠ
            r'^ㅜ+$',  # ㅜㅜㅜ
            r'^[ㅏ-ㅣ]+$',  # 모음만
            r'^[ㄱ-ㅎ]+$',  # 자음만
            r'^[ㅏ-ㅣㄱ-ㅎ]+$',  # 자음모음 조합만
        ]

        # 불완전한 한글 패턴 (자음/모음 혼재)
        self.incomplete_hangul_patterns = [
            r'[ㄱ-ㅎㅏ-ㅣ]',  # 자음이나 모음이 단독으로 존재
        ]

    def detect_jamo_ratio(self, text: str) -> float:
        """자음/모음 단독 문자 비율 계산"""
        if not text:
            return 0.0

        # 자음/모음 패턴 (완성되지 않은 한글)
        jamo_count = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ]', text))
        total_korean = len(re.findall(r'[가-힣ㄱ-ㅎㅏ-ㅣ]', text))

        return jamo_count / total_korean if total_korean > 0 else 0.0

    def check_nonsense_patterns(self, text: str) -> bool:
        """무의미한 패턴 검사"""
        text_clean = re.sub(r'\s+', '', text.strip())

        for pattern in self.nonsense_patterns:
            if re.match(pattern, text_clean):
                return True
        return False

    def extract_meaningful_words_fallback(self, text: str) -> List[str]:
        """폴백: KoNLPy 없을 때 기본 단어 추출"""
        meaningful_words = []

        # 기본 사전에서 찾기
        for word in self.basic_korean_words:
            if word in text:
                meaningful_words.append(word)

        # 한글 2자 이상 단어 추출 (휴리스틱)
        korean_words = re.findall(r'[가-힣]{2,}', text)
        meaningful_words.extend(korean_words)

        return list(set(meaningful_words))

    def extract_meaningful_words_konlpy(self, text: str) -> List[str]:
        """KoNLPy 기반 의미있는 단어 추출"""
        try:
            # 형태소 분석
            morphs = self.okt.morphs(text)
            pos_tags = self.okt.pos(text)
            nouns = self.okt.nouns(text)

            meaningful_words = []

            # 1. 명사 (2글자 이상)
            meaningful_nouns = [noun for noun in nouns if len(noun) >= 2]
            meaningful_words.extend(meaningful_nouns)

            # 2. 의미있는 품사 (동사, 형용사, 부사)
            meaningful_pos = ['Verb', 'Adjective', 'Adverb']
            for morph, pos in pos_tags:
                if pos in meaningful_pos and len(morph) >= 2:
                    meaningful_words.append(morph)

            # 3. 기본 사전과 교차 확인
            for word in self.basic_korean_words:
                if word in text:
                    meaningful_words.append(word)

            return list(set(meaningful_words))

        except Exception as e:
            print(f"⚠️ KoNLPy 분석 중 오류: {e}")
            return self.extract_meaningful_words_fallback(text)

    def extract_meaningful_words(self, text: str) -> List[str]:
        """의미있는 단어 추출 (KoNLPy 우선, 폴백 지원)"""
        if self.okt:
            return self.extract_meaningful_words_konlpy(text)
        else:
            return self.extract_meaningful_words_fallback(text)

    def calculate_word_quality_score(self, text: str) -> Dict:
        """단어 품질 점수 계산"""
        if not text or not text.strip():
            return {
                'meaningful_words': [],
                'word_count': 0,
                'total_morphs': 0,
                'word_ratio': 0.0,
                'quality_score': 0.0
            }

        # 의미있는 단어 추출
        meaningful_words = self.extract_meaningful_words(text)

        # 전체 형태소 개수 (KoNLPy가 있으면 정확히, 없으면 추정)
        if self.okt:
            try:
                total_morphs = len(self.okt.morphs(text))
            except:
                total_morphs = len(text.split())
        else:
            total_morphs = len(text.split())

        # 비율 계산
        word_count = len(meaningful_words)
        word_ratio = word_count / total_morphs if total_morphs > 0 else 0.0

        # 품질 점수 계산 (0.0 ~ 1.0)
        quality_score = min(word_ratio * 2, 1.0)  # 50% 비율이면 만점

        return {
            'meaningful_words': meaningful_words,
            'word_count': word_count,
            'total_morphs': total_morphs,
            'word_ratio': word_ratio,
            'quality_score': quality_score
        }

    def validate_korean_quality(self, text: str) -> Tuple[bool, KoreanValidationError, str]:
        """
        한국어 품질 종합 검증

        Returns:
            Tuple[bool, KoreanValidationError, str]: (유효여부, 오류타입, 오류메시지)
        """
        if not text or not text.strip():
            return False, KoreanValidationError.NO_MEANINGFUL_WORDS, "입력이 비어있습니다."

        text_clean = text.strip()

        # 1. 자음/모음만 있는지 확인
        jamo_ratio = self.detect_jamo_ratio(text_clean)
        if jamo_ratio > self.max_jamo_ratio:
            return False, KoreanValidationError.JAMO_ONLY, f"자음/모음만 있는 입력입니다 ({jamo_ratio:.1%}). 완성된 한글을 사용해주세요."

        # 2. 무의미한 패턴 확인
        if self.check_nonsense_patterns(text_clean):
            return False, KoreanValidationError.NONSENSE_PATTERN, "무의미한 반복 패턴입니다. 구체적인 내용을 입력해주세요."

        # 3. 의미있는 단어 분석
        word_analysis = self.calculate_word_quality_score(text_clean)

        # 4. 의미있는 단어 개수 확인
        if word_analysis['word_count'] < self.min_word_count:
            return False, KoreanValidationError.NO_MEANINGFUL_WORDS, f"의미있는 단어가 없습니다. 구체적인 내용을 입력해주세요."

        # 5. 의미있는 단어 비율 확인
        if word_analysis['word_ratio'] < self.min_meaningful_word_ratio:
            return False, KoreanValidationError.LOW_WORD_RATIO, f"의미있는 단어 비율이 {word_analysis['word_ratio']:.1%}로 너무 낮습니다."

        # 6. 불완전한 단어 확인
        incomplete_count = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ]', text_clean))
        total_korean = len(re.findall(r'[가-힣ㄱ-ㅎㅏ-ㅣ]', text_clean))
        if total_korean > 0 and incomplete_count / total_korean > 0.3:
            return False, KoreanValidationError.INCOMPLETE_WORDS, "불완전한 한글이 포함되어 있습니다."

        # 모든 검증 통과
        return True, KoreanValidationError.VALID, "한국어 품질이 양호합니다."

    def get_quality_suggestions(self, text: str, error_type: KoreanValidationError) -> str:
        """품질 오류에 따른 개선 제안"""
        suggestions = {
            KoreanValidationError.JAMO_ONLY: "예시: 'ㅁㄴㅇㄹ' → '안내 메시지를 작성해주세요'",
            KoreanValidationError.NO_MEANINGFUL_WORDS: "구체적인 알림톡 내용을 입력해주세요. 예: '예약 확인', '할인 안내' 등",
            KoreanValidationError.LOW_WORD_RATIO: "더 많은 의미있는 단어를 포함해주세요.",
            KoreanValidationError.NONSENSE_PATTERN: "예시: 'ㅋㅋㅋ' → '재미있는 이벤트 안내'",
            KoreanValidationError.INCOMPLETE_WORDS: "완성된 한글 단어를 사용해주세요."
        }
        return suggestions.get(error_type, "올바른 한국어로 다시 입력해주세요.")

    def analyze_korean_text(self, text: str) -> Dict:
        """한국어 텍스트 상세 분석"""
        if not text:
            return {}

        analysis = {
            'jamo_ratio': self.detect_jamo_ratio(text),
            'has_nonsense_pattern': self.check_nonsense_patterns(text),
            'word_analysis': self.calculate_word_quality_score(text),
            'konlpy_available': self.okt is not None
        }

        # 검증 결과 추가
        is_valid, error_type, message = self.validate_korean_quality(text)
        analysis['is_valid'] = is_valid
        analysis['error_type'] = error_type.value
        analysis['error_message'] = message

        return analysis


# 전역 인스턴스 (싱글톤)
_korean_validator = None

def get_korean_validator() -> KoreanValidator:
    """싱글톤 한국어 검증기 반환"""
    global _korean_validator
    if _korean_validator is None:
        _korean_validator = KoreanValidator()
    return _korean_validator

# 편의 함수들
def validate_korean_quality(text: str) -> Tuple[bool, KoreanValidationError, str]:
    """한국어 품질 검증 편의 함수"""
    validator = get_korean_validator()
    return validator.validate_korean_quality(text)

def extract_meaningful_words(text: str) -> List[str]:
    """의미있는 단어 추출 편의 함수"""
    validator = get_korean_validator()
    return validator.extract_meaningful_words(text)

def is_quality_korean(text: str) -> bool:
    """간단한 한국어 품질 확인"""
    valid, _, _ = validate_korean_quality(text)
    return valid


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 한국어 품질 검증 시스템 테스트")
    print("=" * 50)

    validator = KoreanValidator()

    # 다양한 한국어 품질 테스트 케이스
    test_cases = [
        # 자음/모음만 있는 케이스
        "ㅁㄴㅇㄹ",
        "ㅋㅋㅋㅋ",
        "ㅠㅠㅠ",
        "ㄱㄴㄷㄹ",

        # 무의미한 패턴
        "ㅎㅎㅎㅎㅎ",
        "ㅏㅏㅏㅏ",

        # 불완전한 한글
        "안녕ㅎ세요",
        "예약ㅇ 확인",

        # 품질이 낮은 한국어
        "ㅇ예약",
        "안ㄴ내",

        # 정상적인 한국어 (통과해야 함)
        "예약 확인 안내",
        "할인 쿠폰 발급",
        "카페 예약 확인 메시지 만들어줘",
        "고객님께 연락드리겠습니다",
        "내일 오후 2시 방문 예정입니다",

        # 혼합 케이스 (한국어 포함)
        "John Smith님 예약 확인",
        "010-1234-5678 연락처 안내",
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i:2d}. 테스트: '{test_input}'")

        # 상세 분석
        analysis = validator.analyze_korean_text(test_input)

        print(f"    자음/모음 비율: {analysis.get('jamo_ratio', 0):.1%}")
        print(f"    의미있는 단어: {analysis.get('word_analysis', {}).get('meaningful_words', [])}")
        print(f"    단어 품질 점수: {analysis.get('word_analysis', {}).get('quality_score', 0):.2f}")

        # 검증 결과
        is_valid = analysis.get('is_valid', False)
        status = "✅ 통과" if is_valid else "❌ 차단"
        message = analysis.get('error_message', '')
        print(f"    결과: {status} - {message}")

        if not is_valid:
            error_type = KoreanValidationError(analysis.get('error_type', 'VALID'))
            suggestion = validator.get_quality_suggestions(test_input, error_type)
            print(f"    제안: {suggestion}")

    print(f"\n🔧 한국어 품질 검증 시스템 테스트 완료")