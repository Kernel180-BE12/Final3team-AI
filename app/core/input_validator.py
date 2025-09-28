#!/usr/bin/env python3
"""
Input Validator - 입력 검증 전담 클래스
Agent1에서 분리된 입력 검증 로직을 담당
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """검증 결과를 담는 데이터 클래스"""
    is_valid: bool
    message: str
    category: Optional[str] = None
    confidence: float = 1.0
    detected_pattern: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class InputValidator:
    """입력 검증 전담 클래스"""

    def __init__(self):
        """입력 검증기 초기화"""
        self._setup_validation_patterns()

    def _setup_validation_patterns(self):
        """검증 패턴들을 설정"""
        # 1. 강력한 부적절 패턴 (즉시 차단급)
        self.high_risk_patterns = {
            '요리_레시피': [
                '김치찌개', '된장찌개', '라면', '파스타', '볶음밥',
                '레시피', '요리법', '만드는 방법', '조리법', '음식 만들기',
                '재료', '양념', '조미료', '맛있게', '달콤한', '매운'
            ],
            '개인_연애_상담': [
                '여자친구', '남자친구', '연애', '데이트', '썸', '고백',
                '이별', '짝사랑', '결혼 준비', '소개팅', '애인',
                '헤어진', '연인', '커플', '사랑해', '좋아하는 사람'
            ],
            '학업_개인지도': [
                '숙제', '과제', '시험 문제', '문제 풀이', '정답',
                '수학', '영어', '국어', '과학', '공부 방법', '한국 지리', '세계 지리', '세계사',
                '한국사', '과외', '대학교', '고등학교'
            ],
            '개인_일상_잡담': [
                '심심해', '졸려', '피곤해', '배고파', '목말라',
                '날씨', '비 와', '눈 와', '더워', '추워',
                '아침', '점심', '저녁', '밤늦게'
            ]
        }

        # 2. 중간 위험 패턴
        self.medium_risk_patterns = {
            '오락_게임': [
                '오락', '놀이', 'pc방', '노래방', '영화 추천',
                '넷플릭스', '유튜브', '만화', '웹툰'
            ],
            '개인적_용도': [
                '개인적으로', '나는', '내가', '나에게', '개인용',
                '친구', '가족', '엄마', '아빠', '혼자서'
            ]
        }

        # 3. 경미한 잡담 신호
        self.casual_patterns = [
            'ㅋㅋ', 'ㅎㅎ', 'ㅠㅠ', 'ㅜㅜ', '^^', '>_<',
            '하하', '호호', '히히', '크크'
        ]

        # 4. 의미없는 입력 패턴
        self.meaningless_patterns = [
            r'^[ㄱ-ㅎㅏ-ㅣ]{2,}$',  # 자음모음만
            r'^[?!]{2,}$',  # 물음표/느낌표만
            r'^[0-9]{1,3}$',  # 단순 숫자
            r'^(안녕|하이|hi|hello)$',  # 단순 인사
        ]

        # 5. 비즈니스 키워드 (긍정적 점수)
        self.business_keywords = {
            '상업_서비스': [
                '예약', '예약확인', '예약취소', '시간예약', '서비스',
                '이벤트', '할인', '프로모션', '혜택', '쿠폰', '적립',
                '매장', '지점', '본점', '영업시간', '운영시간'
            ],
            '알림_안내': [
                '알림', '안내', '공지', '변경', '업데이트', '소식',
                '정보', '일정', '스케줄', '계획', '프로그램',
                '등록', '신청', '접수', '마감', '종료'
            ],
            '고객_관리': [
                '고객', '회원', '사용자', '이용자', '참여자',
                '신규', '기존', '가입', '탈퇴', '등급', '포인트'
            ]
        }

    def validate_language(self, text: str) -> ValidationResult:
        """언어 검증 (한국어 우선)"""
        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                message="텍스트가 비어있습니다.",
                category="empty_input"
            )

        text = text.strip()

        # 영어만 있는지 체크
        english_only = bool(re.match(r'^[a-zA-Z\s\.,!?]+$', text))
        if english_only:
            return ValidationResult(
                is_valid=False,
                message="Please enter in Korean. English-only input cannot generate KakaoTalk templates.",
                category="english_only",
                detected_pattern="english_only"
            )

        # 한국어 포함 여부 체크
        korean_chars = bool(re.search(r'[가-힣]', text))
        if not korean_chars and len(text) > 10:
            return ValidationResult(
                is_valid=False,
                message="한국어를 포함해서 입력해주세요.",
                category="no_korean",
                confidence=0.8
            )

        return ValidationResult(
            is_valid=True,
            message="언어 검증 통과",
            category="language_valid"
        )

    def validate_length_requirements(self, text: str) -> ValidationResult:
        """길이 요구사항 검증"""
        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                message="텍스트가 비어있습니다.",
                category="empty_input"
            )

        text = text.strip()

        # 최소 길이 체크
        if len(text) < 2:
            return ValidationResult(
                is_valid=False,
                message="더 구체적인 요청 내용을 입력해주세요 (최소 2글자 이상)",
                category="too_short"
            )

        # 최소 단어 수 체크
        words = text.split()
        if len(words) < 2:
            return ValidationResult(
                is_valid=False,
                message="더 구체적인 요청 내용을 입력해주세요 (최소 2단어 이상)",
                category="insufficient_words"
            )

        # 의미없는 기본값 체크
        invalid_inputs = ["샘플", "테스트", "없음", "기본값"]
        if text.lower() in invalid_inputs:
            return ValidationResult(
                is_valid=False,
                message="구체적인 템플릿 요청 내용을 입력해주세요",
                category="invalid_default"
            )

        return ValidationResult(
            is_valid=True,
            message="길이 요구사항 통과",
            category="length_valid"
        )

    def validate_business_context(self, user_input: str) -> ValidationResult:
        """
        비즈니스 알림톡에 적합하지 않은 요청을 필터링 (고도화된 룰 기반)
        """
        if not user_input or not user_input.strip():
            return ValidationResult(
                is_valid=False,
                message="빈 입력입니다. 알림톡 템플릿 요청 내용을 입력해주세요.",
                category="empty_input",
                detected_pattern="empty"
            )

        user_input_lower = user_input.lower().strip()

        # 스코어링 기반 필터링 (가중치 적용)
        score = 0
        detected_patterns = []

        # 1. 강력한 부적절 패턴 체크 (즉시 차단급)
        for category, patterns in self.high_risk_patterns.items():
            for pattern in patterns:
                if pattern in user_input_lower:
                    return ValidationResult(
                        is_valid=False,
                        message=f"비즈니스 알림톡에 적합하지 않은 요청입니다. '{pattern}' 관련 내용은 처리할 수 없습니다.",
                        category=category,
                        detected_pattern=pattern,
                        confidence=0.95
                    )

        # 2. 중간 위험 패턴 체크 (-5점)
        for category, patterns in self.medium_risk_patterns.items():
            pattern_count = 0
            for pattern in patterns:
                if pattern in user_input_lower:
                    pattern_count += 1
                    detected_patterns.append(f"medium_risk_{category}_{pattern}")
            if pattern_count >= 1:
                score -= 5 * pattern_count

        # 3. 경미한 잡담 신호 체크 (-2점)
        casual_count = sum(1 for pattern in self.casual_patterns if pattern in user_input_lower)
        if casual_count > 0:
            score -= 2 * casual_count
            detected_patterns.append(f"casual_chat_{casual_count}개")

        # 4. 의미없는 입력 체크 (-5점)
        for pattern in self.meaningless_patterns:
            if re.match(pattern, user_input_lower):
                score -= 5
                detected_patterns.append(f"meaningless_{pattern}")

        # 5. 비즈니스 키워드 체크 (+3점)
        business_score = 0
        for category, patterns in self.business_keywords.items():
            for pattern in patterns:
                if pattern in user_input_lower:
                    business_score += 3
                    detected_patterns.append(f"business_{category}_{pattern}")

        score += business_score

        # 6. 최종 판단
        if score < -5:
            return ValidationResult(
                is_valid=False,
                message="비즈니스 알림톡에 적합하지 않은 요청으로 판단됩니다. 구체적인 비즈니스 목적을 포함해서 다시 요청해주세요.",
                category="inappropriate_business",
                confidence=min(0.9, abs(score) / 10),
                details={
                    "score": score,
                    "detected_patterns": detected_patterns
                }
            )

        # 7. 긍정적 비즈니스 신호가 있으면 통과
        if business_score > 0 or score >= 0:
            return ValidationResult(
                is_valid=True,
                message="비즈니스 컨텍스트 검증 통과",
                category="business_appropriate",
                confidence=min(0.95, (score + 10) / 15),
                details={
                    "score": score,
                    "business_score": business_score,
                    "detected_patterns": detected_patterns
                }
            )

        # 8. 중립적인 경우도 통과 (관대한 정책)
        return ValidationResult(
            is_valid=True,
            message="중립적 요청으로 판단하여 처리합니다.",
            category="neutral_business",
            confidence=0.7,
            details={
                "score": score,
                "detected_patterns": detected_patterns
            }
        )

    def validate_comprehensive(self, text: str) -> ValidationResult:
        """종합적인 입력 검증 수행"""
        # 1. 언어 검증
        language_result = self.validate_language(text)
        if not language_result.is_valid:
            return language_result

        # 2. 길이 요구사항 검증
        length_result = self.validate_length_requirements(text)
        if not length_result.is_valid:
            return length_result

        # 3. 비즈니스 컨텍스트 검증
        business_result = self.validate_business_context(text)
        if not business_result.is_valid:
            return business_result

        # 모든 검증 통과
        return ValidationResult(
            is_valid=True,
            message="모든 입력 검증 통과",
            category="comprehensive_valid",
            confidence=min(language_result.confidence, length_result.confidence, business_result.confidence),
            details={
                "language_check": language_result.category,
                "length_check": length_result.category,
                "business_check": business_result.category
            }
        )