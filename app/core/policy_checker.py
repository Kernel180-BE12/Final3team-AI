#!/usr/bin/env python3
"""
Policy Checker - 정책 준수 검증 전담 클래스
Agent1에서 분리된 정책 및 비속어 검증 로직을 담당
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

from app.tools.profanity_checker import ProfanityChecker


@dataclass
class PolicyResult:
    """정책 검증 결과를 담는 데이터 클래스"""
    is_compliant: bool
    violations: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH, UNKNOWN
    has_profanity: bool = False
    profanity_details: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    details: Optional[Dict[str, Any]] = None


class PolicyChecker:
    """정책 준수 및 비속어 검증 전담 클래스"""

    def __init__(self):
        """정책 검증기 초기화"""
        self.project_root = Path(__file__).parent.parent.parent
        self.policy_content = ""
        self.profanity_checker = ProfanityChecker()
        self._load_policy_document()
        self._setup_policy_patterns()

    def _load_policy_document(self) -> None:
        """정책 문서 로드"""
        try:
            # 알림톡 가이드라인 파일 경로들
            policy_files = [
                self.project_root / "data" / "docs" / "advertise_info.md",
                self.project_root / "data" / "docs" / "guidelines.md",
                self.project_root / "data" / "docs" / "policy.txt"
            ]

            policy_texts = []
            for policy_file in policy_files:
                if policy_file.exists():
                    with open(policy_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        policy_texts.append(f"=== {policy_file.name} ===\n{content}")

            if policy_texts:
                self.policy_content = "\n\n".join(policy_texts)
                print("정책 문서 로드 완료 (알림톡 가이드 + 광고성 정보 정책)")
            else:
                self.policy_content = ""
                print("정책 문서를 찾을 수 없습니다.")

        except Exception as e:
            self.policy_content = ""
            print(f"정책 문서 로드 실패: {e}")

    def _setup_policy_patterns(self) -> None:
        """정책 검증 패턴들을 설정"""
        # 1. 광고성 정보 키워드
        self.ad_keywords = [
            "할인", "이벤트", "무료", "특가", "프로모션", "쿠폰",
            "세일", "혜택", "적립", "리워드"
        ]

        # 2. 프로모션/혜택 키워드
        self.promo_keywords = [
            "혜택", "이득", "받으세요", "드립니다", "증정", "경품", "당첨"
        ]

        # 3. 청소년 유해 정보
        self.youth_harmful_keywords = [
            "주류", "전자담배", "성인", "19세", "술", "담배", "성인용품"
        ]

        # 4. 금융 관련 키워드
        self.financial_keywords = [
            "결제", "송금", "납부", "대출", "투자", "주식",
            "보험", "펀드", "금융상품"
        ]

        # 5. 스팸성 표현
        self.spam_keywords = [
            "마지막", "즉시", "빨리", "한정", "선착순",
            "지금", "바로", "오늘만", "마감임박"
            # "긴급" 제거 - 정당한 서비스 장애/점검 안내에서 사용 가능
        ]

        # 6. 영업시간/휴무 안내
        self.business_hours_keywords = [
            "영업시간", "휴무", "운영시간", "오픈", "클로즈"
        ]

        # 7. 설문조사 관련
        self.survey_keywords = ["설문", "조사", "평가", "의견", "후기"]
        self.product_keywords = ["제품", "상품", "서비스", "브랜드"]

        # 8. 반려 사례 기반 패턴들 (카카오 알림톡 반려 데이터 기반)
        self.kakao_rejection_patterns = {
            "unsolicited_notice": [
                "생일 축하", "기념일 축하", "안부 인사", "앱 업데이트", "새 기능", "업데이트 안내"
            ],
            "promotional_reminder": [
                "쿠폰 리마인드", "포인트 만료", "혜택 소멸", "할인 혜택", "이벤트 참여", "프로모션 안내"
            ],
            "unclear_recipient_action": [
                "안내말씀", "알려드립니다", "변경사항", "개선사항", "소식"
            ],
            "simple_greetings": [
                "감사합니다", "축하드립니다", "안녕하세요만", "좋은 하루", "행복한", "즐거운"
            ]
        }

    def check_profanity(self, text: str) -> PolicyResult:
        """비속어 검증"""
        try:
            profanity_result = self.profanity_checker.check_text(text)

            if not profanity_result["is_clean"]:
                return PolicyResult(
                    is_compliant=False,
                    violations=[f"부적절한 언어가 감지되었습니다: {', '.join(profanity_result['detected_words'])}"],
                    risk_level="HIGH",
                    has_profanity=True,
                    profanity_details=profanity_result,
                    confidence=profanity_result.get("confidence", 0.95)
                )
            else:
                return PolicyResult(
                    is_compliant=True,
                    violations=[],
                    risk_level="LOW",
                    has_profanity=False,
                    profanity_details=profanity_result,
                    confidence=0.95
                )

        except Exception as e:
            print(f"비속어 검증 중 오류: {e}")
            return PolicyResult(
                is_compliant=True,
                violations=["비속어 검증 중 오류 발생 - 검사를 건너뜁니다"],
                risk_level="UNKNOWN",
                has_profanity=False,
                confidence=0.5
            )

    def check_policy_compliance(self, text: str, variables: Optional[Dict[str, str]] = None) -> PolicyResult:
        """
        정책 준수 여부 확인 (알림톡 가이드 + 광고성 정보 정책 기반)
        """
        violations = []
        text_lower = text.lower()

        if not self.policy_content:
            return PolicyResult(
                is_compliant=True,
                violations=["정책 문서를 로드할 수 없어 검사를 건너뜁니다."],
                risk_level="UNKNOWN",
                confidence=0.5
            )

        # 1. [광고] 표기가 있는 경우 - 알림톡은 광고성 메시지 발송 불가
        if "[광고]" in text or "(광고)" in text:
            violations.append("알림톡은 광고성 메시지 발송이 불가능합니다. [광고] 표기가 포함된 메시지는 발송 불가")

        # 2. 광고성 정보 판단 기준 검사
        has_ad_content = any(keyword in text_lower for keyword in self.ad_keywords)
        if has_ad_content:
            violations.append("광고성 내용으로 알림톡 발송이 불가능합니다")

        # 3. 광고성 정보로 판단되는 추가 키워드들
        promo_count = sum(1 for keyword in self.promo_keywords if keyword in text_lower)
        if promo_count >= 2:
            violations.append("프로모션/혜택 관련 내용으로 알림톡 발송이 불가능합니다")

        # 4. 청소년 유해 정보 검사
        if any(keyword in text_lower for keyword in self.youth_harmful_keywords):
            violations.append("청소년 유해 정보 관련 - 연령 인증 필요")

        # 5. 금융 관련 제한사항
        if any(keyword in text_lower for keyword in self.financial_keywords):
            violations.append("금융 관련 내용 - 정책 검토 필요")

        # 6. 개인정보 관련
        if "개인정보" in text_lower or "정보 수집" in text_lower:
            violations.append("개인정보 수집 시 동의 절차 필요")

        # 7. 스팸성 표현 검사
        spam_count = sum(1 for keyword in self.spam_keywords if keyword in text_lower)
        if spam_count >= 2:
            violations.append("스팸성 표현 과다 사용")

        # 8. 영업시간/휴무 안내
        if any(keyword in text_lower for keyword in self.business_hours_keywords):
            violations.append("영업시간/휴무 안내는 광고성 정보에 해당")

        # 9. 설문조사 관련
        if (any(s in text_lower for s in self.survey_keywords) and
            any(p in text_lower for p in self.product_keywords)):
            violations.append("제품 관련 설문조사는 광고성 정보에 해당할 수 있음")

        # 10. 추천/공유 이벤트 관련
        if "친구" in text_lower and ("추천" in text_lower or "공유" in text_lower):
            violations.append("친구 추천/공유 이벤트는 광고성 정보 - 수신동의 필요")

        # 11. 카카오 반려 사례 기반 검증 (강화된 패턴 매칭)
        rejection_violations = self._check_kakao_rejection_patterns(text_lower)
        violations.extend(rejection_violations)

        # 위험도 계산
        if len(violations) >= 3:
            risk_level = "HIGH"
        elif len(violations) >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return PolicyResult(
            is_compliant=len(violations) == 0,
            violations=violations,
            risk_level=risk_level,
            has_profanity=False,
            confidence=0.9 if len(violations) == 0 else 0.95,
            details={
                "ad_content_detected": has_ad_content,
                "promo_count": promo_count,
                "spam_count": spam_count,
                "total_violations": len(violations)
            }
        )

    def check_comprehensive(self, text: str, variables: Optional[Dict[str, str]] = None) -> PolicyResult:
        """종합적인 정책 및 비속어 검증"""
        # 1. 비속어 검증
        profanity_result = self.check_profanity(text)
        if not profanity_result.is_compliant:
            return profanity_result

        # 2. 정책 준수 검증
        policy_result = self.check_policy_compliance(text, variables)

        # 결과 통합
        return PolicyResult(
            is_compliant=policy_result.is_compliant,
            violations=policy_result.violations,
            risk_level=policy_result.risk_level,
            has_profanity=profanity_result.has_profanity,
            profanity_details=profanity_result.profanity_details,
            confidence=min(profanity_result.confidence, policy_result.confidence),
            details={
                "profanity_check": "passed" if not profanity_result.has_profanity else "failed",
                "policy_check": "passed" if policy_result.is_compliant else "failed",
                **policy_result.details
            }
        )

    def get_violation_message(self, result: PolicyResult) -> str:
        """검증 결과를 사용자 친화적 메시지로 변환"""
        if result.is_compliant and not result.has_profanity:
            return "모든 정책 검증을 통과했습니다."

        messages = []

        if result.has_profanity:
            messages.append("부적절한 언어가 감지되었습니다.")

        if result.violations:
            messages.extend(result.violations)

        return "\n".join(messages)

    def _check_kakao_rejection_patterns(self, text_lower: str) -> List[str]:
        """
        카카오 반려 사례 기반 패턴 검증

        Args:
            text_lower: 소문자로 변환된 텍스트

        Returns:
            위반 사항 리스트
        """
        violations = []

        # 1. 수신자가 요청하지 않은 공지성 메시지
        unsolicited_count = sum(1 for pattern in self.kakao_rejection_patterns["unsolicited_notice"]
                               if pattern in text_lower)
        if unsolicited_count >= 1:
            violations.append("수신자가 요청하지 않은 내용으로 광고성 및 공지성 메시지에 해당함에 따라 알림톡 발송 불가")

        # 2. 프로모션 리마인드 (쿠폰, 포인트 등)
        promo_count = sum(1 for pattern in self.kakao_rejection_patterns["promotional_reminder"]
                         if pattern in text_lower)
        if promo_count >= 1:
            violations.append("수신자가 요청하지 않은 보유 쿠폰/포인트 리마인드 안내로 광고성 메시지에 해당함")

        # 3. 수신자 액션이 불명확한 메시지
        unclear_count = sum(1 for pattern in self.kakao_rejection_patterns["unclear_recipient_action"]
                           if pattern in text_lower)
        if unclear_count >= 1:
            violations.append("수신 대상을 명확하게 확인하기 어려움. 수신자의 액션을 메시지 내 명시 필요")

        # 4. 단순 기념일 축하, 감사 인사 (수신자 액션 불명확)
        greeting_count = sum(1 for pattern in self.kakao_rejection_patterns["simple_greetings"]
                            if pattern in text_lower)
        if greeting_count >= 2:  # 2개 이상의 인사말 패턴
            violations.append("단순 기념일 축하, 감사, 안부 인사 등 수신자 액션이 명확하지 않은 메시지는 알림톡 발송 불가")

        # 5. 특정 반려 패턴 조합 검사
        if "휴면" in text_lower and ("포인트" in text_lower or "소멸" in text_lower):
            violations.append("휴면 안내와 포인트 소멸 안내가 함께 기재되어 발송 불가. 각각 분리하여 별도 템플릿 작성 필요")

        if "(광고)" in text_lower and ("정보성" in text_lower or "안내" in text_lower):
            violations.append("정보성 메시지에 (광고)가 기재되어 광고성 메시지가 됨. (광고) 삭제 필요")

        # 6. 업데이트/개선 관련 (수신자 액션 불명확)
        # if any(word in text_lower for word in ["업데이트", "개선", "변경"]) and "액션" not in text_lower:
        #     violations.append("수신자의 별도 액션이 확인되지 않는 업데이트/변경사항으로 공지성 메시지에 해당하여 알림톡 발송 불가")

        return violations