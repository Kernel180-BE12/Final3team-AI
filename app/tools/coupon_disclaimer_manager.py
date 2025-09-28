#!/usr/bin/env python3
"""
쿠폰 발송 근거 문구 관리 도구
카카오 정책에 따른 쿠폰/포인트 관련 발송 근거 문구 자동 추가
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CouponDetectionResult:
    """쿠폰 탐지 결과"""
    has_coupon_content: bool
    detected_keywords: List[str]
    confidence: float
    suggested_disclaimer: Optional[str] = None
    disclaimer_reason: Optional[str] = None


class CouponDisclaimerManager:
    """쿠폰/포인트 관련 발송 근거 문구 관리 클래스"""

    def __init__(self):
        """초기화"""
        self._setup_keywords()
        self._setup_disclaimers()

    def _setup_keywords(self):
        """쿠폰 관련 키워드 설정"""

        # 1. 직접적인 쿠폰 관련 키워드
        self.direct_coupon_keywords = [
            "쿠폰", "할인쿠폰", "할인권", "상품권", "이용권",
            "무료쿠폰", "할인 쿠폰", "적립쿠폰"
        ]

        # 2. 포인트/마일리지 관련 키워드
        self.point_keywords = [
            "포인트", "적립금", "마일리지", "캐시", "카카오페이", "네이버페이",
            "적립", "포인트 적립", "마일리지 적립"
        ]

        # 3. 혜택 관련 키워드
        self.benefit_keywords = [
            "혜택", "이벤트", "프로모션", "특가", "세일",
            "할인", "무료", "증정", "사은품", "리워드"
        ]

        # 4. 발급/소멸 관련 키워드
        self.issuance_keywords = [
            "발급", "지급", "제공", "드립니다", "받으세요",
            "소멸", "만료", "사라집니다", "없어집니다"
        ]

    def _setup_disclaimers(self):
        """발송 근거 문구 설정 (카카오 가이드 기반)"""

        self.disclaimer_templates = {
            "consent_based": "이 메시지는 고객님의 동의에 의해 지급된 쿠폰 안내 메시지입니다.",
            "event_based": "이 메시지는 고객님이 참여한 이벤트 당첨으로 지급된 쿠폰 안내 메시지입니다.",
            "survey_based": "이 메시지는 서비스 만족도조사에 참여하신 고객님들께 지급된 쿠폰 안내 메시지입니다.",
            "purchase_based": "이 메시지는 구매하신 상품(서비스)의 사은품으로 지급된 쿠폰 안내 메시지입니다.",
            "contract_based": "이 메시지는 이용약관(계약서) 동의에 따라 지급된 쿠폰 안내 메시지입니다.",
            "membership_based": "이 메시지는 회원가입 혜택으로 지급된 쿠폰 안내 메시지입니다."
        }

        # 상황별 발송 근거 매핑
        self.context_disclaimer_mapping = {
            "생일": "consent_based",
            "회원가입": "membership_based",
            "구매": "purchase_based",
            "주문": "purchase_based",
            "이벤트": "event_based",
            "당첨": "event_based",
            "설문": "survey_based",
            "조사": "survey_based",
            "리뷰": "survey_based",
            "평가": "survey_based"
        }

    def detect_coupon_content(self, content: str) -> CouponDetectionResult:
        """
        템플릿 내용에서 쿠폰 관련 내용 탐지

        Args:
            content: 검사할 템플릿 내용

        Returns:
            탐지 결과
        """
        content_lower = content.lower()
        detected_keywords = []

        # 키워드 탐지
        all_keywords = (
            self.direct_coupon_keywords +
            self.point_keywords +
            self.benefit_keywords +
            self.issuance_keywords
        )

        for keyword in all_keywords:
            if keyword in content_lower:
                detected_keywords.append(keyword)

        # 쿠폰 관련 내용 여부 판단
        has_coupon_content = len(detected_keywords) > 0

        # 신뢰도 계산
        direct_matches = sum(1 for k in detected_keywords if k in self.direct_coupon_keywords)
        point_matches = sum(1 for k in detected_keywords if k in self.point_keywords)
        benefit_matches = sum(1 for k in detected_keywords if k in self.benefit_keywords)
        issuance_matches = sum(1 for k in detected_keywords if k in self.issuance_keywords)

        # 가중치를 적용한 신뢰도 계산
        confidence = min(1.0, (
            direct_matches * 0.4 +
            point_matches * 0.3 +
            benefit_matches * 0.2 +
            issuance_matches * 0.1
        ))

        # 발송 근거 제안
        suggested_disclaimer = None
        disclaimer_reason = None

        if has_coupon_content and confidence >= 0.3:
            suggested_disclaimer, disclaimer_reason = self._suggest_disclaimer(content)

        return CouponDetectionResult(
            has_coupon_content=has_coupon_content,
            detected_keywords=detected_keywords,
            confidence=confidence,
            suggested_disclaimer=suggested_disclaimer,
            disclaimer_reason=disclaimer_reason
        )

    def _suggest_disclaimer(self, content: str) -> Tuple[str, str]:
        """
        내용을 분석하여 적절한 발송 근거 문구 제안 (LLM 기반 맞춤화)

        Args:
            content: 템플릿 내용

        Returns:
            (발송 근거 문구, 제안 이유)
        """
        content_lower = content.lower()

        # LLM을 사용한 맞춤화된 발송 근거 생성
        customized_disclaimer = self._generate_customized_disclaimer(content)
        if customized_disclaimer:
            return customized_disclaimer, "LLM 기반 맞춤화"

        # 폴백: 상황별 발송 근거 매핑 확인
        for context_keyword, disclaimer_type in self.context_disclaimer_mapping.items():
            if context_keyword in content_lower:
                return (
                    self.disclaimer_templates[disclaimer_type],
                    f"'{context_keyword}' 키워드 기반"
                )

        # 기본값: 동의 기반
        return (
            self.disclaimer_templates["consent_based"],
            "일반적인 쿠폰 발급 상황"
        )

    def _generate_customized_disclaimer(self, content: str) -> Optional[str]:
        """
        LLM을 사용하여 상황에 맞는 맞춤화된 발송 근거 문구 생성

        Args:
            content: 템플릿 내용

        Returns:
            맞춤화된 발송 근거 문구 또는 None
        """
        try:
            # LLM 호출을 위한 동적 import (순환 import 방지)
            from app.utils.llm_provider_manager import invoke_llm_with_fallback

            prompt = f"""다음 알림톡 템플릿의 내용을 분석하여 카카오 정책에 맞는 적절한 발송 근거 문구를 생성해주세요.

템플릿 내용:
{content}

카카오 가이드 기본 문구들:
- 이 메시지는 고객님의 동의에 의해 지급된 쿠폰 안내 메시지입니다.
- 이 메시지는 고객님이 참여한 이벤트 당첨으로 지급된 쿠폰 안내 메시지입니다.
- 이 메시지는 서비스 만족도조사에 참여하신 고객님들께 지급된 쿠폰 안내 메시지입니다.
- 이 메시지는 구매하신 상품(서비스)의 사은품으로 지급된 쿠폰 안내 메시지입니다.
- 이 메시지는 이용약관(계약서) 동의에 따라 지급된 쿠폰 안내 메시지입니다.

지시사항:
1. 템플릿 내용을 분석하여 쿠폰 지급 상황을 파악하세요
2. 위 기본 문구를 참고하여 상황에 맞게 약간 수정한 문구를 만드세요
3. "이 메시지는 ~으로 지급된 쿠폰 안내 메시지입니다." 형식을 유지하세요
4. 한 문장으로만 응답하세요

응답 예시:
- 생일 쿠폰 → "이 메시지는 생일 이벤트 참여로 지급된 축하 쿠폰 안내 메시지입니다."
- 리뷰 쿠폰 → "이 메시지는 상품 리뷰 작성 완료로 지급된 감사 쿠폰 안내 메시지입니다."

발송 근거 문구:"""

            response, _, _ = invoke_llm_with_fallback(prompt=prompt)

            # 응답에서 적절한 발송 근거 문구 추출
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('이 메시지는') and line.endswith('메시지입니다.'):
                    print(f" LLM 맞춤화 발송 근거: {line}")
                    return line

            return None

        except Exception as e:
            print(f" LLM 기반 발송 근거 생성 실패: {e}")
            return None

    def add_disclaimer_to_template(self, template: str, detection_result: CouponDetectionResult) -> str:
        """
        템플릿에 발송 근거 문구 추가

        Args:
            template: 원본 템플릿
            detection_result: 쿠폰 탐지 결과

        Returns:
            발송 근거 문구가 추가된 템플릿
        """
        if not detection_result.has_coupon_content or not detection_result.suggested_disclaimer:
            return template

        # 이미 발송 근거 문구가 있는지 확인
        existing_disclaimers = [
            "이 메시지는",
            "쿠폰 안내 메시지입니다",
            "지급된 쿠폰",
            "발송 근거"
        ]

        template_lower = template.lower()
        if any(disclaimer in template_lower for disclaimer in existing_disclaimers):
            print("ℹ️ 이미 발송 근거 문구가 포함되어 있습니다.")
            return template

        # 템플릿 끝에 발송 근거 문구 추가
        disclaimer = detection_result.suggested_disclaimer

        # 기존 템플릿 끝에 빈 줄 + 발송 근거 추가
        if not template.endswith('\n'):
            template += '\n'

        template += f"\n{disclaimer}"

        print(f" 발송 근거 문구 추가됨: {detection_result.disclaimer_reason}")
        return template

    def validate_disclaimer_compliance(self, content: str) -> Dict[str, Any]:
        """
        발송 근거 문구 준수 여부 검증

        Args:
            content: 검증할 내용

        Returns:
            준수 여부 정보
        """
        detection_result = self.detect_coupon_content(content)

        if not detection_result.has_coupon_content:
            return {
                "compliant": True,
                "reason": "쿠폰 관련 내용이 없음",
                "action_needed": False
            }

        # 발송 근거 문구 존재 여부 확인
        has_disclaimer = any(
            disclaimer in content.lower()
            for disclaimer in ["이 메시지는", "쿠폰 안내 메시지입니다"]
        )

        if has_disclaimer:
            return {
                "compliant": True,
                "reason": "발송 근거 문구 포함됨",
                "action_needed": False
            }
        else:
            return {
                "compliant": False,
                "reason": f"쿠폰 관련 내용 발견 (키워드: {', '.join(detection_result.detected_keywords[:3])}), 발송 근거 문구 누락",
                "action_needed": True,
                "suggested_disclaimer": detection_result.suggested_disclaimer
            }

    def get_all_disclaimer_templates(self) -> Dict[str, str]:
        """모든 발송 근거 템플릿 반환"""
        return self.disclaimer_templates.copy()


# 싱글톤 인스턴스
_coupon_manager_instance: Optional[CouponDisclaimerManager] = None


def get_coupon_disclaimer_manager() -> CouponDisclaimerManager:
    """전역 쿠폰 발송 근거 관리자 인스턴스 반환"""
    global _coupon_manager_instance
    if _coupon_manager_instance is None:
        _coupon_manager_instance = CouponDisclaimerManager()
    return _coupon_manager_instance


def auto_add_coupon_disclaimer(template: str) -> str:
    """
    편의 함수: 쿠폰 발송 근거 문구 자동 추가

    Args:
        template: 원본 템플릿

    Returns:
        발송 근거 문구가 추가된 템플릿
    """
    manager = get_coupon_disclaimer_manager()
    detection_result = manager.detect_coupon_content(template)
    return manager.add_disclaimer_to_template(template, detection_result)


if __name__ == "__main__":
    # 테스트
    print("=== 쿠폰 발송 근거 문구 관리 도구 테스트 ===")

    manager = CouponDisclaimerManager()

    # 반려 사례 기반 테스트
    test_cases = [
        "생일을 축하드리며 특별한 쿠폰을 발급해드렸습니다.",
        "회원가입 감사 쿠폰 5,000원권을 지급합니다.",
        "리뷰 작성 완료로 감사 쿠폰을 드립니다.",
        "안녕하세요. 예약 확인 메시지입니다.",  # 쿠폰 없음
        "포인트 15,000원이 소멸 예정입니다.",
    ]

    for i, test_template in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_template}")

        detection_result = manager.detect_coupon_content(test_template)
        print(f"쿠폰 내용: {'있음' if detection_result.has_coupon_content else '없음'} (신뢰도: {detection_result.confidence:.2f})")

        if detection_result.has_coupon_content:
            print(f"탐지 키워드: {', '.join(detection_result.detected_keywords)}")
            enhanced_template = manager.add_disclaimer_to_template(test_template, detection_result)
            if enhanced_template != test_template:
                print(f"개선된 템플릿:\n{enhanced_template}")