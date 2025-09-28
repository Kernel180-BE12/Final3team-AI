#!/usr/bin/env python3
"""
Joke Detector - 장난/농담 감지기
"짬짜면" 같은 개인적 장난 요청을 AI로 감지하여 친화적으로 거절
"""

import asyncio
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class JokeDetectionResult:
    """장난 감지 결과"""
    is_joke: bool
    confidence: float  # 0.0 ~ 1.0
    category: str  # 'business', 'personal_joke', 'casual_chat'
    reason: str
    friendly_message: str


class JokeDetector:
    """AI 기반 장난/농담 감지기"""

    def __init__(self):
        """장난 감지기 초기화"""
        self.business_examples = [
            "예약 확인 안내를 보내주세요",
            "이벤트 공지를 고객들에게 알려주세요",
            "서비스 점검 안내 템플릿을 만들어주세요",
            "회원가입 완료 알림을 만들어주세요",
            "배송 상태 안내 메시지를 작성해주세요"
        ]

        self.joke_examples = [
            "짬짜면 먹고 싶어. 같이 먹을 분은 10C 강의실에 오세요",
            "친구들이랑 놀러가자. 재미있을 것 같아",
            "심심해서 그냥 써봤어요. 안녕하세요",
            "오늘 날씨가 좋네요. 산책하고 싶어",
            "숙제 도와줘. 수학 문제가 너무 어려워"
        ]

    async def detect_joke_async(self, text: str) -> JokeDetectionResult:
        """
        비동기 장난 감지

        Args:
            text: 사용자 입력 텍스트

        Returns:
            JokeDetectionResult: 감지 결과
        """
        try:
            # AI 분류 실행
            result = await self._classify_with_ai_async(text)
            return result

        except Exception as e:
            print(f"장난 감지 오류: {e}")
            # 오류 시 보수적 접근 (비즈니스로 간주)
            return JokeDetectionResult(
                is_joke=False,
                confidence=0.3,
                category='business',
                reason=f'AI 감지 오류: {str(e)}',
                friendly_message=""
            )

    def detect_joke_sync(self, text: str) -> JokeDetectionResult:
        """
        동기 장난 감지 (하위 호환성)

        Args:
            text: 사용자 입력 텍스트

        Returns:
            JokeDetectionResult: 감지 결과
        """
        try:
            return asyncio.run(self.detect_joke_async(text))
        except Exception as e:
            print(f"동기 장난 감지 오류: {e}")
            return JokeDetectionResult(
                is_joke=False,
                confidence=0.3,
                category='business',
                reason=f'동기 감지 오류: {str(e)}',
                friendly_message=""
            )

    async def _classify_with_ai_async(self, text: str) -> JokeDetectionResult:
        """AI로 장난 여부 분류"""
        try:
            # LLM 호출
            from app.utils.llm_provider_manager import ainvoke_llm_with_fallback

            prompt = self._create_classification_prompt(text)
            response_text, provider, model = await ainvoke_llm_with_fallback(prompt)

            # 응답 파싱
            result = self._parse_ai_response(response_text)

            print(f"장난 감지 완료 - Provider: {provider}, Model: {model}, Is_Joke: {result.is_joke}")

            return result

        except Exception as e:
            print(f"AI 장난 감지 오류: {e}")
            return JokeDetectionResult(
                is_joke=False,
                confidence=0.3,
                category='business',
                reason=f'AI 호출 실패: {str(e)}',
                friendly_message=""
            )

    def _create_classification_prompt(self, text: str) -> str:
        """AI 분류용 프롬프트 생성"""
        business_examples_str = "\n".join([f"- {ex}" for ex in self.business_examples])
        joke_examples_str = "\n".join([f"- {ex}" for ex in self.joke_examples])

        return f"""
다음 텍스트가 진짜 비즈니스 알림톡 요청인지, 아니면 개인적인 장난/농담인지 정확히 판단해주세요.

[분석할 텍스트]
"{text}"

[진짜 비즈니스 요청 예시들]
{business_examples_str}

[개인적 장난/농담 예시들]
{joke_examples_str}

[판단 기준]
1. **비즈니스 목적**: 고객에게 보낼 공식적인 알림/안내/서비스 관련 내용
   - 예약, 주문, 배송, 이벤트, 회원, 서비스 관련
   - 공식적이고 업무적인 톤

2. **개인적 장난/농담**:
   - 음식 모임, 개인 약속, 친구 관계, 일상 잡담
   - "먹고 싶어", "놀러가자", "심심해", "재미있어" 등
   - 반말, 친근한 톤, 개인적 관계 암시

3. **캐주얼 채팅**:
   - 의미없는 테스트, 단순 인사, 장난성 입력
   - "안녕", "테스트", "심심해서" 등

[응답 형식 - 정확히 이 형식으로만 답변]
카테고리: [business|personal_joke|casual_chat]
신뢰도: [0.0-1.0]
이유: [한 줄로 판단 근거]
"""

    def _parse_ai_response(self, response: str) -> JokeDetectionResult:
        """AI 응답 파싱"""
        lines = response.strip().split('\n')

        # 기본값
        category = 'business'
        confidence = 0.5
        reason = 'AI 응답 파싱 실패'

        # 응답 파싱
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                if '카테고리' in key or 'category' in key:
                    category = value.lower()
                elif '신뢰도' in key or 'confidence' in key:
                    try:
                        confidence = float(value)
                        confidence = max(0.0, min(1.0, confidence))  # 0.0~1.0 범위 제한
                    except ValueError:
                        confidence = 0.5
                elif '이유' in key or 'reason' in key:
                    reason = value

        # 카테고리 정규화
        if category not in ['business', 'personal_joke', 'casual_chat']:
            if 'joke' in category or 'personal' in category or '장난' in category or '개인' in category:
                category = 'personal_joke'
            elif 'casual' in category or 'chat' in category or '잡담' in category:
                category = 'casual_chat'
            else:
                category = 'business'

        # 장난 여부 판단
        is_joke = category in ['personal_joke', 'casual_chat']

        # 친화적 메시지 생성
        friendly_message = self._generate_friendly_message(category, reason) if is_joke else ""

        return JokeDetectionResult(
            is_joke=is_joke,
            confidence=confidence,
            category=category,
            reason=reason,
            friendly_message=friendly_message
        )

    def _generate_friendly_message(self, category: str, reason: str) -> str:
        """사용자 친화적 거절 메시지 생성"""
        base_message = "비즈니스 알림톡에 적합한 내용으로 다시 입력해주세요."
        examples = "\n\n예시: 예약 확인 안내, 이벤트 공지, 서비스 안내, 주문 상태 알림 등"

        if category == 'personal_joke':
            specific_message = "\n현재 입력: 개인적인 모임이나 농담성 내용으로 판단됩니다."
        elif category == 'casual_chat':
            specific_message = "\n현재 입력: 일상적인 대화나 잡담으로 판단됩니다."
        else:
            specific_message = "\n좀 더 구체적인 비즈니스 목적을 포함해주세요."

        return base_message + specific_message + examples

    def is_high_confidence_joke(self, result: JokeDetectionResult, threshold: float = 0.8) -> bool:
        """
        높은 신뢰도로 장난이라고 판단되는지 확인 (Fast Fail용)

        Args:
            result: 장난 감지 결과
            threshold: 신뢰도 임계값

        Returns:
            bool: 확실한 장난 여부
        """
        return result.is_joke and result.confidence >= threshold


# 편의 함수들
async def detect_joke_async(text: str) -> JokeDetectionResult:
    """
    편의 함수: 비동기 장난 감지

    Args:
        text: 사용자 입력 텍스트

    Returns:
        JokeDetectionResult: 감지 결과
    """
    detector = JokeDetector()
    return await detector.detect_joke_async(text)


def detect_joke_sync(text: str) -> JokeDetectionResult:
    """
    편의 함수: 동기 장난 감지

    Args:
        text: 사용자 입력 텍스트

    Returns:
        JokeDetectionResult: 감지 결과
    """
    detector = JokeDetector()
    return detector.detect_joke_sync(text)


# 테스트 코드
if __name__ == "__main__":
    async def test_detector():
        detector = JokeDetector()

        test_cases = [
            "짬짜면 먹고 싶어. 같이 먹을 분은 내일 10시까지 10C 강의실에 오세요.",
            "예약이 확정되었습니다. 내일 오후 3시에 방문해주세요.",
            "심심해서 그냥 써봤어요",
            "이벤트 안내를 고객들에게 보내주세요",
            "친구들이랑 놀러가자. 재미있을 것 같아",
            "회원가입 완료 알림 템플릿을 만들어주세요"
        ]

        print("=== 장난 감지 테스트 ===")
        for text in test_cases:
            print(f"\n입력: {text}")
            result = await detector.detect_joke_async(text)
            print(f"장난 여부: {'장난' if result.is_joke else ' 비즈니스'}")
            print(f"카테고리: {result.category}")
            print(f"신뢰도: {result.confidence:.2f}")
            print(f"이유: {result.reason}")

            if result.is_joke:
                print(f"거절 메시지:\n{result.friendly_message}")

            if detector.is_high_confidence_joke(result):
                print(" Fast Fail: 확실한 장난으로 판단")

            print("-" * 60)

    # 비동기 테스트 실행
    asyncio.run(test_detector())