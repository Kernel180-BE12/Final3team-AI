"""
개선된 LLM 공급자 관리 시스템
Gemini와 OpenAI 모두 지원하는 통합 관리자
"""

import os
import asyncio
import logging
from typing import Any, Dict, Optional, Tuple, Union
from enum import Enum

import google.generativeai as genai
import openai
from openai import AsyncOpenAI


class LLMProvider(Enum):
    """지원되는 LLM 제공자들"""
    GEMINI = "gemini"
    OPENAI = "openai"


class LLMProviderManager:
    """개선된 LLM 공급자 관리자"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # API 키 로드
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        # 모델 설정
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

        # 클라이언트 초기화
        self._init_clients()

        # 사용 가능한 제공자 결정
        self.available_providers = self._get_available_providers()
        self.primary_provider = self._get_primary_provider()

        # 실패 추적
        self.failure_counts = {LLMProvider.GEMINI: 0, LLMProvider.OPENAI: 0}
        self.max_failures = 3

    def _init_clients(self):
        """클라이언트 초기화"""
        # Gemini 클라이언트
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(self.gemini_model)
                self.logger.info("Gemini 클라이언트 초기화 완료")
            except Exception as e:
                self.logger.error(f"Gemini 클라이언트 초기화 실패: {e}")
                self.gemini_client = None
        else:
            self.gemini_client = None

        # OpenAI 클라이언트
        if self.openai_api_key:
            try:
                openai.api_key = self.openai_api_key
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
                self.logger.info("OpenAI 클라이언트 초기화 완료")
            except Exception as e:
                self.logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
                self.openai_client = None
        else:
            self.openai_client = None

    def _get_available_providers(self) -> list[LLMProvider]:
        """사용 가능한 제공자 목록 반환"""
        available = []
        if self.gemini_client:
            available.append(LLMProvider.GEMINI)
        if self.openai_client:
            available.append(LLMProvider.OPENAI)
        return available

    def _get_primary_provider(self) -> Optional[LLMProvider]:
        """기본 제공자 결정"""
        # 환경변수로 우선순위 지정 가능
        primary = os.getenv("PRIMARY_LLM_PROVIDER", "gemini").lower()

        if primary == "openai" and LLMProvider.OPENAI in self.available_providers:
            return LLMProvider.OPENAI
        elif primary == "gemini" and LLMProvider.GEMINI in self.available_providers:
            return LLMProvider.GEMINI

        # 기본값: 사용 가능한 첫 번째 제공자
        return self.available_providers[0] if self.available_providers else None

    async def _invoke_gemini(self, prompt: str) -> str:
        """Gemini API 호출"""
        if not self.gemini_client:
            raise Exception("Gemini 클라이언트가 초기화되지 않았습니다")

        try:
            response = self.gemini_client.generate_content(prompt)
            return response.text
        except Exception as e:
            self.failure_counts[LLMProvider.GEMINI] += 1
            self.logger.error(f"Gemini API 호출 실패: {e}")
            raise

    async def _invoke_openai(self, prompt: str) -> str:
        """OpenAI API 호출"""
        if not self.openai_client:
            raise Exception("OpenAI 클라이언트가 초기화되지 않았습니다")

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            self.failure_counts[LLMProvider.OPENAI] += 1
            self.logger.error(f"OpenAI API 호출 실패: {e}")
            raise

    def _should_skip_provider(self, provider: LLMProvider) -> bool:
        """제공자를 건너뛸지 결정"""
        return self.failure_counts[provider] >= self.max_failures

    def _get_next_provider(self, current_provider: LLMProvider) -> Optional[LLMProvider]:
        """다음 시도할 제공자 반환"""
        for provider in self.available_providers:
            if provider != current_provider and not self._should_skip_provider(provider):
                return provider
        return None

    async def invoke_with_fallback(self, prompt: str) -> Tuple[str, str, str]:
        """
        폴백 기능이 있는 LLM 호출

        Returns:
            Tuple[응답_텍스트, 사용된_제공자, 사용된_모델]
        """
        if not self.available_providers:
            raise Exception("사용 가능한 LLM 제공자가 없습니다")

        # 기본 제공자부터 시도
        provider_to_try = self.primary_provider
        last_error = None

        while provider_to_try:
            if self._should_skip_provider(provider_to_try):
                provider_to_try = self._get_next_provider(provider_to_try)
                continue

            try:
                self.logger.info(f"{provider_to_try.value} 제공자로 LLM 호출 시도")

                if provider_to_try == LLMProvider.GEMINI:
                    response = await self._invoke_gemini(prompt)
                    return response, "gemini", self.gemini_model
                elif provider_to_try == LLMProvider.OPENAI:
                    response = await self._invoke_openai(prompt)
                    return response, "openai", self.openai_model

            except Exception as e:
                last_error = e
                self.logger.warning(f"{provider_to_try.value} 제공자 실패, 다음 제공자 시도: {e}")
                provider_to_try = self._get_next_provider(provider_to_try)

        # 모든 제공자 실패
        raise Exception(f"모든 LLM 제공자 호출 실패. 마지막 오류: {last_error}")

    def reset_failure_counts(self):
        """실패 카운트 리셋"""
        self.failure_counts = {LLMProvider.GEMINI: 0, LLMProvider.OPENAI: 0}
        self.logger.info("LLM 제공자 실패 카운트 리셋됨")

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "available_providers": [p.value for p in self.available_providers],
            "primary_provider": self.primary_provider.value if self.primary_provider else None,
            "failure_counts": {p.value: count for p, count in self.failure_counts.items()},
            "gemini_configured": bool(self.gemini_client),
            "openai_configured": bool(self.openai_client)
        }


# 전역 인스턴스
_llm_manager = None


def get_llm_manager() -> LLMProviderManager:
    """LLM 관리자 싱글톤 인스턴스 반환"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMProviderManager()
    return _llm_manager


# 편의 함수들 (하위 호환성을 위해 유지)
async def ainvoke_llm_with_fallback(prompt: str) -> Tuple[str, str, str]:
    """
    편의 함수: 폴백 기능이 있는 비동기 LLM 호출

    Returns:
        Tuple[응답_텍스트, 사용된_제공자, 사용된_모델]
    """
    manager = get_llm_manager()
    return await manager.invoke_with_fallback(prompt)


def invoke_llm_with_fallback(prompt: str) -> Tuple[str, str, str]:
    """
    편의 함수: 폴백 기능이 있는 동기 LLM 호출 (하위 호환성)

    Returns:
        Tuple[응답_텍스트, 사용된_제공자, 사용된_모델]
    """
    return asyncio.run(ainvoke_llm_with_fallback(prompt))