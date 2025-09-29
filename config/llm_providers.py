"""
LLM Provider Configuration
Gemini와 OpenAI 양쪽 모두 지원하는 통합 설정
"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class LLMProvider(Enum):
    """지원되는 LLM 제공자들"""
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """LLM 설정 정보"""
    provider: LLMProvider
    api_key: str
    model_name: str
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    timeout: int = 30


class LLMProviderManager:
    """LLM 제공자 관리 클래스"""

    def __init__(self):
        self.providers = self._load_providers()
        self.primary_provider = self._get_primary_provider()
        self.fallback_provider = self._get_fallback_provider()

    def _load_providers(self) -> Dict[LLMProvider, LLMConfig]:
        """환경변수에서 제공자 설정 로드"""
        providers = {}

        # Gemini 설정
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            providers[LLMProvider.GEMINI] = LLMConfig(
                provider=LLMProvider.GEMINI,
                api_key=gemini_key,
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
                max_tokens=None,  # Gemini는 max_tokens 사용 안함
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
                timeout=int(os.getenv("GEMINI_TIMEOUT", "30"))
            )

        # OpenAI 설정
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            providers[LLMProvider.OPENAI] = LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=openai_key,
                model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                timeout=int(os.getenv("OPENAI_TIMEOUT", "30"))
            )

        return providers

    def _get_primary_provider(self) -> Optional[LLMProvider]:
        """기본 제공자 결정"""
        # 환경변수로 우선순위 지정 가능
        primary = os.getenv("PRIMARY_LLM_PROVIDER", "openai").lower()

        if primary == "openai" and LLMProvider.OPENAI in self.providers:
            return LLMProvider.OPENAI
        elif primary == "gemini" and LLMProvider.GEMINI in self.providers:
            return LLMProvider.GEMINI

        # 기본값: 사용 가능한 첫 번째 제공자
        if self.providers:
            return next(iter(self.providers.keys()))

        return None

    def _get_fallback_provider(self) -> Optional[LLMProvider]:
        """폴백 제공자 결정"""
        if len(self.providers) < 2:
            return None

        # 기본 제공자가 아닌 다른 제공자
        for provider in self.providers:
            if provider != self.primary_provider:
                return provider

        return None

    def get_config(self, provider: LLMProvider) -> Optional[LLMConfig]:
        """특정 제공자 설정 반환"""
        return self.providers.get(provider)

    def get_primary_config(self) -> Optional[LLMConfig]:
        """기본 제공자 설정 반환"""
        if self.primary_provider:
            return self.providers.get(self.primary_provider)
        return None

    def get_fallback_config(self) -> Optional[LLMConfig]:
        """폴백 제공자 설정 반환"""
        if self.fallback_provider:
            return self.providers.get(self.fallback_provider)
        return None

    def is_available(self, provider: LLMProvider) -> bool:
        """제공자 사용 가능 여부 확인"""
        return provider in self.providers

    def get_available_providers(self) -> list[LLMProvider]:
        """사용 가능한 제공자 목록"""
        return list(self.providers.keys())


# 전역 인스턴스
llm_manager = LLMProviderManager()


def get_llm_manager() -> LLMProviderManager:
    """LLM 관리자 인스턴스 반환"""
    return llm_manager