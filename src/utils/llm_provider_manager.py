"""
LLM 공급자 관리 시스템
Gemini API 장애 시 OpenAI로 자동 Fallback
"""
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

class LLMProviderManager:
    """
    LLM 공급자 관리 및 Fallback 시스템
    
    우선순위:
    1. Gemini (기본)
    2. OpenAI (백업)
    """
    
    def __init__(self, gemini_api_key: str = None, openai_api_key: str = None):
        """
        Args:
            gemini_api_key: Gemini API 키
            openai_api_key: OpenAI API 키 (백업용)
        """
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 사용 가능한 공급자들
        self.providers = []
        self._initialize_providers()
        
        # 현재 활성 공급자
        self.current_provider = None
        self.current_model = None
        
        # 장애 추적
        self.failure_counts = {"gemini": 0, "openai": 0}
        self.max_failures = 3
        
    def _initialize_providers(self):
        """사용 가능한 LLM 공급자들 초기화"""
        self.providers = []
        
        # Gemini 공급자 (우선순위 1)
        if self.gemini_api_key:
            self.providers.append({
                "name": "gemini",
                "priority": 1,
                "models": [
                    "gemini-2.0-flash-exp",
                    "gemini-1.5-flash", 
                    "gemini-1.5-pro"
                ],
                "api_key": self.gemini_api_key,
                "available": True
            })
        
        # OpenAI 공급자 (우선순위 2)
        if self.openai_api_key:
            self.providers.append({
                "name": "openai", 
                "priority": 2,
                "models": [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-3.5-turbo"
                ],
                "api_key": self.openai_api_key,
                "available": True
            })
        
        # 우선순위 정렬
        self.providers.sort(key=lambda x: x["priority"])
        
        if not self.providers:
            raise ValueError("최소 하나의 API 키가 필요합니다 (GEMINI_API_KEY 또는 OPENAI_API_KEY)")
    
    def get_llm_instance(self, provider_name: str = None, model_name: str = None) -> Tuple[Any, str, str]:
        """
        LLM 인스턴스 생성
        
        Args:
            provider_name: 특정 공급자 지정 (선택사항)
            model_name: 특정 모델 지정 (선택사항)
            
        Returns:
            (llm_instance, provider_name, model_name)
        """
        # 특정 공급자 지정된 경우
        if provider_name:
            provider = next((p for p in self.providers if p["name"] == provider_name), None)
            if not provider or not provider["available"]:
                raise ValueError(f"공급자 '{provider_name}'를 사용할 수 없습니다")
            
            model = model_name or provider["models"][0]
            return self._create_llm_instance(provider["name"], model), provider["name"], model
        
        # 자동 선택 (우선순위 순)
        for provider in self.providers:
            if not provider["available"]:
                continue
                
            if self.failure_counts[provider["name"]] >= self.max_failures:
                self.logger.warning(f"{provider['name']} 공급자가 최대 실패 횟수를 초과했습니다")
                continue
            
            model = model_name or provider["models"][0]
            
            try:
                llm_instance = self._create_llm_instance(provider["name"], model)
                self.current_provider = provider["name"]
                self.current_model = model
                return llm_instance, provider["name"], model
            except Exception as e:
                self.logger.error(f"{provider['name']} 초기화 실패: {e}")
                self.failure_counts[provider["name"]] += 1
                continue
        
        raise RuntimeError("사용 가능한 LLM 공급자가 없습니다")
    
    def _create_llm_instance(self, provider_name: str, model_name: str):
        """LLM 인스턴스 생성"""
        if provider_name == "gemini":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.gemini_api_key,
                temperature=0.1
            )
        elif provider_name == "openai":
            return ChatOpenAI(
                model=model_name,
                openai_api_key=self.openai_api_key,
                temperature=0.1
            )
        else:
            raise ValueError(f"지원하지 않는 공급자: {provider_name}")
    
    def invoke_with_fallback(self, messages: List[BaseMessage], **kwargs) -> Tuple[str, str, str]:
        """
        Fallback을 포함한 LLM 호출

        Args:
            messages: 메시지 리스트
            **kwargs: 추가 파라미터

        Returns:
            (response_content, provider_used, model_used)
        """
        last_error = None

        for provider in self.providers:
            if not provider["available"]:
                continue

            if self.failure_counts[provider["name"]] >= self.max_failures:
                continue

            for model in provider["models"]:
                try:
                    self.logger.info(f"시도 중: {provider['name']} - {model}")

                    llm_instance = self._create_llm_instance(provider["name"], model)
                    response = llm_instance.invoke(messages, **kwargs)

                    # 성공 시 실패 카운트 리셋
                    self.failure_counts[provider["name"]] = 0

                    self.current_provider = provider["name"]
                    self.current_model = model

                    self.logger.info(f"성공: {provider['name']} - {model}")
                    return response.content, provider["name"], model

                except Exception as e:
                    last_error = e
                    self.logger.warning(f"{provider['name']} - {model} 실패: {e}")
                    self.failure_counts[provider["name"]] += 1

                    # 잠시 대기 후 다음 시도
                    time.sleep(1)
                    continue

        # 모든 공급자 실패
        raise RuntimeError(f"모든 LLM 공급자에서 실패했습니다. 마지막 오류: {last_error}")

    async def ainvoke_with_fallback(self, messages: List[BaseMessage], **kwargs) -> Tuple[str, str, str]:
        """
        비동기 Fallback을 포함한 LLM 호출

        Args:
            messages: 메시지 리스트
            **kwargs: 추가 파라미터

        Returns:
            (response_content, provider_used, model_used)
        """
        import asyncio

        last_error = None

        for provider in self.providers:
            if not provider["available"]:
                continue

            if self.failure_counts[provider["name"]] >= self.max_failures:
                continue

            for model in provider["models"]:
                try:
                    self.logger.info(f"비동기 시도 중: {provider['name']} - {model}")

                    llm_instance = self._create_llm_instance(provider["name"], model)
                    response = await llm_instance.ainvoke(messages, **kwargs)

                    # 성공 시 실패 카운트 리셋
                    self.failure_counts[provider["name"]] = 0

                    self.current_provider = provider["name"]
                    self.current_model = model

                    self.logger.info(f"비동기 성공: {provider['name']} - {model}")
                    return response.content, provider["name"], model

                except Exception as e:
                    last_error = e
                    self.logger.warning(f"{provider['name']} - {model} 비동기 실패: {e}")
                    self.failure_counts[provider["name"]] += 1

                    # 잠시 대기 후 다음 시도
                    await asyncio.sleep(1)
                    continue

        # 모든 공급자 실패
        raise RuntimeError(f"모든 LLM 공급자에서 실패했습니다. 마지막 오류: {last_error}")
    
    def invoke_simple(self, prompt: str, system_message: str = None, **kwargs) -> Tuple[str, str, str]:
        """
        간단한 텍스트 프롬프트로 LLM 호출

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지 (선택사항)
            **kwargs: 추가 파라미터

        Returns:
            (response_content, provider_used, model_used)
        """
        messages = []

        if system_message:
            messages.append(SystemMessage(content=system_message))

        messages.append(HumanMessage(content=prompt))

        return self.invoke_with_fallback(messages, **kwargs)

    async def ainvoke_simple(self, prompt: str, system_message: str = None, **kwargs) -> Tuple[str, str, str]:
        """
        비동기 간단한 텍스트 프롬프트로 LLM 호출

        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지 (선택사항)
            **kwargs: 추가 파라미터

        Returns:
            (response_content, provider_used, model_used)
        """
        messages = []

        if system_message:
            messages.append(SystemMessage(content=system_message))

        messages.append(HumanMessage(content=prompt))

        return await self.ainvoke_with_fallback(messages, **kwargs)
    
    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "current_provider": self.current_provider,
            "current_model": self.current_model,
            "available_providers": [p["name"] for p in self.providers if p["available"]],
            "failure_counts": self.failure_counts.copy(),
            "total_providers": len(self.providers)
        }
    
    def reset_failure_counts(self):
        """실패 카운트 리셋"""
        self.failure_counts = {provider["name"]: 0 for provider in self.providers}
        self.logger.info("실패 카운트가 리셋되었습니다")
    
    def set_provider_availability(self, provider_name: str, available: bool):
        """공급자 사용 가능 상태 설정"""
        for provider in self.providers:
            if provider["name"] == provider_name:
                provider["available"] = available
                self.logger.info(f"{provider_name} 공급자 상태: {'사용가능' if available else '사용불가'}")
                break


# 전역 인스턴스 (싱글톤 패턴)
_global_llm_manager = None

def get_llm_manager() -> LLMProviderManager:
    """전역 LLM 매니저 인스턴스 반환"""
    global _global_llm_manager
    if _global_llm_manager is None:
        _global_llm_manager = LLMProviderManager()
    return _global_llm_manager

def invoke_llm_with_fallback(prompt: str, system_message: str = None, **kwargs) -> Tuple[str, str, str]:
    """
    편의 함수: Fallback을 포함한 LLM 호출

    Args:
        prompt: 사용자 프롬프트
        system_message: 시스템 메시지
        **kwargs: 추가 파라미터

    Returns:
        (response, provider_used, model_used)
    """
    manager = get_llm_manager()
    return manager.invoke_simple(prompt, system_message, **kwargs)


async def ainvoke_llm_with_fallback(prompt: str, system_message: str = None, **kwargs) -> Tuple[str, str, str]:
    """
    편의 함수: 비동기 Fallback을 포함한 LLM 호출

    Args:
        prompt: 사용자 프롬프트
        system_message: 시스템 메시지
        **kwargs: 추가 파라미터

    Returns:
        (response, provider_used, model_used)
    """
    manager = get_llm_manager()
    return await manager.ainvoke_simple(prompt, system_message, **kwargs)


if __name__ == "__main__":
    # 테스트
    print("=== LLM Provider Manager 테스트 ===")
    
    try:
        manager = LLMProviderManager()
        print(f"초기화 완료: {len(manager.providers)}개 공급자")
        
        # 상태 확인
        status = manager.get_current_status()
        print(f"사용 가능한 공급자: {status['available_providers']}")
        
        # 간단한 호출 테스트
        response, provider, model = manager.invoke_simple(
            "안녕하세요! 간단한 인사말을 해주세요.",
            "당신은 친근한 AI 어시스턴트입니다."
        )
        
        print(f" 성공: {provider} - {model}")
        print(f"응답: {response[:100]}...")
        
    except Exception as e:
        print(f" 오류: {e}")