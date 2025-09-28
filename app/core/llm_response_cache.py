#!/usr/bin/env python3
"""
메모리 기반 LLM 응답 캐시 시스템
동일한 프롬프트에 대한 LLM 응답을 캐시하여 API 호출 절약 및 성능 향상
"""

import hashlib
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json


@dataclass
class CachedLLMResponse:
    """캐시된 LLM 응답 데이터"""
    prompt_hash: str
    prompt_preview: str  # 프롬프트 미리보기 (처음 100자)
    response: str
    model_params: Dict[str, Any]
    provider: str
    model_name: str
    created_at: datetime
    access_count: int
    last_accessed: datetime
    response_time_ms: float

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data


class LLMResponseCache:
    """메모리 기반 LLM 응답 캐시 시스템"""

    def __init__(self, max_cache_size: int = 500, ttl_hours: int = 12):
        """
        Args:
            max_cache_size: 최대 캐시 항목 수
            ttl_hours: 캐시 생존 시간 (시간)
        """
        self.cache: Dict[str, CachedLLMResponse] = {}
        self.max_cache_size = max_cache_size
        self.ttl = timedelta(hours=ttl_hours)
        self.max_prompt_length = 8000  # 너무 긴 프롬프트는 캐시하지 않음

        # 통계
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_writes": 0,
            "cache_cleanups": 0,
            "total_response_time_saved_ms": 0
        }

    async def get_cached_response(
        self,
        prompt: str,
        model_params: Dict[str, Any] = None,
        provider: str = "unknown",
        model_name: str = "unknown"
    ) -> Optional[str]:
        """
        캐시된 LLM 응답 조회

        Args:
            prompt: LLM 프롬프트
            model_params: 모델 파라미터
            provider: LLM 프로바이더
            model_name: 모델명

        Returns:
            캐시된 응답 또는 None
        """
        self.stats["total_requests"] += 1

        # 프롬프트가 너무 긴 경우 캐시 조회 안함
        if len(prompt) > self.max_prompt_length:
            self.stats["cache_misses"] += 1
            return None

        # 캐시 정리
        await self._cleanup_expired_cache()

        # 프롬프트 해시 생성
        prompt_hash = self._generate_prompt_hash(prompt, model_params)

        if prompt_hash in self.cache:
            cached_item = self.cache[prompt_hash]

            # 접근 통계 업데이트
            cached_item.access_count += 1
            cached_item.last_accessed = datetime.now()

            # 통계 업데이트
            self.stats["cache_hits"] += 1
            self.stats["total_response_time_saved_ms"] += cached_item.response_time_ms

            return cached_item.response

        self.stats["cache_misses"] += 1
        return None

    async def cache_response(
        self,
        prompt: str,
        response: str,
        model_params: Dict[str, Any] = None,
        provider: str = "unknown",
        model_name: str = "unknown",
        response_time_ms: float = 0.0
    ) -> bool:
        """
        LLM 응답을 캐시에 저장

        Args:
            prompt: LLM 프롬프트
            response: LLM 응답
            model_params: 모델 파라미터
            provider: LLM 프로바이더
            model_name: 모델명
            response_time_ms: 응답 시간 (밀리초)

        Returns:
            저장 성공 여부
        """
        try:
            # 프롬프트가 너무 긴 경우 캐시 안함
            if len(prompt) > self.max_prompt_length:
                return False

            # 빈 응답은 캐시하지 않음
            if not response or not response.strip():
                return False

            # 캐시 크기 제한 확인
            if len(self.cache) >= self.max_cache_size:
                await self._evict_lru_items(count=int(self.max_cache_size * 0.1))

            prompt_hash = self._generate_prompt_hash(prompt, model_params)

            cached_response = CachedLLMResponse(
                prompt_hash=prompt_hash,
                prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                response=response,
                model_params=model_params or {},
                provider=provider,
                model_name=model_name,
                created_at=datetime.now(),
                access_count=1,
                last_accessed=datetime.now(),
                response_time_ms=response_time_ms
            )

            self.cache[prompt_hash] = cached_response
            self.stats["cache_writes"] += 1
            return True

        except Exception as e:
            print(f"LLM 응답 캐시 저장 오류: {e}")
            return False

    def _generate_prompt_hash(self, prompt: str, model_params: Dict[str, Any] = None) -> str:
        """프롬프트와 모델 파라미터의 해시 생성"""
        # 프롬프트 정규화 (공백 정리)
        normalized_prompt = " ".join(prompt.split())

        # 모델 파라미터를 정렬된 JSON 문자열로 변환
        params_str = json.dumps(model_params or {}, sort_keys=True)

        # 결합하여 해시 생성
        combined = f"{normalized_prompt}:{params_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    async def _cleanup_expired_cache(self):
        """만료된 캐시 항목 제거"""
        current_time = datetime.now()
        expired_keys = []

        for key, cached_item in self.cache.items():
            if current_time - cached_item.created_at > self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.stats["cache_cleanups"] += 1

    async def _evict_lru_items(self, count: int):
        """LRU 방식으로 캐시 항목 제거"""
        if count <= 0 or not self.cache:
            return

        # 마지막 접근 시간 기준으로 정렬
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )

        # 오래된 항목부터 제거
        for i in range(min(count, len(sorted_items))):
            key = sorted_items[i][0]
            del self.cache[key]

    async def get_cache_stats(self) -> Dict:
        """캐시 통계 조회"""
        total_requests = self.stats["total_requests"]
        hit_ratio = (
            self.stats["cache_hits"] / total_requests
            if total_requests > 0 else 0.0
        )

        # 평균 응답 시간 절약 계산
        avg_time_saved = (
            self.stats["total_response_time_saved_ms"] / self.stats["cache_hits"]
            if self.stats["cache_hits"] > 0 else 0.0
        )

        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "total_requests": total_requests,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_writes": self.stats["cache_writes"],
            "hit_ratio": round(hit_ratio, 3),
            "cache_cleanups": self.stats["cache_cleanups"],
            "total_time_saved_ms": self.stats["total_response_time_saved_ms"],
            "average_time_saved_ms": round(avg_time_saved, 2),
            "estimated_api_calls_saved": self.stats["cache_hits"]
        }

    async def clear_cache(self):
        """캐시 전체 비우기"""
        self.cache.clear()
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_writes": 0,
            "cache_cleanups": 0,
            "total_response_time_saved_ms": 0
        }

    async def get_cached_items_info(self) -> list:
        """캐시된 항목들의 정보 조회"""
        items = []
        for cached_item in self.cache.values():
            items.append({
                "prompt_hash": cached_item.prompt_hash,
                "prompt_preview": cached_item.prompt_preview,
                "provider": cached_item.provider,
                "model_name": cached_item.model_name,
                "response_length": len(cached_item.response),
                "created_at": cached_item.created_at.isoformat(),
                "access_count": cached_item.access_count,
                "last_accessed": cached_item.last_accessed.isoformat(),
                "response_time_ms": cached_item.response_time_ms
            })

        # 접근 횟수 기준으로 정렬
        items.sort(key=lambda x: x["access_count"], reverse=True)
        return items

    async def invalidate_by_pattern(self, pattern: str):
        """패턴에 맞는 캐시 무효화 (프롬프트 미리보기 기준)"""
        keys_to_remove = []

        for key, cached_item in self.cache.items():
            if pattern.lower() in cached_item.prompt_preview.lower():
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

        return len(keys_to_remove)

    async def get_cache_efficiency(self) -> Dict:
        """캐시 효율성 분석"""
        if not self.cache:
            return {"efficiency": "No data"}

        # 접근 횟수 분석
        access_counts = [item.access_count for item in self.cache.values()]
        total_accesses = sum(access_counts)
        unique_items = len(access_counts)

        # 시간 절약 분석
        total_time_saved_hours = self.stats["total_response_time_saved_ms"] / (1000 * 60 * 60)

        return {
            "total_unique_responses": unique_items,
            "total_cache_accesses": total_accesses,
            "average_reuse_per_item": round(total_accesses / unique_items, 2) if unique_items > 0 else 0,
            "most_accessed_count": max(access_counts) if access_counts else 0,
            "least_accessed_count": min(access_counts) if access_counts else 0,
            "total_time_saved_hours": round(total_time_saved_hours, 2),
            "api_calls_avoided": self.stats["cache_hits"],
            "efficiency_score": round(hit_ratio * 100, 1) if (hit_ratio := self.stats["cache_hits"] / max(self.stats["total_requests"], 1)) else 0
        }


# 싱글톤 인스턴스
_llm_cache_instance = None

def get_llm_cache() -> LLMResponseCache:
    """LLM 캐시 싱글톤 인스턴스 반환"""
    global _llm_cache_instance
    if _llm_cache_instance is None:
        _llm_cache_instance = LLMResponseCache()
    return _llm_cache_instance


# 편의 함수들
async def get_cached_llm_response(
    prompt: str,
    model_params: Dict[str, Any] = None,
    provider: str = "unknown",
    model_name: str = "unknown"
) -> Optional[str]:
    """캐시된 LLM 응답 조회 편의 함수"""
    cache = get_llm_cache()
    return await cache.get_cached_response(prompt, model_params, provider, model_name)


async def save_llm_response_to_cache(
    prompt: str,
    response: str,
    model_params: Dict[str, Any] = None,
    provider: str = "unknown",
    model_name: str = "unknown",
    response_time_ms: float = 0.0
) -> bool:
    """LLM 응답 캐시 저장 편의 함수"""
    cache = get_llm_cache()
    return await cache.cache_response(prompt, response, model_params, provider, model_name, response_time_ms)


if __name__ == "__main__":
    import asyncio

    async def test_llm_cache():
        cache = LLMResponseCache()

        # 테스트 프롬프트와 응답
        test_prompt = "Create a professional email template for order confirmation"
        test_response = "Dear {customer_name}, your order #{order_id} has been confirmed."
        test_params = {"temperature": 0.7, "max_tokens": 150}

        # 캐시 저장
        await cache.cache_response(
            test_prompt,
            test_response,
            test_params,
            "openai",
            "gpt-3.5-turbo",
            1500.0
        )

        # 캐시 조회
        cached_response = await cache.get_cached_response(
            test_prompt,
            test_params,
            "openai",
            "gpt-3.5-turbo"
        )
        print("캐시 히트:", cached_response is not None)

        # 다른 파라미터로 조회 (캐시 미스여야 함)
        cached_response = await cache.get_cached_response(
            test_prompt,
            {"temperature": 0.8, "max_tokens": 150},
            "openai",
            "gpt-3.5-turbo"
        )
        print("다른 파라미터 캐시 미스:", cached_response is None)

        # 통계 확인
        stats = await cache.get_cache_stats()
        print("캐시 통계:", stats)

        # 효율성 분석
        efficiency = await cache.get_cache_efficiency()
        print("캐시 효율성:", efficiency)

    asyncio.run(test_llm_cache())