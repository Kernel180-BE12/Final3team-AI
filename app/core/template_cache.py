#!/usr/bin/env python3
"""
메모리 기반 템플릿 캐시 시스템
유사한 요청에 대해 캐시된 템플릿을 재사용하여 성능 향상
"""

import hashlib
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json


@dataclass
class CachedTemplate:
    """캐시된 템플릿 데이터"""
    template_id: str
    user_input: str
    industry: str
    purpose: str
    template_content: str
    variables: List[str]
    similarity_score: float
    created_at: datetime
    access_count: int
    last_accessed: datetime

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data


class TemplatePatternCache:
    """메모리 기반 지능형 템플릿 캐시 시스템"""

    def __init__(self, max_cache_size: int = 1000, ttl_hours: int = 24):
        """
        Args:
            max_cache_size: 최대 캐시 항목 수
            ttl_hours: 캐시 생존 시간 (시간)
        """
        self.cache: Dict[str, CachedTemplate] = {}
        self.max_cache_size = max_cache_size
        self.ttl = timedelta(hours=ttl_hours)
        self.similarity_threshold = 0.75  # 유사도 임계값

        # 통계
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_cleanups": 0
        }

    async def get_similar_template(
        self,
        user_input: str,
        industry: str,
        purpose: str
    ) -> Optional[Dict]:
        """
        유사한 템플릿을 캐시에서 검색

        Args:
            user_input: 사용자 입력
            industry: 업종
            purpose: 목적

        Returns:
            유사한 템플릿 데이터 또는 None
        """
        self.stats["total_requests"] += 1

        # 캐시 정리 (만료된 항목 제거)
        await self._cleanup_expired_cache()

        # 정확히 일치하는 항목 먼저 찾기
        exact_key = self._generate_cache_key(user_input, industry, purpose)
        if exact_key in self.cache:
            cached_item = self.cache[exact_key]
            cached_item.access_count += 1
            cached_item.last_accessed = datetime.now()
            self.stats["cache_hits"] += 1

            return {
                "id": cached_item.template_id,
                "content": cached_item.template_content,
                "variables": cached_item.variables,
                "status": "completed",
                "industry": cached_item.industry,
                "purpose": cached_item.purpose,
                "cache_hit": True,
                "similarity_score": 1.0
            }

        # 유사한 항목 찾기
        best_match = await self._find_similar_template(user_input, industry, purpose)

        if best_match:
            self.stats["cache_hits"] += 1
            return best_match

        self.stats["cache_misses"] += 1
        return None

    async def cache_template(
        self,
        user_input: str,
        industry: str,
        purpose: str,
        template: Dict
    ) -> bool:
        """
        생성된 템플릿을 캐시에 저장

        Args:
            user_input: 사용자 입력
            industry: 업종
            purpose: 목적
            template: 템플릿 데이터

        Returns:
            저장 성공 여부
        """
        try:
            # 캐시 크기 제한 확인
            if len(self.cache) >= self.max_cache_size:
                await self._evict_lru_items(count=int(self.max_cache_size * 0.1))

            cache_key = self._generate_cache_key(user_input, industry, purpose)

            cached_template = CachedTemplate(
                template_id=template.get("id", cache_key),
                user_input=user_input,
                industry=industry,
                purpose=purpose,
                template_content=template.get("content", ""),
                variables=template.get("variables", []),
                similarity_score=1.0,
                created_at=datetime.now(),
                access_count=1,
                last_accessed=datetime.now()
            )

            self.cache[cache_key] = cached_template
            return True

        except Exception as e:
            print(f"템플릿 캐시 저장 오류: {e}")
            return False

    async def _find_similar_template(
        self,
        user_input: str,
        industry: str,
        purpose: str
    ) -> Optional[Dict]:
        """유사한 템플릿 찾기"""

        best_match = None
        best_similarity = 0.0

        for cached_item in self.cache.values():
            # 업종과 목적이 일치하는 경우만 고려
            if cached_item.industry != industry or cached_item.purpose != purpose:
                continue

            # 텍스트 유사도 계산
            similarity = self._calculate_text_similarity(
                user_input,
                cached_item.user_input
            )

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = cached_item

        if best_match:
            # 접근 통계 업데이트
            best_match.access_count += 1
            best_match.last_accessed = datetime.now()

            return {
                "id": best_match.template_id,
                "content": best_match.template_content,
                "variables": best_match.variables,
                "status": "completed",
                "industry": best_match.industry,
                "purpose": best_match.purpose,
                "cache_hit": True,
                "similarity_score": best_similarity
            }

        return None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """간단한 텍스트 유사도 계산 (Jaccard 유사도)"""
        if not text1 or not text2:
            return 0.0

        # 단어 단위로 분할
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard 유사도 계산
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _generate_cache_key(self, user_input: str, industry: str, purpose: str) -> str:
        """캐시 키 생성"""
        content = f"{user_input.strip()}:{industry}:{purpose}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

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

        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "total_requests": total_requests,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_ratio": round(hit_ratio, 3),
            "cache_cleanups": self.stats["cache_cleanups"],
            "memory_usage_estimate_kb": len(self.cache) * 2  # 대략적인 추정
        }

    async def clear_cache(self):
        """캐시 전체 비우기"""
        self.cache.clear()
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_cleanups": 0
        }

    async def get_cached_items_info(self) -> List[Dict]:
        """캐시된 항목들의 정보 조회"""
        items = []
        for key, cached_item in self.cache.items():
            items.append({
                "cache_key": key,
                "template_id": cached_item.template_id,
                "industry": cached_item.industry,
                "purpose": cached_item.purpose,
                "user_input_preview": cached_item.user_input[:50] + "..." if len(cached_item.user_input) > 50 else cached_item.user_input,
                "created_at": cached_item.created_at.isoformat(),
                "access_count": cached_item.access_count,
                "last_accessed": cached_item.last_accessed.isoformat()
            })

        # 접근 횟수 기준으로 정렬
        items.sort(key=lambda x: x["access_count"], reverse=True)
        return items


# 싱글톤 인스턴스
_template_cache_instance = None

def get_template_cache() -> TemplatePatternCache:
    """템플릿 캐시 싱글톤 인스턴스 반환"""
    global _template_cache_instance
    if _template_cache_instance is None:
        _template_cache_instance = TemplatePatternCache()
    return _template_cache_instance


# 편의 함수들
async def get_cached_template(user_input: str, industry: str, purpose: str) -> Optional[Dict]:
    """캐시된 템플릿 조회 편의 함수"""
    cache = get_template_cache()
    return await cache.get_similar_template(user_input, industry, purpose)


async def save_template_to_cache(user_input: str, industry: str, purpose: str, template: Dict) -> bool:
    """템플릿 캐시 저장 편의 함수"""
    cache = get_template_cache()
    return await cache.cache_template(user_input, industry, purpose, template)


if __name__ == "__main__":
    import asyncio

    async def test_cache():
        cache = TemplatePatternCache()

        # 테스트 데이터
        test_template = {
            "id": "test_001",
            "content": "안녕하세요, {고객명}님. {상품명} 주문이 완료되었습니다.",
            "variables": ["고객명", "상품명"],
            "status": "completed",
            "industry": "이커머스",
            "purpose": "주문확인"
        }

        # 캐시 저장
        await cache.cache_template(
            "주문 완료 알림 템플릿 만들어주세요",
            "이커머스",
            "주문확인",
            test_template
        )

        # 캐시 조회 (정확한 일치)
        result = await cache.get_similar_template(
            "주문 완료 알림 템플릿 만들어주세요",
            "이커머스",
            "주문확인"
        )
        print("정확한 일치:", result is not None)

        # 캐시 조회 (유사한 입력)
        result = await cache.get_similar_template(
            "주문 완료 메시지 템플릿 생성해주세요",
            "이커머스",
            "주문확인"
        )
        print("유사한 일치:", result is not None)

        # 통계 확인
        stats = await cache.get_cache_stats()
        print("캐시 통계:", stats)

    asyncio.run(test_cache())