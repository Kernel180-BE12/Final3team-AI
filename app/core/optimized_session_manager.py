#!/usr/bin/env python3
"""
최적화된 세션 관리 시스템
메모리 효율적이고 성능이 최적화된 세션 관리
"""

import time
import asyncio
import gc
import sys
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import OrderedDict
import json
import psutil
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionMetrics:
    """세션 메트릭"""
    created_at: datetime
    last_activity: datetime
    request_count: int = 0
    memory_usage_bytes: int = 0
    processing_time_total: float = 0.0
    average_response_time: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def update_response_time(self, response_time: float):
        """응답 시간 통계 업데이트"""
        self.processing_time_total += response_time
        self.average_response_time = self.processing_time_total / max(1, self.request_count)

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "request_count": self.request_count,
            "memory_usage_kb": round(self.memory_usage_bytes / 1024, 2),
            "average_response_time": round(self.average_response_time, 3),
            "error_count": self.error_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": round(self.cache_hits / max(1, self.cache_hits + self.cache_misses), 3)
        }


@dataclass
class OptimizedSession:
    """최적화된 세션 객체"""
    session_id: str
    user_data: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    context_cache: Dict[str, Any] = field(default_factory=dict)
    template_cache: Dict[str, Any] = field(default_factory=dict)
    metrics: SessionMetrics = field(default_factory=lambda: SessionMetrics(
        created_at=datetime.now(),
        last_activity=datetime.now()
    ))

    # 메모리 최적화 설정
    MAX_CONVERSATION_HISTORY = 20  # 최대 대화 히스토리 수
    MAX_CONTEXT_CACHE_SIZE = 50    # 최대 컨텍스트 캐시 크기
    MAX_TEMPLATE_CACHE_SIZE = 10   # 최대 템플릿 캐시 크기

    def update_activity(self, response_time: float = 0.0):
        """활동 업데이트"""
        self.metrics.last_activity = datetime.now()
        self.metrics.request_count += 1
        if response_time > 0:
            self.metrics.update_response_time(response_time)

    def add_conversation_turn(self, user_input: str, response: str, metadata: Dict = None):
        """대화 턴 추가"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "metadata": metadata or {}
        }

        self.conversation_history.append(turn)

        # 히스토리 크기 제한
        if len(self.conversation_history) > self.MAX_CONVERSATION_HISTORY:
            # 오래된 대화를 요약으로 압축
            self._compress_old_conversations()

    def _compress_old_conversations(self):
        """오래된 대화를 요약으로 압축"""
        if len(self.conversation_history) <= self.MAX_CONVERSATION_HISTORY:
            return

        # 최신 대화는 유지, 오래된 대화는 요약
        recent_conversations = self.conversation_history[-self.MAX_CONVERSATION_HISTORY:]
        old_conversations = self.conversation_history[:-self.MAX_CONVERSATION_HISTORY]

        # 간단한 요약 생성
        summary = {
            "type": "conversation_summary",
            "timestamp": datetime.now().isoformat(),
            "conversation_count": len(old_conversations),
            "time_range": {
                "start": old_conversations[0]["timestamp"] if old_conversations else None,
                "end": old_conversations[-1]["timestamp"] if old_conversations else None
            },
            "topics": self._extract_conversation_topics(old_conversations),
            "request_count": len(old_conversations)
        }

        # 요약 + 최신 대화로 교체
        self.conversation_history = [summary] + recent_conversations

    def _extract_conversation_topics(self, conversations: List[Dict]) -> List[str]:
        """대화에서 주요 토픽 추출"""
        topics = set()
        keywords = {
            "템플릿": ["템플릿", "template"],
            "이메일": ["이메일", "email", "메일"],
            "주문": ["주문", "order", "구매"],
            "알림": ["알림", "notification", "알람"],
            "계약": ["계약", "contract"],
            "마케팅": ["마케팅", "marketing", "광고"]
        }

        for conv in conversations:
            user_input = conv.get("user_input", "").lower()
            for topic, topic_keywords in keywords.items():
                if any(keyword in user_input for keyword in topic_keywords):
                    topics.add(topic)

        return list(topics)[:5]  # 최대 5개 토픽

    def cache_context(self, key: str, value: Any, ttl_minutes: int = 30):
        """컨텍스트 캐시"""
        # 캐시 크기 제한
        if len(self.context_cache) >= self.MAX_CONTEXT_CACHE_SIZE:
            self._evict_old_cache_items(self.context_cache, 5)

        self.context_cache[key] = {
            "value": value,
            "cached_at": datetime.now(),
            "ttl_minutes": ttl_minutes,
            "access_count": 1
        }

    def get_cached_context(self, key: str) -> Optional[Any]:
        """캐시된 컨텍스트 조회"""
        if key not in self.context_cache:
            self.metrics.cache_misses += 1
            return None

        cached_item = self.context_cache[key]
        cached_at = cached_item["cached_at"]
        ttl = timedelta(minutes=cached_item["ttl_minutes"])

        # TTL 확인
        if datetime.now() - cached_at > ttl:
            del self.context_cache[key]
            self.metrics.cache_misses += 1
            return None

        # 접근 통계 업데이트
        cached_item["access_count"] += 1
        self.metrics.cache_hits += 1
        return cached_item["value"]

    def cache_template(self, template_key: str, template_data: Dict):
        """템플릿 캐시"""
        # 캐시 크기 제한
        if len(self.template_cache) >= self.MAX_TEMPLATE_CACHE_SIZE:
            self._evict_old_cache_items(self.template_cache, 2)

        self.template_cache[template_key] = {
            "data": template_data,
            "cached_at": datetime.now(),
            "access_count": 1
        }

    def get_cached_template(self, template_key: str) -> Optional[Dict]:
        """캐시된 템플릿 조회"""
        if template_key not in self.template_cache:
            return None

        cached_item = self.template_cache[template_key]
        cached_item["access_count"] += 1
        return cached_item["data"]

    def _evict_old_cache_items(self, cache_dict: Dict, count: int):
        """오래된 캐시 항목 제거"""
        if not cache_dict:
            return

        # 접근 시간 기준으로 정렬 (캐시된 시간)
        sorted_items = sorted(
            cache_dict.items(),
            key=lambda x: x[1].get("cached_at", datetime.min)
        )

        # 오래된 항목부터 제거
        for i in range(min(count, len(sorted_items))):
            key = sorted_items[i][0]
            del cache_dict[key]

    def calculate_memory_usage(self) -> int:
        """세션 메모리 사용량 계산"""
        memory = 0

        # 각 필드의 메모리 사용량 추정
        memory += sys.getsizeof(self.user_data)
        memory += sys.getsizeof(self.conversation_history)
        memory += sys.getsizeof(self.context_cache)
        memory += sys.getsizeof(self.template_cache)

        # 중첩 객체들의 메모리도 계산
        for item in self.conversation_history:
            if isinstance(item, dict):
                memory += sys.getsizeof(str(item))

        for value in self.context_cache.values():
            if isinstance(value, dict):
                memory += sys.getsizeof(str(value))

        for value in self.template_cache.values():
            if isinstance(value, dict):
                memory += sys.getsizeof(str(value))

        self.metrics.memory_usage_bytes = memory
        return memory

    def cleanup_expired_cache(self):
        """만료된 캐시 정리"""
        current_time = datetime.now()
        expired_keys = []

        for key, cached_item in self.context_cache.items():
            cached_at = cached_item.get("cached_at", current_time)
            ttl = timedelta(minutes=cached_item.get("ttl_minutes", 30))

            if current_time - cached_at > ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.context_cache[key]

    def get_session_info(self) -> Dict:
        """세션 정보 조회"""
        self.calculate_memory_usage()
        age_minutes = (datetime.now() - self.metrics.created_at).total_seconds() / 60
        inactive_minutes = (datetime.now() - self.metrics.last_activity).total_seconds() / 60

        return {
            "session_id": self.session_id,
            "age_minutes": round(age_minutes, 2),
            "inactive_minutes": round(inactive_minutes, 2),
            "conversation_turns": len([h for h in self.conversation_history if h.get("type") != "conversation_summary"]),
            "total_history_items": len(self.conversation_history),
            "context_cache_size": len(self.context_cache),
            "template_cache_size": len(self.template_cache),
            "metrics": self.metrics.to_dict()
        }


class OptimizedSessionManager:
    """최적화된 세션 관리자"""

    def __init__(
        self,
        max_sessions: int = 1000,
        session_ttl_minutes: int = 60,
        memory_limit_mb: int = 300,
        cleanup_interval_seconds: int = 300
    ):
        """
        Args:
            max_sessions: 최대 세션 수
            session_ttl_minutes: 세션 TTL (분)
            memory_limit_mb: 메모리 한계 (MB)
            cleanup_interval_seconds: 정리 주기 (초)
        """
        # 세션 저장소 (LRU 순서 유지)
        self.sessions: OrderedDict[str, OptimizedSession] = OrderedDict()

        # 설정
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval_seconds

        # 모니터링
        self.total_memory_usage = 0
        self.cleanup_stats = {
            "expired_sessions_removed": 0,
            "lru_sessions_removed": 0,
            "memory_cleanups": 0,
            "last_cleanup": datetime.now(),
            "peak_memory_usage": 0,
            "peak_session_count": 0
        }

        # 백그라운드 태스크
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False

    async def start(self):
        """세션 관리자 시작"""
        if self.is_running:
            return

        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("최적화된 세션 관리자 시작")

    async def stop(self):
        """세션 관리자 중지"""
        if not self.is_running:
            return

        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("최적화된 세션 관리자 중지")

    async def get_or_create_session(self, session_id: str) -> OptimizedSession:
        """세션 조회 또는 생성"""
        # 기존 세션 확인
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # TTL 확인
            if datetime.now() - session.metrics.last_activity > self.session_ttl:
                await self._remove_session(session_id)
            else:
                # LRU 순서 업데이트
                self.sessions.move_to_end(session_id)
                return session

        # 새 세션 생성
        return await self._create_new_session(session_id)

    async def _create_new_session(self, session_id: str) -> OptimizedSession:
        """새 세션 생성"""
        # 메모리 및 세션 수 확인
        await self._ensure_capacity()

        # 새 세션 생성
        session = OptimizedSession(session_id=session_id)
        self.sessions[session_id] = session

        # 메모리 사용량 업데이트
        memory_usage = session.calculate_memory_usage()
        self.total_memory_usage += memory_usage

        # 통계 업데이트
        current_count = len(self.sessions)
        if current_count > self.cleanup_stats["peak_session_count"]:
            self.cleanup_stats["peak_session_count"] = current_count

        if self.total_memory_usage > self.cleanup_stats["peak_memory_usage"]:
            self.cleanup_stats["peak_memory_usage"] = self.total_memory_usage

        return session

    async def _remove_session(self, session_id: str):
        """세션 제거"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            self.total_memory_usage -= session.metrics.memory_usage_bytes
            del self.sessions[session_id]

    async def update_session_activity(
        self,
        session_id: str,
        user_input: str = None,
        response: str = None,
        response_time: float = 0.0,
        metadata: Dict = None
    ):
        """세션 활동 업데이트"""
        session = await self.get_or_create_session(session_id)

        # 활동 업데이트
        session.update_activity(response_time)

        # 대화 턴 추가
        if user_input and response:
            session.add_conversation_turn(user_input, response, metadata)

        # 메모리 사용량 재계산
        old_memory = session.metrics.memory_usage_bytes
        new_memory = session.calculate_memory_usage()
        self.total_memory_usage += (new_memory - old_memory)

    async def cache_session_context(self, session_id: str, key: str, value: Any, ttl_minutes: int = 30):
        """세션 컨텍스트 캐시"""
        session = await self.get_or_create_session(session_id)
        session.cache_context(key, value, ttl_minutes)

    async def get_session_context(self, session_id: str, key: str) -> Optional[Any]:
        """세션 컨텍스트 조회"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return session.get_cached_context(key)

    async def _cleanup_loop(self):
        """정리 루프"""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_sessions()
                await self._optimize_memory()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"세션 정리 오류: {e}")

    async def cleanup_sessions(self):
        """만료된 세션 정리"""
        current_time = datetime.now()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if current_time - session.metrics.last_activity > self.session_ttl:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self._remove_session(session_id)
            self.cleanup_stats["expired_sessions_removed"] += 1

        if expired_sessions:
            logger.info(f"만료된 세션 {len(expired_sessions)}개 제거")

        # 개별 세션의 만료된 캐시도 정리
        for session in self.sessions.values():
            session.cleanup_expired_cache()

        self.cleanup_stats["last_cleanup"] = current_time

    async def _optimize_memory(self):
        """메모리 사용량 최적화"""
        # 시스템 메모리 확인
        try:
            memory_info = psutil.virtual_memory()
            system_memory_percent = memory_info.percent
        except:
            system_memory_percent = 0

        # 메모리 한계 초과 또는 시스템 메모리 부족
        if (self.total_memory_usage > self.memory_limit_bytes or system_memory_percent > 85):
            await self._reduce_memory_usage()
            self.cleanup_stats["memory_cleanups"] += 1

    async def _reduce_memory_usage(self):
        """메모리 사용량 감소"""
        # 1. 세션별 캐시 정리
        for session in self.sessions.values():
            session.cleanup_expired_cache()

        # 2. LRU 세션 제거 (메모리 사용량이 많은 순)
        if len(self.sessions) > self.max_sessions * 0.8:
            # 메모리 사용량 기준으로 정렬
            sessions_by_memory = sorted(
                self.sessions.items(),
                key=lambda x: x[1].calculate_memory_usage(),
                reverse=True
            )

            # 상위 10% 제거
            remove_count = max(1, int(len(sessions_by_memory) * 0.1))
            for i in range(remove_count):
                session_id = sessions_by_memory[i][0]
                await self._remove_session(session_id)
                self.cleanup_stats["lru_sessions_removed"] += 1

        # 3. 가비지 컬렉션 강제 실행
        gc.collect()

        # 4. 메모리 사용량 재계산
        self.total_memory_usage = sum(
            session.calculate_memory_usage() for session in self.sessions.values()
        )

    async def _ensure_capacity(self):
        """용량 확보"""
        # 세션 수 확인
        if len(self.sessions) >= self.max_sessions:
            remove_count = int(self.max_sessions * 0.1)  # 10% 제거
            await self._remove_lru_sessions(remove_count)

        # 메모리 확인
        if self.total_memory_usage > self.memory_limit_bytes * 0.9:
            await self._reduce_memory_usage()

    async def _remove_lru_sessions(self, count: int):
        """LRU 방식으로 세션 제거"""
        session_ids_to_remove = list(self.sessions.keys())[:count]

        for session_id in session_ids_to_remove:
            await self._remove_session(session_id)
            self.cleanup_stats["lru_sessions_removed"] += 1

    async def get_manager_stats(self) -> Dict:
        """관리자 통계 조회"""
        active_sessions = len(self.sessions)
        memory_usage_mb = self.total_memory_usage / (1024 * 1024)

        # 세션별 통계
        if active_sessions > 0:
            total_requests = sum(s.metrics.request_count for s in self.sessions.values())
            total_errors = sum(s.metrics.error_count for s in self.sessions.values())
            avg_memory_per_session = memory_usage_mb / active_sessions
            avg_requests_per_session = total_requests / active_sessions
        else:
            total_requests = total_errors = avg_memory_per_session = avg_requests_per_session = 0

        return {
            "active_sessions": active_sessions,
            "max_sessions": self.max_sessions,
            "memory_usage_mb": round(memory_usage_mb, 2),
            "memory_limit_mb": round(self.memory_limit_bytes / (1024 * 1024), 2),
            "memory_utilization_percent": round((memory_usage_mb / (self.memory_limit_bytes / (1024 * 1024))) * 100, 2),
            "average_memory_per_session_kb": round(avg_memory_per_session * 1024, 2),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "average_requests_per_session": round(avg_requests_per_session, 2),
            "session_ttl_minutes": self.session_ttl.total_seconds() / 60,
            "cleanup_stats": self.cleanup_stats.copy()
        }

    async def get_session_details(self, session_id: str) -> Optional[Dict]:
        """세션 상세 정보"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return session.get_session_info()

    async def get_top_sessions(self, limit: int = 10, sort_by: str = "memory") -> List[Dict]:
        """상위 세션 목록"""
        sessions_info = []

        for session in self.sessions.values():
            info = session.get_session_info()
            sessions_info.append(info)

        # 정렬
        if sort_by == "memory":
            sessions_info.sort(key=lambda x: x["metrics"]["memory_usage_kb"], reverse=True)
        elif sort_by == "requests":
            sessions_info.sort(key=lambda x: x["metrics"]["request_count"], reverse=True)
        elif sort_by == "age":
            sessions_info.sort(key=lambda x: x["age_minutes"], reverse=True)

        return sessions_info[:limit]


# 싱글톤 인스턴스
_session_manager_instance = None

def get_session_manager() -> OptimizedSessionManager:
    """세션 관리자 싱글톤 인스턴스 반환"""
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = OptimizedSessionManager()
    return _session_manager_instance


if __name__ == "__main__":
    import asyncio

    async def test_session_manager():
        manager = OptimizedSessionManager(max_sessions=5, session_ttl_minutes=1)
        await manager.start()

        # 테스트 세션들 생성
        for i in range(7):
            session_id = f"test_session_{i}"
            session = await manager.get_or_create_session(session_id)

            # 테스트 활동
            await manager.update_session_activity(
                session_id,
                f"테스트 요청 {i}",
                f"테스트 응답 {i}",
                response_time=0.5
            )

            # 컨텍스트 캐시
            await manager.cache_session_context(session_id, f"key_{i}", f"value_{i}")

        # 통계 확인
        stats = await manager.get_manager_stats()
        print("관리자 통계:", json.dumps(stats, indent=2, ensure_ascii=False))

        # 상위 세션 확인
        top_sessions = await manager.get_top_sessions(3, "memory")
        print("상위 세션들:", json.dumps(top_sessions, indent=2, ensure_ascii=False))

        # 정리 대기
        await asyncio.sleep(2)

        await manager.stop()

    asyncio.run(test_session_manager())