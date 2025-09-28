#!/usr/bin/env python3
"""
LLM 호출 배치 처리 시스템
여러 LLM 요청을 배치로 묶어서 처리하여 처리량 향상
"""

import asyncio
import time
from typing import List, Callable, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """배치 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BatchRequest:
    """배치 요청 데이터"""
    request_id: str
    prompt: str
    model_params: Dict[str, Any]
    callback: Callable[[str, Optional[str], Optional[str]], None]  # (request_id, response, error)
    priority: BatchPriority = BatchPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: float = 30.0
    provider: str = "default"
    model_name: str = "default"

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


@dataclass
class BatchStats:
    """배치 처리 통계"""
    total_requests: int = 0
    total_batches: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    average_batch_size: float = 0.0
    average_processing_time: float = 0.0
    throughput_requests_per_second: float = 0.0
    total_time_saved_seconds: float = 0.0


class LLMBatchProcessor:
    """LLM 호출 배치 처리 시스템"""

    def __init__(
        self,
        batch_size: int = 5,
        batch_timeout: float = 2.0,
        max_concurrent_batches: int = 3,
        worker_count: int = 5
    ):
        """
        Args:
            batch_size: 배치당 최대 요청 수
            batch_timeout: 배치 타임아웃 (초)
            max_concurrent_batches: 최대 동시 배치 수
            worker_count: 워커 수
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrent_batches = max_concurrent_batches
        self.worker_count = worker_count

        # 큐들 (우선순위별)
        self.priority_queues: Dict[BatchPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in BatchPriority
        }

        # 처리 중인 요청들
        self.active_requests: Dict[str, BatchRequest] = {}

        # 통계
        self.stats = BatchStats()

        # 배치 처리 상태
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.batch_collector_task: Optional[asyncio.Task] = None

        # 성능 모니터링
        self.start_time = time.time()
        self.last_batch_time = time.time()

    async def start(self):
        """배치 프로세서 시작"""
        if self.is_running:
            return

        self.is_running = True
        self.start_time = time.time()

        # 워커 태스크들 시작
        self.worker_tasks = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.worker_count)
        ]

        # 배치 수집기 시작
        self.batch_collector_task = asyncio.create_task(self._batch_collector())

        logger.info(f"LLM 배치 프로세서 시작: {self.worker_count}개 워커, 배치 크기 {self.batch_size}")

    async def stop(self):
        """배치 프로세서 중지"""
        if not self.is_running:
            return

        self.is_running = False

        # 배치 수집기 중지
        if self.batch_collector_task:
            self.batch_collector_task.cancel()
            try:
                await self.batch_collector_task
            except asyncio.CancelledError:
                pass

        # 워커들 중지
        for task in self.worker_tasks:
            task.cancel()

        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()

        logger.info("LLM 배치 프로세서 중지")

    async def add_request(
        self,
        prompt: str,
        callback: Callable[[str, Optional[str], Optional[str]], None],
        model_params: Dict[str, Any] = None,
        priority: BatchPriority = BatchPriority.NORMAL,
        timeout_seconds: float = 30.0,
        provider: str = "default",
        model_name: str = "default",
        request_id: str = None
    ) -> str:
        """
        배치에 요청 추가

        Args:
            prompt: LLM 프롬프트
            callback: 결과 콜백 함수
            model_params: 모델 파라미터
            priority: 요청 우선순위
            timeout_seconds: 타임아웃
            provider: LLM 프로바이더
            model_name: 모델명
            request_id: 요청 ID (없으면 자동 생성)

        Returns:
            요청 ID
        """
        request = BatchRequest(
            request_id=request_id or str(uuid.uuid4()),
            prompt=prompt,
            model_params=model_params or {},
            callback=callback,
            priority=priority,
            timeout_seconds=timeout_seconds,
            provider=provider,
            model_name=model_name
        )

        # 큐에 추가
        await self.priority_queues[priority].put(request)

        # 활성 요청에 추가
        self.active_requests[request.request_id] = request

        # 통계 업데이트
        self.stats.total_requests += 1

        return request.request_id

    async def _batch_collector(self):
        """배치 수집기 - 우선순위에 따라 요청들을 배치로 묶음"""
        while self.is_running:
            try:
                batch = await self._collect_batch()
                if batch:
                    # 배치 처리 태스크 생성
                    asyncio.create_task(self._process_batch(batch))

                await asyncio.sleep(0.1)  # 100ms 간격으로 체크

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"배치 수집기 오류: {e}")

    async def _collect_batch(self) -> List[BatchRequest]:
        """우선순위에 따라 배치 수집"""
        batch = []
        batch_start_time = time.time()

        # 우선순위 순서대로 요청 수집 (URGENT > HIGH > NORMAL > LOW)
        for priority in [BatchPriority.URGENT, BatchPriority.HIGH, BatchPriority.NORMAL, BatchPriority.LOW]:
            queue = self.priority_queues[priority]

            # 해당 우선순위 큐에서 요청 수집
            while len(batch) < self.batch_size and not queue.empty():
                try:
                    request = queue.get_nowait()
                    batch.append(request)
                except asyncio.QueueEmpty:
                    break

            # URGENT나 HIGH 우선순위 요청이 있으면 즉시 배치 처리
            if batch and priority in [BatchPriority.URGENT, BatchPriority.HIGH]:
                break

            # 배치 크기가 다 찼으면 중단
            if len(batch) >= self.batch_size:
                break

        # 타임아웃 체크
        if batch and (time.time() - batch_start_time) >= self.batch_timeout:
            return batch

        # 최소 배치 크기 체크 (우선순위가 높은 요청이 있거나, 타임아웃된 경우)
        if batch and (
            any(req.priority in [BatchPriority.URGENT, BatchPriority.HIGH] for req in batch) or
            (time.time() - self.last_batch_time) >= self.batch_timeout
        ):
            return batch

        return []

    async def _worker(self, worker_name: str):
        """배치 처리 워커"""
        while self.is_running:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"워커 {worker_name} 오류: {e}")

    async def _process_batch(self, batch: List[BatchRequest]):
        """배치 처리"""
        if not batch:
            return

        batch_start_time = time.time()
        batch_id = str(uuid.uuid4())[:8]

        logger.info(f"배치 {batch_id} 처리 시작: {len(batch)}개 요청")

        try:
            # 모델 파라미터별로 그룹화
            param_groups = self._group_by_model_params(batch)

            # 각 그룹별로 병렬 처리
            tasks = []
            for group_requests in param_groups.values():
                task = asyncio.create_task(self._process_request_group(group_requests))
                tasks.append(task)

            # 모든 그룹 처리 완료 대기
            await asyncio.gather(*tasks, return_exceptions=True)

            # 통계 업데이트
            processing_time = time.time() - batch_start_time
            self._update_batch_stats(batch, processing_time)
            self.last_batch_time = time.time()

            logger.info(f"배치 {batch_id} 처리 완료: {processing_time:.2f}초")

        except Exception as e:
            logger.error(f"배치 {batch_id} 처리 오류: {e}")

            # 오류 시 모든 요청에 오류 콜백 호출
            for request in batch:
                try:
                    await request.callback(request.request_id, None, str(e))
                    self.stats.failed_requests += 1
                except Exception:
                    pass
                finally:
                    self.active_requests.pop(request.request_id, None)

    def _group_by_model_params(self, batch: List[BatchRequest]) -> Dict[str, List[BatchRequest]]:
        """모델 파라미터별로 요청 그룹화"""
        groups = {}

        for request in batch:
            # 모델 파라미터와 프로바이더로 키 생성
            param_key = f"{request.provider}:{request.model_name}:{hash(str(sorted(request.model_params.items())))}"

            if param_key not in groups:
                groups[param_key] = []
            groups[param_key].append(request)

        return groups

    async def _process_request_group(self, requests: List[BatchRequest]):
        """동일한 모델 파라미터를 가진 요청 그룹 처리"""
        try:
            # 실제 LLM API 호출 (병렬)
            tasks = []
            for request in requests:
                task = asyncio.create_task(self._call_llm_api(request))
                tasks.append(task)

            # 모든 요청 처리 완료 대기
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 처리
            for request, result in zip(requests, results):
                try:
                    if isinstance(result, Exception):
                        await request.callback(request.request_id, None, str(result))
                        self.stats.failed_requests += 1
                    else:
                        await request.callback(request.request_id, result, None)
                        self.stats.completed_requests += 1

                except Exception as e:
                    logger.error(f"콜백 실행 오류 {request.request_id}: {e}")
                    self.stats.failed_requests += 1

                finally:
                    # 활성 요청에서 제거
                    self.active_requests.pop(request.request_id, None)

        except Exception as e:
            logger.error(f"요청 그룹 처리 오류: {e}")

    async def _call_llm_api(self, request: BatchRequest) -> str:
        """실제 LLM API 호출"""
        try:
            # 실제 구현에서는 LLM 프로바이더 호출
            # 여기서는 시뮬레이션
            await asyncio.sleep(0.5)  # API 호출 시뮬레이션

            # 실제 구현 예시:
            # from app.utils.llm_provider_manager import ainvoke_llm_with_fallback
            # response, provider, model = await ainvoke_llm_with_fallback(request.prompt)
            # return response

            return f"Response for: {request.prompt[:50]}..."

        except asyncio.TimeoutError:
            raise Exception(f"LLM API 호출 타임아웃: {request.timeout_seconds}초")
        except Exception as e:
            raise Exception(f"LLM API 호출 오류: {e}")

    def _update_batch_stats(self, batch: List[BatchRequest], processing_time: float):
        """배치 통계 업데이트"""
        self.stats.total_batches += 1
        batch_size = len(batch)

        # 평균 배치 크기 업데이트
        total_batches = self.stats.total_batches
        current_avg_size = self.stats.average_batch_size
        self.stats.average_batch_size = (
            (current_avg_size * (total_batches - 1) + batch_size) / total_batches
        )

        # 평균 처리 시간 업데이트
        current_avg_time = self.stats.average_processing_time
        self.stats.average_processing_time = (
            (current_avg_time * (total_batches - 1) + processing_time) / total_batches
        )

        # 처리량 계산
        total_runtime = time.time() - self.start_time
        self.stats.throughput_requests_per_second = (
            self.stats.completed_requests / total_runtime if total_runtime > 0 else 0
        )

        # 시간 절약 추정 (배치 처리 vs 개별 처리)
        individual_time_estimate = batch_size * 2.0  # 개별 호출 시 예상 시간
        time_saved = max(0, individual_time_estimate - processing_time)
        self.stats.total_time_saved_seconds += time_saved

    async def get_stats(self) -> Dict:
        """배치 처리 통계 조회"""
        total_runtime = time.time() - self.start_time
        success_rate = (
            self.stats.completed_requests / max(1, self.stats.total_requests) * 100
        )

        return {
            "total_requests": self.stats.total_requests,
            "completed_requests": self.stats.completed_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "total_batches": self.stats.total_batches,
            "average_batch_size": round(self.stats.average_batch_size, 2),
            "average_processing_time_seconds": round(self.stats.average_processing_time, 2),
            "throughput_requests_per_second": round(self.stats.throughput_requests_per_second, 2),
            "total_time_saved_seconds": round(self.stats.total_time_saved_seconds, 2),
            "active_requests_count": len(self.active_requests),
            "queue_sizes": {
                priority.name: queue.qsize()
                for priority, queue in self.priority_queues.items()
            },
            "total_runtime_seconds": round(total_runtime, 2)
        }

    async def get_active_requests(self) -> List[Dict]:
        """활성 요청 목록 조회"""
        active_list = []
        current_time = time.time()

        for request in self.active_requests.values():
            wait_time = current_time - request.created_at.timestamp()
            active_list.append({
                "request_id": request.request_id,
                "priority": request.priority.name,
                "provider": request.provider,
                "model_name": request.model_name,
                "prompt_preview": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt,
                "wait_time_seconds": round(wait_time, 2),
                "timeout_seconds": request.timeout_seconds
            })

        return sorted(active_list, key=lambda x: x["wait_time_seconds"], reverse=True)

    async def cancel_request(self, request_id: str) -> bool:
        """요청 취소"""
        if request_id in self.active_requests:
            request = self.active_requests.pop(request_id)
            try:
                await request.callback(request_id, None, "Request cancelled")
            except Exception:
                pass
            return True
        return False

    async def force_process_all(self):
        """모든 대기 중인 요청 즉시 처리"""
        all_requests = []

        # 모든 우선순위 큐에서 요청 수집
        for queue in self.priority_queues.values():
            while not queue.empty():
                try:
                    request = queue.get_nowait()
                    all_requests.append(request)
                except asyncio.QueueEmpty:
                    break

        # 배치로 나누어 처리
        for i in range(0, len(all_requests), self.batch_size):
            batch = all_requests[i:i + self.batch_size]
            await self._process_batch(batch)


# 싱글톤 인스턴스
_batch_processor_instance = None

def get_batch_processor() -> LLMBatchProcessor:
    """배치 프로세서 싱글톤 인스턴스 반환"""
    global _batch_processor_instance
    if _batch_processor_instance is None:
        _batch_processor_instance = LLMBatchProcessor()
    return _batch_processor_instance


if __name__ == "__main__":
    import asyncio

    async def test_callback(request_id: str, response: Optional[str], error: Optional[str]):
        if error:
            print(f"요청 {request_id} 실패: {error}")
        else:
            print(f"요청 {request_id} 성공: {response[:50]}...")

    async def test_batch_processor():
        processor = LLMBatchProcessor(batch_size=3, batch_timeout=1.0)
        await processor.start()

        # 테스트 요청들 추가
        requests = [
            "Create a welcome email template",
            "Generate order confirmation message",
            "Write a product description",
            "Create a newsletter template",
            "Generate a thank you message"
        ]

        for i, prompt in enumerate(requests):
            await processor.add_request(
                prompt=prompt,
                callback=test_callback,
                priority=BatchPriority.HIGH if i < 2 else BatchPriority.NORMAL
            )

        # 잠시 대기
        await asyncio.sleep(5)

        # 통계 확인
        stats = await processor.get_stats()
        print("배치 처리 통계:", stats)

        # 프로세서 중지
        await processor.stop()

    asyncio.run(test_batch_processor())