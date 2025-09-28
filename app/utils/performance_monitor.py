#!/usr/bin/env python3
"""
성능 모니터링 시스템
실시간 성능 메트릭 수집, 분석 및 리포팅
"""

import time
import asyncio
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
import json
import statistics
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """메트릭 타입"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATIO = "cache_hit_ratio"
    CONCURRENT_USERS = "concurrent_users"


@dataclass
class PerformanceMetric:
    """성능 메트릭 데이터"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "value": self.value,
            "labels": self.labels,
            "metadata": self.metadata
        }


@dataclass
class RequestMetrics:
    """요청별 메트릭"""
    request_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    user_input_length: int = 0
    response_length: int = 0
    industry: str = ""
    purpose: str = ""
    cache_hit: bool = False
    llm_provider: str = ""
    session_id: str = ""

    def finish(self, status_code: int, response_length: int = 0, error_message: str = None):
        """요청 완료 처리"""
        self.end_time = datetime.now()
        self.response_time = (self.end_time - self.start_time).total_seconds()
        self.status_code = status_code
        self.response_length = response_length
        self.error_message = error_message

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "response_time": self.response_time,
            "status_code": self.status_code,
            "error_message": self.error_message,
            "user_input_length": self.user_input_length,
            "response_length": self.response_length,
            "industry": self.industry,
            "purpose": self.purpose,
            "cache_hit": self.cache_hit,
            "llm_provider": self.llm_provider,
            "session_id": self.session_id
        }


class PerformanceMonitor:
    """성능 모니터링 시스템"""

    def __init__(self, retention_hours: int = 24, metric_interval_seconds: int = 30):
        """
        Args:
            retention_hours: 메트릭 보관 시간
            metric_interval_seconds: 시스템 메트릭 수집 간격
        """
        self.retention_period = timedelta(hours=retention_hours)
        self.metric_interval = metric_interval_seconds

        # 메트릭 저장소
        self.metrics: deque[PerformanceMetric] = deque()
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.completed_requests: deque[RequestMetrics] = deque()

        # 실시간 통계
        self.current_stats = {
            "active_requests": 0,
            "total_requests": 0,
            "total_errors": 0,
            "average_response_time": 0.0,
            "requests_per_minute": 0.0,
            "cache_hit_ratio": 0.0,
            "system_cpu_percent": 0.0,
            "system_memory_percent": 0.0,
            "error_rate": 0.0
        }

        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # 성능 임계값
        self.thresholds = {
            "response_time_warning": 5.0,      # 5초
            "response_time_critical": 10.0,    # 10초
            "error_rate_warning": 0.05,        # 5%
            "error_rate_critical": 0.10,       # 10%
            "memory_warning": 80.0,            # 80%
            "memory_critical": 90.0,           # 90%
            "cpu_warning": 80.0,               # 80%
            "cpu_critical": 90.0               # 90%
        }

        # 알림 상태
        self.alerts = defaultdict(list)

    async def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("성능 모니터링 시작")

    async def stop_monitoring(self):
        """모니터링 중지"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("성능 모니터링 중지")

    def start_request(
        self,
        request_id: str,
        user_input: str = "",
        industry: str = "",
        purpose: str = "",
        session_id: str = ""
    ) -> RequestMetrics:
        """요청 시작 추적"""
        request_metrics = RequestMetrics(
            request_id=request_id,
            start_time=datetime.now(),
            user_input_length=len(user_input),
            industry=industry,
            purpose=purpose,
            session_id=session_id
        )

        self.request_metrics[request_id] = request_metrics
        self.current_stats["active_requests"] += 1
        self.current_stats["total_requests"] += 1

        return request_metrics

    def finish_request(
        self,
        request_id: str,
        status_code: int,
        response_content: str = "",
        error_message: str = None,
        cache_hit: bool = False,
        llm_provider: str = ""
    ):
        """요청 완료 추적"""
        if request_id not in self.request_metrics:
            logger.warning(f"Unknown request ID: {request_id}")
            return

        request_metrics = self.request_metrics.pop(request_id)
        request_metrics.finish(status_code, len(response_content), error_message)
        request_metrics.cache_hit = cache_hit
        request_metrics.llm_provider = llm_provider

        # 완료된 요청 저장
        self.completed_requests.append(request_metrics)
        self.current_stats["active_requests"] -= 1

        # 에러 카운트 업데이트
        if status_code >= 400:
            self.current_stats["total_errors"] += 1

        # 응답 시간 메트릭 추가
        if request_metrics.response_time:
            self._add_metric(
                MetricType.RESPONSE_TIME,
                request_metrics.response_time,
                labels={
                    "industry": request_metrics.industry,
                    "purpose": request_metrics.purpose,
                    "status_code": str(status_code),
                    "cache_hit": str(cache_hit),
                    "llm_provider": llm_provider
                }
            )

        # 실시간 통계 업데이트
        self._update_real_time_stats()

    def _add_metric(self, metric_type: MetricType, value: float, labels: Dict[str, str] = None):
        """메트릭 추가"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            labels=labels or {}
        )
        self.metrics.append(metric)

        # 오래된 메트릭 정리
        self._cleanup_old_metrics()

    def _update_real_time_stats(self):
        """실시간 통계 업데이트"""
        if not self.completed_requests:
            return

        # 최근 완료된 요청들 (최근 1시간)
        recent_time = datetime.now() - timedelta(hours=1)
        recent_requests = [r for r in self.completed_requests if r.end_time and r.end_time > recent_time]

        if recent_requests:
            # 평균 응답 시간
            response_times = [r.response_time for r in recent_requests if r.response_time]
            if response_times:
                self.current_stats["average_response_time"] = statistics.mean(response_times)

            # 분당 요청 수 (최근 5분)
            recent_5min = datetime.now() - timedelta(minutes=5)
            recent_5min_requests = [r for r in recent_requests if r.end_time > recent_5min]
            self.current_stats["requests_per_minute"] = len(recent_5min_requests) / 5

            # 에러율
            error_requests = [r for r in recent_requests if r.status_code and r.status_code >= 400]
            self.current_stats["error_rate"] = len(error_requests) / len(recent_requests)

            # 캐시 히트율
            cache_hits = [r for r in recent_requests if r.cache_hit]
            self.current_stats["cache_hit_ratio"] = len(cache_hits) / len(recent_requests)

    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # 시스템 메트릭 수집
                await self._collect_system_metrics()

                # 알림 확인
                await self._check_alerts()

                # 대기
                await asyncio.sleep(self.metric_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")

    async def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            self.current_stats["system_cpu_percent"] = cpu_percent
            self._add_metric(MetricType.CPU_USAGE, cpu_percent)

            # 메모리 사용률
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            self.current_stats["system_memory_percent"] = memory_percent
            self._add_metric(MetricType.MEMORY_USAGE, memory_percent)

            # 처리량
            self._add_metric(MetricType.THROUGHPUT, self.current_stats["requests_per_minute"])

            # 에러율
            self._add_metric(MetricType.ERROR_RATE, self.current_stats["error_rate"])

            # 캐시 히트율
            self._add_metric(MetricType.CACHE_HIT_RATIO, self.current_stats["cache_hit_ratio"])

            # 동시 사용자 수
            self._add_metric(MetricType.CONCURRENT_USERS, len(set(
                r.session_id for r in self.request_metrics.values() if r.session_id
            )))

        except Exception as e:
            logger.error(f"시스템 메트릭 수집 오류: {e}")

    async def _check_alerts(self):
        """임계값 기반 알림 확인"""
        current_time = datetime.now()

        # 응답 시간 알림
        avg_response_time = self.current_stats["average_response_time"]
        if avg_response_time > self.thresholds["response_time_critical"]:
            self._add_alert("CRITICAL", f"평균 응답 시간이 매우 높습니다: {avg_response_time:.2f}초")
        elif avg_response_time > self.thresholds["response_time_warning"]:
            self._add_alert("WARNING", f"평균 응답 시간이 높습니다: {avg_response_time:.2f}초")

        # 에러율 알림
        error_rate = self.current_stats["error_rate"]
        if error_rate > self.thresholds["error_rate_critical"]:
            self._add_alert("CRITICAL", f"에러율이 매우 높습니다: {error_rate:.1%}")
        elif error_rate > self.thresholds["error_rate_warning"]:
            self._add_alert("WARNING", f"에러율이 높습니다: {error_rate:.1%}")

        # 메모리 사용률 알림
        memory_percent = self.current_stats["system_memory_percent"]
        if memory_percent > self.thresholds["memory_critical"]:
            self._add_alert("CRITICAL", f"메모리 사용률이 매우 높습니다: {memory_percent:.1f}%")
        elif memory_percent > self.thresholds["memory_warning"]:
            self._add_alert("WARNING", f"메모리 사용률이 높습니다: {memory_percent:.1f}%")

        # CPU 사용률 알림
        cpu_percent = self.current_stats["system_cpu_percent"]
        if cpu_percent > self.thresholds["cpu_critical"]:
            self._add_alert("CRITICAL", f"CPU 사용률이 매우 높습니다: {cpu_percent:.1f}%")
        elif cpu_percent > self.thresholds["cpu_warning"]:
            self._add_alert("WARNING", f"CPU 사용률이 높습니다: {cpu_percent:.1f}%")

    def _add_alert(self, level: str, message: str):
        """알림 추가"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.alerts[level].append(alert)

        # 알림 히스토리 제한 (각 레벨별 최대 100개)
        if len(self.alerts[level]) > 100:
            self.alerts[level].pop(0)

        logger.warning(f"성능 알림 [{level}]: {message}")

    def _cleanup_old_metrics(self):
        """오래된 메트릭 정리"""
        cutoff_time = datetime.now() - self.retention_period

        # 메트릭 정리
        while self.metrics and self.metrics[0].timestamp < cutoff_time:
            self.metrics.popleft()

        # 완료된 요청 정리
        while self.completed_requests and self.completed_requests[0].end_time and self.completed_requests[0].end_time < cutoff_time:
            self.completed_requests.popleft()

    def get_current_stats(self) -> Dict[str, Any]:
        """현재 통계 조회"""
        return {
            **self.current_stats,
            "last_updated": datetime.now().isoformat()
        }

    def get_performance_report(self, hours: int = 1) -> Dict[str, Any]:
        """성능 리포트 생성"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # 최근 완료된 요청들
        recent_requests = [r for r in self.completed_requests if r.end_time and r.end_time > cutoff_time]

        if not recent_requests:
            return {"message": "최근 데이터가 없습니다", "period_hours": hours}

        # 응답 시간 통계
        response_times = [r.response_time for r in recent_requests if r.response_time]
        response_time_stats = {}
        if response_times:
            response_time_stats = {
                "average": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "p95": self._percentile(response_times, 95),
                "p99": self._percentile(response_times, 99)
            }

        # 업종별 통계
        industry_stats = defaultdict(list)
        for req in recent_requests:
            if req.industry and req.response_time:
                industry_stats[req.industry].append(req.response_time)

        industry_report = {}
        for industry, times in industry_stats.items():
            industry_report[industry] = {
                "count": len(times),
                "average_response_time": statistics.mean(times),
                "median_response_time": statistics.median(times)
            }

        # 에러 분석
        error_requests = [r for r in recent_requests if r.status_code and r.status_code >= 400]
        error_analysis = {
            "total_errors": len(error_requests),
            "error_rate": len(error_requests) / len(recent_requests),
            "error_types": defaultdict(int)
        }

        for error_req in error_requests:
            status_code = f"{error_req.status_code}xx"
            error_analysis["error_types"][status_code] += 1

        # 캐시 효율성
        cache_hits = [r for r in recent_requests if r.cache_hit]
        cache_analysis = {
            "total_requests": len(recent_requests),
            "cache_hits": len(cache_hits),
            "cache_hit_ratio": len(cache_hits) / len(recent_requests),
            "cache_time_saved": sum(r.response_time for r in cache_hits if r.response_time)
        }

        # LLM 프로바이더별 통계
        provider_stats = defaultdict(list)
        for req in recent_requests:
            if req.llm_provider and req.response_time:
                provider_stats[req.llm_provider].append(req.response_time)

        provider_report = {}
        for provider, times in provider_stats.items():
            provider_report[provider] = {
                "count": len(times),
                "average_response_time": statistics.mean(times),
                "median_response_time": statistics.median(times)
            }

        return {
            "period_hours": hours,
            "summary": {
                "total_requests": len(recent_requests),
                "successful_requests": len([r for r in recent_requests if r.status_code and r.status_code < 400]),
                "average_response_time": response_time_stats.get("average", 0),
                "requests_per_hour": len(recent_requests) / hours,
                "error_rate": error_analysis["error_rate"],
                "cache_hit_ratio": cache_analysis["cache_hit_ratio"]
            },
            "response_time_stats": response_time_stats,
            "industry_performance": industry_report,
            "error_analysis": dict(error_analysis),
            "cache_analysis": cache_analysis,
            "provider_performance": provider_report,
            "generated_at": datetime.now().isoformat()
        }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """백분위수 계산"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def get_metrics_by_type(self, metric_type: MetricType, hours: int = 1) -> List[Dict]:
        """타입별 메트릭 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_metrics = [
            m.to_dict() for m in self.metrics
            if m.metric_type == metric_type and m.timestamp > cutoff_time
        ]
        return filtered_metrics

    def get_alerts(self, level: str = None, hours: int = 24) -> List[Dict]:
        """알림 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if level:
            alerts = self.alerts.get(level, [])
        else:
            alerts = []
            for alert_list in self.alerts.values():
                alerts.extend(alert_list)

        # 시간 필터링
        filtered_alerts = [
            alert for alert in alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]

        # 시간순 정렬 (최신 순)
        filtered_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        return filtered_alerts

    def clear_alerts(self, level: str = None):
        """알림 정리"""
        if level:
            self.alerts[level].clear()
        else:
            self.alerts.clear()

    def export_metrics(self, hours: int = 24) -> Dict:
        """메트릭 내보내기"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return {
            "export_info": {
                "generated_at": datetime.now().isoformat(),
                "period_hours": hours,
                "total_metrics": len(self.metrics),
                "total_requests": len(self.completed_requests)
            },
            "metrics": [m.to_dict() for m in self.metrics if m.timestamp > cutoff_time],
            "completed_requests": [r.to_dict() for r in self.completed_requests if r.end_time and r.end_time > cutoff_time],
            "current_stats": self.get_current_stats(),
            "alerts": dict(self.alerts)
        }


# 싱글톤 인스턴스
_performance_monitor_instance = None

def get_performance_monitor() -> PerformanceMonitor:
    """성능 모니터 싱글톤 인스턴스 반환"""
    global _performance_monitor_instance
    if _performance_monitor_instance is None:
        _performance_monitor_instance = PerformanceMonitor()
    return _performance_monitor_instance


# 편의 함수들
def start_request_monitoring(request_id: str, user_input: str = "", industry: str = "", purpose: str = "", session_id: str = "") -> RequestMetrics:
    """요청 모니터링 시작"""
    monitor = get_performance_monitor()
    return monitor.start_request(request_id, user_input, industry, purpose, session_id)

def finish_request_monitoring(request_id: str, status_code: int, response_content: str = "", error_message: str = None, cache_hit: bool = False, llm_provider: str = ""):
    """요청 모니터링 완료"""
    monitor = get_performance_monitor()
    monitor.finish_request(request_id, status_code, response_content, error_message, cache_hit, llm_provider)

def get_current_performance_stats() -> Dict[str, Any]:
    """현재 성능 통계 조회"""
    monitor = get_performance_monitor()
    return monitor.get_current_stats()

def get_performance_report(hours: int = 1) -> Dict[str, Any]:
    """성능 리포트 조회"""
    monitor = get_performance_monitor()
    return monitor.get_performance_report(hours)


if __name__ == "__main__":
    import asyncio
    import uuid

    async def test_performance_monitor():
        monitor = PerformanceMonitor()
        await monitor.start_monitoring()

        # 테스트 요청들 시뮬레이션
        for i in range(10):
            request_id = str(uuid.uuid4())

            # 요청 시작
            request_metrics = monitor.start_request(
                request_id,
                f"테스트 요청 {i}",
                "이커머스",
                "주문확인",
                f"session_{i % 3}"
            )

            # 처리 시간 시뮬레이션
            await asyncio.sleep(0.1)

            # 요청 완료
            status_code = 200 if i < 8 else 500  # 80% 성공률
            monitor.finish_request(
                request_id,
                status_code,
                f"응답 내용 {i}",
                "오류 메시지" if status_code == 500 else None,
                cache_hit=(i % 3 == 0),
                llm_provider="openai"
            )

        # 잠시 대기 (메트릭 수집)
        await asyncio.sleep(2)

        # 현재 통계 확인
        stats = monitor.get_current_stats()
        print("현재 통계:", json.dumps(stats, indent=2, ensure_ascii=False))

        # 성능 리포트 확인
        report = monitor.get_performance_report(1)
        print("성능 리포트:", json.dumps(report, indent=2, ensure_ascii=False))

        await monitor.stop_monitoring()

    asyncio.run(test_performance_monitor())