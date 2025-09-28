"""
Performance Logging Utility
요청 처리 시간을 상세하게 로깅하는 유틸리티
"""

import time
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

class PerformanceLogger:
    """성능 로깅을 위한 클래스"""

    def __init__(self, log_dir: str = "storage/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 성능 로거 설정
        self.perf_logger = logging.getLogger('performance')
        self.perf_logger.setLevel(logging.INFO)

        # 파일 핸들러 설정 (일별 로그 파일)
        log_filename = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # 핸들러가 이미 있는지 확인 후 추가
        if not self.perf_logger.handlers:
            self.perf_logger.addHandler(file_handler)

        # 콘솔 출력도 추가
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # 중복 방지
        if len(self.perf_logger.handlers) < 2:
            self.perf_logger.addHandler(console_handler)

    def log_request_timing(self,
                          request_id: str,
                          user_id: int,
                          request_content: str,
                          total_time: float,
                          stage_times: Dict[str, float],
                          metadata: Dict[str, Any] = None):
        """
        요청 처리 시간을 로깅합니다.

        Args:
            request_id: 요청 고유 ID
            user_id: 사용자 ID
            request_content: 요청 내용 (앞 50자만)
            total_time: 총 처리 시간 (초)
            stage_times: 단계별 처리 시간
            metadata: 추가 메타데이터
        """

        # 요청 내용은 50자로 제한 (로그 가독성)
        truncated_content = request_content[:50] + "..." if len(request_content) > 50 else request_content

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "user_id": user_id,
            "request_content": truncated_content,
            "total_time": round(total_time, 3),
            "stage_times": {k: round(v, 3) for k, v in stage_times.items()},
            "metadata": metadata or {}
        }

        # JSON 형태로 로깅
        self.perf_logger.info(f"REQUEST_TIMING: {json.dumps(log_data, ensure_ascii=False)}")

        # 콘솔용 간단한 요약
        stage_summary = " | ".join([f"{k}: {v:.2f}s" for k, v in stage_times.items()])
        print(f"[TIMING] Total: {total_time:.2f}s | {stage_summary} | Request: '{truncated_content}'")

    def log_stage_start(self, stage_name: str, request_id: str = None):
        """단계 시작 로깅"""
        msg = f"[{request_id}] {stage_name} 시작" if request_id else f"{stage_name} 시작"
        self.perf_logger.info(f"STAGE_START: {msg}")
        return time.time()

    def log_stage_end(self, stage_name: str, start_time: float, request_id: str = None):
        """단계 완료 로깅"""
        duration = time.time() - start_time
        msg = f"[{request_id}] {stage_name} 완료 - {duration:.3f}s" if request_id else f"{stage_name} 완료 - {duration:.3f}s"
        self.perf_logger.info(f"STAGE_END: {msg}")
        return duration

    def log_error(self, error_msg: str, request_id: str = None, duration: float = None):
        """에러 로깅"""
        timestamp = datetime.now().isoformat()
        msg = f"ERROR: [{request_id}] {error_msg}" if request_id else f"ERROR: {error_msg}"
        if duration:
            msg += f" (처리시간: {duration:.3f}s)"

        self.perf_logger.error(msg)


class TimingContext:
    """타이밍 측정을 위한 컨텍스트 매니저"""

    def __init__(self, logger: PerformanceLogger, stage_name: str, request_id: str = None):
        self.logger = logger
        self.stage_name = stage_name
        self.request_id = request_id
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = self.logger.log_stage_start(self.stage_name, self.request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = self.logger.log_stage_end(self.stage_name, self.start_time, self.request_id)


# 전역 인스턴스
_performance_logger = None

def get_performance_logger() -> PerformanceLogger:
    """전역 성능 로거 인스턴스 반환"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger