"""
Health Check API Router
서비스 상태 확인 및 헬스체크 엔드포인트
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel
import psutil
import time
from datetime import datetime

from app.utils.llm_provider_manager import get_llm_manager
from app.dto.api_result import ApiResult, ErrorResponse
from config.settings import Settings

router = APIRouter()


# Response Models
class BasicHealthResponse(BaseModel):
    """기본 헬스체크 응답 모델"""
    status: str
    timestamp: str
    service: str
    version: str


class SystemInfo(BaseModel):
    """시스템 정보 모델"""
    cpu_usage_percent: float
    memory: Dict[str, float]
    disk: Dict[str, float]


class ServiceInfo(BaseModel):
    """서비스 정보 모델"""
    name: str
    version: str
    uptime: float


class Dependencies(BaseModel):
    """의존성 상태 모델"""
    database: str
    cache: str


class DetailedHealthResponse(BaseModel):
    """상세 헬스체크 응답 모델"""
    status: str
    timestamp: str
    service: ServiceInfo
    system: SystemInfo
    llm_providers: Dict[str, Any]
    dependencies: Dependencies


class LLMTestInfo(BaseModel):
    """LLM 테스트 정보 모델"""
    success: bool
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    response_time_seconds: Optional[float] = None
    response_preview: Optional[str] = None
    error: Optional[str] = None


class LLMHealthResponse(BaseModel):
    """LLM 헬스체크 응답 모델"""
    status: str
    timestamp: str
    llm_test: LLMTestInfo
    provider_status: Dict[str, Any]


class LLMResetResponse(BaseModel):
    """LLM 리셋 응답 모델"""
    status: str
    message: str
    timestamp: str
    provider_status: Dict[str, Any]


@router.get("/health", tags=["System"], response_model=BasicHealthResponse)
async def health_check() -> Dict[str, Any]:
    """
    기본 헬스체크
    서비스가 정상적으로 실행 중인지 확인
    """
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": Settings.PROJECT_NAME,
        "version": Settings.PROJECT_VERSION
    }
    return ApiResult.ok(health_data)


@router.get("/health/detailed", tags=["System"], response_model=DetailedHealthResponse)
async def detailed_health_check() -> Dict[str, Any]:
    """
    상세 헬스체크
    시스템 리소스, LLM 제공자 상태 등 포함
    """
    try:
        # 시스템 리소스 정보
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # LLM 제공자 상태
        llm_manager = get_llm_manager()
        llm_status = llm_manager.get_status()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": {
                "name": Settings.PROJECT_NAME,
                "version": Settings.PROJECT_VERSION,
                "uptime": time.time()  # 실제로는 시작 시간 기록 필요
            },
            "system": {
                "cpu_usage_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "usage_percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "usage_percent": round((disk.used / disk.total) * 100, 2)
                }
            },
            "llm_providers": llm_status,
            "dependencies": {
                "database": "healthy",  # 실제 DB 연결 확인 필요
                "cache": "healthy"      # 실제 캐시 연결 확인 필요
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/health/llm", tags=["System"], response_model=LLMHealthResponse)
async def llm_health_check() -> Dict[str, Any]:
    """
    LLM 제공자 전용 헬스체크
    실제 API 호출 테스트 포함
    """
    try:
        llm_manager = get_llm_manager()

        # 간단한 테스트 프롬프트
        test_prompt = "안녕하세요! 이것은 헬스체크 테스트입니다. '정상'이라고 답변해주세요."

        start_time = time.time()
        response, provider, model = await llm_manager.invoke_with_fallback(test_prompt)
        end_time = time.time()

        response_time = end_time - start_time

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "llm_test": {
                "success": True,
                "provider_used": provider,
                "model_used": model,
                "response_time_seconds": round(response_time, 2),
                "response_preview": response[:100] if response else None
            },
            "provider_status": llm_manager.get_status()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "llm_test": {
                "success": False,
                "error": str(e)
            },
            "provider_status": llm_manager.get_status() if llm_manager else None
        }


@router.post("/health/llm/reset", tags=["System"], response_model=LLMResetResponse)
async def reset_llm_failures() -> Dict[str, Any]:
    """
    LLM 제공자 실패 카운트 리셋
    관리자용 엔드포인트
    """
    try:
        llm_manager = get_llm_manager()
        llm_manager.reset_failure_counts()

        return {
            "status": "success",
            "message": "LLM 제공자 실패 카운트가 리셋되었습니다",
            "timestamp": datetime.now().isoformat(),
            "provider_status": llm_manager.get_status()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"실패 카운트 리셋 실패: {e}",
                "timestamp": datetime.now().isoformat()
            }
        )