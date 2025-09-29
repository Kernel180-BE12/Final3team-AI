"""
JOBER AI - Main Application Entry Point
FastAPI 기반 알림톡 템플릿 생성 서비스
"""

import logging
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn

from app.api import templates, health, sessions
from app.utils.llm_provider_manager import get_llm_manager
from config.llm_providers import get_llm_manager as get_config_manager
from config.settings import Settings
from app.core.exceptions import AIException
from app.dto.api_result import ApiResult, ErrorResponse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
print(f"[DEBUG] Loaded Gemini Model: {Settings.GEMINI_MODEL}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시
    logger.info("JOBER AI 서비스 시작")

    # LLM 제공자 초기화 확인
    try:
        llm_manager = get_llm_manager()
        status = llm_manager.get_status()
        logger.info(f"LLM 제공자 상태: {status}")

        if not status['available_providers']:
            logger.warning("사용 가능한 LLM 제공자가 없습니다. API 키를 확인해주세요.")
        else:
            logger.info(f"사용 가능한 LLM 제공자: {status['available_providers']}")

    except Exception as e:
        logger.error(f"LLM 제공자 초기화 실패: {e}")

    yield

    # 종료 시
    logger.info("JOBER AI 서비스 종료")


# FastAPI 앱 생성
app = FastAPI(
    title=Settings.PROJECT_NAME,
    description=Settings.PROJECT_DESCRIPTION,
    version=Settings.PROJECT_VERSION,
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Template Generation",
            "description": "AI 기반 템플릿 생성 및 관리 API"
        },
        {
            "name": "Real-time Chat",
            "description": "실시간 채팅 및 변수 업데이트 API"
        },
        {
            "name": "Session Management",
            "description": "세션 관리 및 모니터링 API"
        },
        {
            "name": "System",
            "description": "시스템 상태 및 헬스체크 API"
        }
    ]
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AIException 핸들러 (커스텀 AI 예외)
@app.exception_handler(AIException)
async def ai_exception_handler(request: Request, exc: AIException):
    """AI 예외 처리 - Java 백엔드와 호환되는 응답 형식"""
    logger.error(f"AI 예외 발생: {exc.code} - {exc.message}", exc_info=True)

    error_response = ErrorResponse(
        code=exc.code,
        message=exc.message,
        details=exc.details
    )

    api_result = ApiResult(data=None, message=None, error=error_response)

    return JSONResponse(
        status_code=exc.status_code,
        content=api_result.dict()
    )


# HTTPException 핸들러
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 처리"""
    logger.error(f"HTTP 예외 발생: {exc.status_code} - {exc.detail}")

    error_response = ErrorResponse(
        code="HTTP_ERROR",
        message=exc.detail
    )

    api_result = ApiResult(data=None, message=None, error=error_response)

    return JSONResponse(
        status_code=exc.status_code,
        content=api_result.dict()
    )


# 글로벌 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """글로벌 예외 처리"""
    logger.error(f"글로벌 예외 발생: {exc}", exc_info=True)

    error_response = ErrorResponse(
        code="INTERNAL_SERVER_ERROR",
        message="서버 내부 오류가 발생했습니다.",
        details=str(exc) if os.getenv("DEBUG") == "true" else None
    )

    api_result = ApiResult(data=None, message=None, error=error_response)

    return JSONResponse(
        status_code=500,
        content=api_result.dict()
    )

# 라우터 등록
app.include_router(
    health.router,
    prefix="/ai"
)

app.include_router(
    templates.router,
    prefix="/ai"
)

app.include_router(
    sessions.router,
    prefix="/ai"
)

class RootResponse(BaseModel):
    """루트 엔드포인트 응답 모델"""
    service: str
    version: str
    description: str
    status: str
    docs: str

# 루트 엔드포인트
@app.get("/", tags=["System"], response_model=RootResponse)
async def root():
    """루트 엔드포인트"""
    return {
        "service": Settings.PROJECT_NAME,
        "version": Settings.PROJECT_VERSION,
        "description": Settings.PROJECT_DESCRIPTION,
        "status": "running",
        "docs": "/docs"
    }

class LLMStatusMainResponse(BaseModel):
    """메인 LLM 상태 응답 모델"""
    available_providers: list
    primary_provider: str
    failure_counts: Dict[str, int]
    gemini_configured: bool
    openai_configured: bool

# LLM 제공자 상태 확인 엔드포인트
@app.get("/ai/llm/status", tags=["System"], response_model=LLMStatusMainResponse)
async def llm_status():
    """LLM 제공자 상태 확인"""
    try:
        manager = get_llm_manager()
        return manager.get_status()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 상태 확인 실패: {e}"
        )


if __name__ == "__main__":
    # 개발 서버 실행
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"개발 서버 시작: http://{host}:{port}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )