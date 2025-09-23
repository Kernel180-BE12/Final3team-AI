#!/usr/bin/env python3
"""
API Result 래퍼 클래스
Java 백엔드팀 ApiResult<T> 구조와 호환되는 응답 래퍼
"""

from typing import Optional, Generic, TypeVar
from pydantic import BaseModel
from datetime import datetime

T = TypeVar('T')

class ErrorResponse(BaseModel):
    """에러 응답 모델 - Java ErrorResponse와 호환"""
    code: str
    message: str
    timestamp: str = None

    def __init__(self, **data):
        if not data.get('timestamp'):
            data['timestamp'] = datetime.now().isoformat() + "Z"
        super().__init__(**data)

class ApiResult(BaseModel, Generic[T]):
    """
    Java 백엔드팀과 호환되는 API 응답 래퍼

    성공 시: data에 실제 데이터, message/error는 null
    실패 시: data/message는 null, error에 에러 정보
    """
    data: Optional[T] = None
    message: Optional[str] = None
    error: Optional[ErrorResponse] = None

    @classmethod
    def ok(cls, data: T, message: Optional[str] = None):
        """성공 응답 생성"""
        return cls(data=data, message=message, error=None)

    @classmethod
    def error(cls, code: str, message: str):
        """에러 응답 생성"""
        return cls(data=None, message=None, error=ErrorResponse(code=code, message=message))