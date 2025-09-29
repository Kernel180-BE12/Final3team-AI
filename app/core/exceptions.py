#!/usr/bin/env python3
"""
커스텀 예외 클래스들
Java 백엔드와 호환되는 에러 코드 및 HTTP 상태 코드 매핑
"""

from typing import Optional, Any


class AIException(Exception):
    """AI 서비스 관련 기본 예외 클래스"""

    def __init__(self, code: str, message: str, status_code: int = 500, details: Optional[Any] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ProcessingTimeoutException(AIException):
    """처리 시간 초과 예외 (408)"""

    def __init__(self, message: str = "템플릿 생성 시간이 초과되었습니다. 다시 시도해주세요.", details: Optional[Any] = None):
        super().__init__(
            code="PROCESSING_TIMEOUT",
            message=message,
            status_code=408,
            details=details
        )


class APIQuotaExceededException(AIException):
    """API 할당량 초과 예외 (429)"""

    def __init__(self, message: str = "API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요.", details: Optional[Any] = None):
        super().__init__(
            code="API_QUOTA_EXCEEDED",
            message=message,
            status_code=429,
            details=details
        )


class InappropriateRequestException(AIException):
    """부적절한 요청 예외 (400)"""

    def __init__(self, message: str = "비즈니스 알림톡에 적합하지 않은 요청입니다.", details: Optional[Any] = None):
        super().__init__(
            code="INAPPROPRIATE_REQUEST",
            message=message,
            status_code=400,
            details=details
        )


class LanguageValidationException(AIException):
    """언어 검증 오류 예외 (400)"""

    def __init__(self, message: str = "Please enter in Korean. English-only input cannot generate KakaoTalk templates.", details: Optional[Any] = None):
        super().__init__(
            code="LANGUAGE_VALIDATION_ERROR",
            message=message,
            status_code=400,
            details=details
        )


class ProfanityDetectedException(AIException):
    """비속어 검출 예외 (400)"""

    def __init__(self, message: str = "부적절한 언어가 감지되었습니다.", details: Optional[Any] = None):
        super().__init__(
            code="PROFANITY_DETECTED",
            message=message,
            status_code=400,
            details=details
        )


class PolicyViolationException(AIException):
    """정책 위반 예외 (400)"""

    def __init__(self, message: str = "정책에 위반되는 내용이 포함되었습니다.", details: Optional[Any] = None):
        super().__init__(
            code="POLICY_VIOLATION",
            message=message,
            status_code=400,
            details=details
        )


class TemplateGenerationFailedException(AIException):
    """템플릿 생성 실패 예외 (500)"""

    def __init__(self, message: str = "템플릿 생성에 실패했습니다.", details: Optional[Any] = None):
        super().__init__(
            code="TEMPLATE_GENERATION_FAILED",
            message=message,
            status_code=500,
            details=details
        )


class ValidationException(AIException):
    """입력 검증 실패 예외 (422)"""

    def __init__(self, message: str = "입력값 검증에 실패했습니다.", details: Optional[Any] = None):
        super().__init__(
            code="VALIDATION_ERROR",
            message=message,
            status_code=422,
            details=details
        )