"""
Application Settings
환경변수 기반 애플리케이션 설정 관리
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Settings:
    """애플리케이션 설정 클래스"""

    # 프로젝트 정보
    PROJECT_NAME: str = "JOBER AI"
    PROJECT_VERSION: str = "2.0.0"
    PROJECT_DESCRIPTION: str = "AI 기반 알림톡 템플릿 생성 서비스"

    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"

    # API 설정
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["*"]  # 실제 배포 시 특정 도메인으로 제한

    # LLM 제공자 설정
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "30"))

    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_TIMEOUT: int = int(os.getenv("OPENAI_TIMEOUT", "30"))

    PRIMARY_LLM_PROVIDER: str = os.getenv("PRIMARY_LLM_PROVIDER", "openai")

    # 경로 설정
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    CACHE_DIR: Path = STORAGE_DIR / "cache"
    CHROMA_DB_DIR: Path = STORAGE_DIR / "chroma_db"
    LOGS_DIR: Path = STORAGE_DIR / "logs"

    # 데이터베이스 설정 (ChromaDB)
    CHROMA_DB_PERSIST_DIR: str = str(CHROMA_DB_DIR)

    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 보안 설정 (필수)
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # 성능 설정
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))

    # 템플릿 생성 설정
    MAX_TEMPLATE_LENGTH: int = int(os.getenv("MAX_TEMPLATE_LENGTH", "1000"))
    MAX_VARIABLES_COUNT: int = int(os.getenv("MAX_VARIABLES_COUNT", "10"))

    # 템플릿 포맷팅 설정
    TEMPLATE_USE_SPECIAL_CHARACTERS: bool = os.getenv("TEMPLATE_USE_SPECIAL_CHARACTERS", "true").lower() == "true"
    TEMPLATE_VARIABLE_PREFIX: str = os.getenv("TEMPLATE_VARIABLE_PREFIX", "▶")
    TEMPLATE_USE_EMOJI: bool = os.getenv("TEMPLATE_USE_EMOJI", "false").lower() == "true"

    # 연락처 정보 처리 설정
    REQUIRE_USER_CONTACT_INFO: bool = os.getenv("REQUIRE_USER_CONTACT_INFO", "true").lower() == "true"
    AUTO_ADD_CONTACT_FIELDS: bool = os.getenv("AUTO_ADD_CONTACT_FIELDS", "false").lower() == "true"

    def __init__(self):
        """설정 초기화 시 필요한 디렉토리 생성"""
        self._ensure_directories()

    def _ensure_directories(self):
        """필요한 디렉토리들 생성"""
        directories = [
            self.DATA_DIR,
            self.STORAGE_DIR,
            self.CACHE_DIR,
            self.CHROMA_DB_DIR,
            self.LOGS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_database_url(self) -> str:
        """데이터베이스 URL 반환 (미래 확장용)"""
        return os.getenv("DATABASE_URL", "sqlite:///./storage/jober_ai.db")

    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.DEBUG

    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return not self.DEBUG

    def validate_settings(self) -> list[str]:
        """설정 유효성 검사"""
        errors = []

        # LLM API 키 확인
        if not self.GEMINI_API_KEY and not self.OPENAI_API_KEY:
            errors.append("GEMINI_API_KEY 또는 OPENAI_API_KEY 중 하나는 필수입니다")

        # 기타 필수 설정 검사
        if not self.SECRET_KEY:
            errors.append("보안을 위해 SECRET_KEY 환경변수를 반드시 설정해주세요")

        return errors


# 전역 설정 인스턴스
settings = Settings()


def get_settings() -> Settings:
    """설정 인스턴스 반환"""
    return settings