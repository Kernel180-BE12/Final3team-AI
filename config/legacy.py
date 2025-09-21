"""
Legacy Config Support
기존 코드와의 하위 호환성을 위한 설정
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Key - 절대로 하드코딩하지 않고 .env에서만 로드
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = GOOGLE_API_KEY

# OpenAI API Key 추가
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 경고 메시지 (API 키가 없을 때)
if not GOOGLE_API_KEY and not OPENAI_API_KEY:
    print("⚠️ 경고: GEMINI_API_KEY 또는 OPENAI_API_KEY가 설정되지 않았습니다.")
    print("   환경변수나 .env 파일에 API 키를 설정해주세요.")