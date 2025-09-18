import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Key - 절대로 하드코딩하지 않고 .env에서만 로드
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")

# 하위 호환성을 위해 GEMINI_API_KEY도 제공
GEMINI_API_KEY = GOOGLE_API_KEY

