import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Key - 절대로 하드코딩하지 않고 .env에서만 로드
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")