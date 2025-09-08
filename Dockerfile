# JOBER AI - 알림톡 템플릿 생성기 Docker 이미지
# Python 3.11.13 기반 FastAPI 서버

FROM python:3.11.13-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 빌드 도구 설치
# gcc, g++: FAISS, numpy 등 C 확장 모듈 컴파일용
# curl: 헬스체크용
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python 의존성 설치 (캐시 최적화를 위해 먼저 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 노출 (FastAPI 기본 포트)
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]