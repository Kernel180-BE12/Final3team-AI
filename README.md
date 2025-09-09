# JOBER AI - 알림톡 템플릿 생성기

**AI 기반 카카오 알림톡 템플릿 자동 생성 시스템**

[![Python](https://img.shields.io/badge/Python-3.11.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![Chroma DB](https://img.shields.io/badge/ChromaDB-0.4.24-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 핵심 기능

### AI 기반 지능형 생성
- **Google Gemini 2.0 Flash**: 최신 AI 모델 기반 고품질 템플릿 생성
- **Agent1 & Agent2**: 2단계 AI 검증 시스템으로 완벽한 품질 보장
- **Chroma DB 벡터 검색**: 768차원 임베딩을 통한 정확한 유사도 검색
- **자연어 처리**: "내일", "다음 주" 등 자연어 날짜를 실제 날짜로 자동 변환

### 카테고리 시스템 (신규)
- **세분화된 분류**: 001-009 대분류, 001001-009005 소분류 체계
- **자동 카테고리 매칭**: 사용자 입력에 따른 지능형 카테고리 자동 선택
- **업종별 최적화**: 회원, 구매, 예약, 서비스이용, 배송 등 9개 대분류 지원

### 완벽한 컴플라이언스
- **4개 Tool 병렬 검증**: BlackList, WhiteList, 가이드라인, 정보통신법 동시 검사
- **정보통신망법 준수**: PDF 추출 기반 법령 자동 검증 시스템
- **카카오 정책 준수**: 실시간 가이드라인 매칭 및 위반 사항 자동 감지

---

## 시스템 아키텍처

### 핵심 워크플로우
```
사용자 입력 → Agent1 검증 → 템플릿 생성 → Agent2 병렬 검증 → RAGAS를 통한 검증 → 결과 반환
```

### 주요 컴포넌트

**1. Agent1 - 입력 분석 & 검증**
   - 사용자 의도 분류 (새로운 카테고리 체계 적용)
   - 6W 변수 추출 (누가, 언제, 어디서, 무엇을, 왜, 어떻게)
   - 날짜 표현 자동 변환 ("내일" → "2025-01-16")

**2. Agent2 - 4개 Tool 병렬 검증**
   - **BlackList Tool**: 금지 키워드 및 패턴 검사
   - **WhiteList Tool**: 승인 패턴 매칭
   - **Guideline Tool**: 통합 가이드라인 준수 검증
   - **InfoCommLaw Tool**: 정보통신망법 법령 검증

**3. 벡터 검색 엔진 (Chroma DB)**
   - 9개 전처리된 가이드라인 파일 임베딩
   - 768차원 Gemini 임베딩 모델 사용
   - 컬렉션별 분리 저장 (guidelines, templates)

**4. FastAPI 백엔드 서버**
   - RESTful API 엔드포인트
   - JSON 응답 형식 (Spring Boot 호환)
   - 자동 카테고리 분류 및 매핑

---

## 빠른 시작

### 1. 설치 및 환경 설정
```bash
# 레포지토리 클론
git clone https://github.com/your-username/Jober_ai.git
cd Jober_ai

# Python 3.11.13 설치 (권장)
pyenv install 3.11.13
pyenv local 3.11.13

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정
```bash
# .env 파일 생성
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

### 3. 서버 실행
```bash
# FastAPI 서버 실행
python server.py
# 또는
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 4. API 테스트
```bash
# 템플릿 생성 요청
curl -X POST http://localhost:8000/ai/templates \
  -H "Content-Type: application/json" \
  -d '{"userId": 123, "requestContent": "예약이 완료되었습니다. 고객에게 알려주세요"}'

# 서버 상태 확인
curl http://localhost:8000/health
```

### 5. 응답 예시
```json
{
  "id": 1,
  "userId": 123,
  "categoryId": "003001",
  "title": "예약 완료 안내",
  "content": "안녕하세요, #{고객명}님.\n\n예약이 완료되었습니다...",
  "type": "MESSAGE",
  "variables": [
    {
      "id": 1,
      "variableKey": "고객명",
      "placeholder": "#{고객명}",
      "inputType": "TEXT"
    }
  ]
}
```

---

## API 엔드포인트

### POST /ai/templates
템플릿 생성 요청 (FastAPI)

**요청 형식:**
```json
{
  "userId": 101,
  "requestContent": "쿠폰 발급 안내"
}
```

**응답 형식:**
```json
{
  "id": 1,
  "userId": 101,
  "categoryId": 9101,
  "title": "쿠폰 발급 안내",
  "content": "안녕하세요, #{고객명}님.\n#{쿠폰명} 쿠폰 발급을 안내드립니다...",
  "imageUrl": null,
  "type": "MESSAGE",
  "buttons": [],
  "variables": [
    {
      "id": 1,
      "variableKey": "고객명",
      "placeholder": "고객 이름",
      "inputType": "text"
    }
  ],
  "industries": [],
  "purposes": []
}
```

### GET /health
서버 상태 확인 (헬스 체크)

---

## 카테고리 분류 시스템

### 새로운 카테고리 체계
**대분류 (001-009)**
- 001: 회원, 002: 구매, 003: 예약, 004: 서비스이용, 005: 리포팅
- 006: 배송, 007: 법적고지, 008: 업무알림, 009: 쿠폰/포인트

**소분류 (세부 분류)**
- 예: 001001(회원가입), 002001(구매완료), 003001(예약완료) 등

### 자동 키워드 매칭
| 카테고리 | 키워드 | 분류 코드 |
|---------|--------|----------|
| 회원가입 | 가입, 신규, 등록 | 001001 |
| 구매완료 | 구매완료, 주문완료, 결제완료 | 002001 |
| 예약완료 | 예약완료, 예약확인, 방문예약 | 003001 |
| 배송상태 | 배송상태, 배송조회, 발송 | 006001 |
| 쿠폰발급 | 쿠폰발급, 쿠폰지급, 혜택 | 009001 |

---

## 테스트

### API 기본 테스트
```bash
# 서버 실행
python server.py

# API 테스트
curl -X POST http://localhost:8000/ai/templates \
  -H "Content-Type: application/json" \
  -d '{"userId": 123, "requestContent": "예약 완료 안내"}'
```

### 벡터 검색 테스트
```bash
# Chroma DB 상태 확인
python -c "from src.core.index_manager import IndexManager; manager = IndexManager(); print(f'컬렉션: {len(manager.chroma_client.list_collections())}개')"
```

---

## 프로젝트 구조

```
Jober_ai/
├── README.md                    # 프로젝트 설명서
├── requirements.txt             # Python 의존성
├── config.py                   # 환경변수 설정
├── server.py                   # FastAPI 서버
├── api.py                      # 메인 API 로직
├── src/                        # 소스코드
│   ├── core/                   # 핵심 엔진
│   │   ├── base_processor.py   # 기본 프로세서
│   │   ├── entity_extractor.py # 엔티티 추출
│   │   ├── template_generator.py # 템플릿 생성
│   │   └── index_manager.py    # Chroma DB 관리
│   ├── agents/                 # AI 에이전트
│   │   ├── agent1.py          # 입력 검증 에이전트
│   │   └── agent2.py          # 4개 Tool 병렬 검증
│   ├── tools/                  # 검증 도구
│   │   ├── blacklist_tool.py   # 금지어 검사
│   │   ├── whitelist_tool.py   # 허용 패턴 검사
│   │   ├── info_comm_law_tool.py # 법령 검사
│   │   └── intent_classifier.py # 카테고리 분류
│   └── utils/                  # 유틸리티
├── predata/                    # 전처리된 데이터 (9개 파일)
├── chroma_db/                  # 벡터 데이터베이스
├── cache/                      # 캐시 파일
└── docs/                       # 문서
```

---

## 주요 특징

### 기술 스택
- **Python 3.11.13**: 안정적인 파이썬 버전
- **Google Gemini 2.0 Flash**: 최신 AI 모델
- **Chroma DB**: 벡터 데이터베이스 (로컬 저장)
- **FastAPI**: 고성능 웹 프레임워크
- **LangChain**: AI 에이전트 프레임워크

### 데이터 처리
- **9개 가이드라인 파일**: 전처리된 정책 데이터
- **768차원 임베딩**: Gemini 임베딩 모델 사용
- **벡터 검색**: 유사도 기반 컨텍스트 매칭
- **로컬 캐싱**: 성능 최적화를 위한 캐시 시스템

---