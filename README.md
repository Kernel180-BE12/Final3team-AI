# JOBER AI - 알림톡 템플릿 생성기

**AI 기반 카카오 알림톡 템플릿 자동 생성 시스템**

[![Python](https://img.shields.io/badge/Python-3.11.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![Chroma DB](https://img.shields.io/badge/ChromaDB-0.4.24-orange.svg)](https://www.trychroma.com/)

## 핵심 기능 

###  3단계 템플릿 선택 시스템
1. **기존 승인 템플릿**: predata 패턴 기반 검색 (임계값 0.7)
2. **공용 템플릿**: 105개 공용 템플릿 검색 (임계값 0.6)
3. **새 템플릿 생성**: AI 기반 신규 생성 + 자체 품질 검증

###  AI 기반 지능형 생성
- **Google Gemini 2.0 Flash + OpenAI GPT-4**: LLM Fallback 시스템
- **Agent1 & Agent2**: 2단계 AI 검증 시스템
- **자체 품질 검증**: 생성된 템플릿 자동 품질 검증 시스템

###  벡터 검색 엔진
- **Chroma DB**: 로컬 벡터 데이터베이스
- **공용 템플릿 매칭**: 키워드 기반 유사도 검색
- **실시간 캐싱**: 성능 최적화 캐시 시스템

###  완벽한 컴플라이언스
- **4개 Tool 병렬 검증**: BlackList, WhiteList, 가이드라인, 정보통신법
- **정보통신망법 준수**: PDF 추출 기반 법령 자동 검증
- **카카오 정책 준수**: 실시간 가이드라인 매칭

---

## 시스템 아키텍처 

### 3단계 템플릿 선택 워크플로우
```
사용자 입력
    ↓
1️⃣ 기존 승인 템플릿 검색 (predata 기반)
    ↓ (실패시)
2️⃣ 공용 템플릿 검색 (105개 템플릿)
    ↓ (실패시)  
3️⃣ AI 새 템플릿 생성 + 자체 검증
    ↓
 표준 형식 변환 + 결과 반환
```

### 주요 컴포넌트

** Template Selector**
- 3단계 선택 로직 관리
- 유사도 임계값 설정 (기존: 0.7, 공용: 0.6)
- 선택 경로 추적 및 메타데이터 제공

** Public Template Manager**
- 105개 공용 템플릿 관리
- 키워드 기반 유사도 검색
- `#{변수명}` → `${변수명}` 자동 변환

** LLM Provider Manager**
- Gemini → OpenAI 자동 Fallback
- 실패 횟수 추적 및 우선순위 관리
- 모델별 retry 로직

** Template Validator**
- 자체 품질 검증 시스템
- 다중 메트릭 종합 평가
- 자동 재생성 로직

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
cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

### 3. 서버 실행
```bash
# FastAPI 서버 실행
python server.py

# 또는 uvicorn 직접 실행
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

---

## API 엔드포인트 📡

### POST /ai/templates
템플릿 생성 요청

**요청 형식:**
```json
{
  "userId": 101,
  "requestContent": "국가 건강검진 안내"
}
```

**응답 형식:**
```json
{
  "success": true,
  "template": "안녕하세요, ${고객명}님!\n\n${검진내용} 국가건강검진 안내입니다...",
  "variables": [
    {
      "name": "${고객명}",
      "description": "고객명",
      "required": true
    }
  ],
  "metadata": {
    "source": "public",
    "selection_path": ["stage1_existing", "stage2_public"],
    "source_info": {
      "template_code": "guide_img_02",
      "similarity": 0.85
    },
    "quality_verified": false
  }
}
```

### GET /health
시스템 상태 확인

**응답 예시:**
```json
{
  "status": "healthy",
  "components": {
    "template_selector": "ready",
    "llm_manager": "ready",
    "template_validator": "ready"
  },
  "template_selection": {
    "existing_templates": {
      "available": "predata 기반 검색",
      "threshold": 0.7
    },
    "public_templates": {
      "available": 105,
      "threshold": 0.6
    }
  },
  "llm_providers": {
    "available_providers": ["gemini", "openai"]
  }
}
```

---

## 테스트 🧪

### 3단계 선택 시스템 테스트
```bash
# 공용 템플릿 매칭 테스트
python -c "
from src.core.template_selector import get_template_selector
selector = get_template_selector()
result = selector.select_template('국가 건강검진 안내 메시지 만들어줘')
print(f'선택 경로: {\" -> \".join(result.selection_path)}')
print(f'소스: {result.source}')
"

# LLM Fallback 테스트
python test_llm_fallback.py

# 자체 품질 검증 테스트
python test_template_validator.py
```

### API 통합 테스트
```bash
# API 테스트
curl -X POST http://localhost:8000/ai/templates \
  -H "Content-Type: application/json" \
  -d '{"userId": 123, "requestContent": "신년 할인 쿠폰 발급 알림톡 만들어줘"}'
```

---

## 프로젝트 구조 

```
Jober_ai/
├── README.md                           # 프로젝트 설명서
├── requirements.txt                    # Python 의존성
├── config.py                          # 환경변수 설정
├── server.py                          # FastAPI 서버
├── api.py                             # 메인 API 로직
├── src/                               # 소스코드
│   ├── core/                          # 핵심 엔진
│   │   ├── template_selector.py       # 3단계 선택 시스템 ⭐
│   │   ├── public_template_manager.py # 공용 템플릿 관리 ⭐
│   │   ├── index_manager.py           # Chroma DB 관리
│   │   ├── entity_extractor.py        # 엔티티 추출
│   │   └── template_generator.py      # 템플릿 생성
│   ├── agents/                        # AI 에이전트
│   │   ├── agent1.py                  # 입력 검증 에이전트
│   │   └── agent2.py                  # 4개 Tool 병렬 검증
│   ├── evaluation/                    # 품질 검증 ⭐
│   │   └── template_validator.py      # 자체 품질 검증기
│   ├── utils/                         # 유틸리티
│   │   ├── llm_provider_manager.py    # LLM Fallback 시스템 ⭐
│   │   ├── common_init.py             # 공통 초기화
│   │   └── sample_templates.py        # 샘플 템플릿
│   └── tools/                         # 검증 도구
│       ├── blacklist_tool.py          # 금지어 검사
│       ├── whitelist_tool.py          # 허용 패턴 검사
│       └── info_comm_law_tool.py      # 법령 검사
├── predata/                           # 전처리된 가이드라인 (9개 파일)
├── data/                              # 데이터
│   └── temp_fix.json                  # 105개 공용 템플릿
└── docs/                              # 문서
```

⭐ = 새로 구현된 핵심 기능

---

## 주요 기술 스택 

### AI & ML
- **Google Gemini 2.0 Flash**: 최신 AI 모델 (주 LLM)
- **OpenAI GPT-4**: Fallback LLM
- **자체 검증 시스템**: 생성 품질 평가
- **LangChain**: AI 에이전트 프레임워크

### 백엔드
- **Python 3.11.13**: 안정적인 파이썬 버전
- **FastAPI**: 고성능 웹 프레임워크
- **Chroma DB**: 벡터 데이터베이스 (로컬)
- **Pydantic**: 데이터 검증 및 시리얼라이제이션

### 데이터 처리
- **9개 가이드라인 파일**: 전처리된 정책 데이터
- **768차원 임베딩**: Gemini 임베딩 모델
- **벡터 검색**: 유사도 기반 컨텍스트 매칭
- **로컬 캐싱**: 성능 최적화 캐시 시스템

---

## 품질 보증 

### 자체 품질 검증 메트릭
1. **템플릿 구조**: 기본 구조 및 필수 요소 검증
2. **변수 유효성**: 변수 형식 및 일관성 확인
3. **정책 준수**: 가이드라인 및 법령 준수 여부
4. **내용 품질**: 의미론적 일관성 및 완성도
5. **사용성**: 실제 사용 가능성 평가

### 품질 게이트
- **종합 점수 기반**: 자동 승인/재생성 결정
- **다단계 검증**: 여러 관점에서 품질 평가
- **재생성 로직**: 품질 미달 시 자동 재생성 (최대 3회)

### Fallback 시스템
- **Gemini API 장애**: OpenAI로 자동 전환
- **실패 카운트 추적**: 연속 실패시 우선순위 조정
- **복구 감지**: 정상 작동시 원래 우선순위 복원
