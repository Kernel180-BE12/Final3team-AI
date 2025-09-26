# JOBER AI - 알림톡 템플릿 생성기 🚀

**AI 기반 카카오 알림톡 템플릿 자동 생성 시스템**

[![Python](https://img.shields.io/badge/Python-3.11.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![Chroma DB](https://img.shields.io/badge/ChromaDB-0.4.24-orange.svg)](https://www.trychroma.com/)
[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0-brightgreen.svg)](http://localhost:8000/docs)

## 🌟 핵심 기능

### 🎯 3단계 템플릿 선택 시스템
1. **기존 승인 템플릿**: predata 패턴 기반 검색 (임계값 0.85)
2. **공용 템플릿**: 168개 공용 템플릿 검색 (임계값 0.8)
3. **새 템플릿 생성**: AI 기반 신규 생성 + 4-Tool 병렬 검증

### 🤖 AI 기반 지능형 생성
- **Google Gemini 2.0 Flash + OpenAI GPT-4**: LLM Fallback 시스템
- **Agent1 & Agent2**: 2단계 AI 검증 시스템
- **conversationContext**: 재질문 기능 지원
- **4개 Tool 병렬 검증**: BlackList, WhiteList, 가이드라인, 정보통신망법

### 💬 실시간 채팅 지원
- **세션 기반 변수 업데이트**: 실시간 템플릿 수정
- **템플릿 미리보기**: 단계별 완성도 확인
- **WebSocket 준비**: 백엔드 연동 지원

### 📚 완벽한 API 문서화
- **28개 응답 스키마**: 완전한 타입 안정성
- **태그별 분류**: Template Generation, Real-time Chat, Session Management, System
- **Swagger UI**: 자동 생성된 API 문서

---

## 시스템 아키텍처

### 템플릿 생성 워크플로우
```
사용자 입력 (conversationContext 지원)
    ↓
🔍 Agent1: 변수 추출 + 의도 분석 + 비속어 검증
    ↓
📋 3단계 템플릿 선택:
   1️⃣ BasicTemplateMatcher (기존 승인 템플릿)
   2️⃣ PublicTemplateManager (168개 공용 템플릿)
   3️⃣ Agent2 (새 템플릿 생성 + 4-Tool 검증)
    ↓
🔧 VariableMapper: Agent1 변수 → 템플릿 변수 매핑
    ↓
✅ TemplateValidator: 품질 검증 + 재생성 로직
    ↓
📤 최종 템플릿 + 세션 정보 반환
```

### 주요 컴포넌트

**🔧 Template Selector**
- 3단계 선택 로직 관리
- 유사도 임계값 설정 및 최적화
- 선택 경로 추적 및 메타데이터 제공

**📦 Public Template Manager**
- 168개 공용 템플릿 관리 (Chroma DB)
- 키워드 + 벡터 하이브리드 검색
- 자동 변수 매핑 시스템

**🤖 LLM Provider Manager**
- Gemini → OpenAI 자동 Fallback
- 실패 횟수 추적 및 우선순위 관리
- 모델별 retry 로직

**✅ Session Manager**
- Thread-safe 세션 관리
- 실시간 변수 업데이트
- 자동 만료 및 정리

---

## 🚀 빠른 시작

### 1. 설치 및 환경 설정
```bash
# 레포지토리 클론
git clone https://github.com/your-username/Jober_ai.git
cd Jober_ai

# Python 3.11.13 설치 (권장)
pyenv install 3.11.13
pyenv local 3.11.13

# Poetry로 의존성 설치
poetry install
```

### 2. 환경변수 설정
```bash
# .env 파일 생성 (필수)
cp .env.example .env

# .env 파일 편집
cat > .env << EOF
# LLM API 키 (둘 중 하나는 필수)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# 보안 설정 (필수)
SECRET_KEY=your-super-secret-key-32-characters-long

# 선택적 설정
DEBUG=false
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
EOF
```

### 3. 서버 실행
```bash
# FastAPI 서버 실행
python app/main.py

# 또는 Poetry로 실행
poetry run python app/main.py

# 또는 uvicorn 직접 실행
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. API 테스트
```bash
# 템플릿 생성 요청
curl -X POST http://localhost:8000/ai/templates \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 123,
    "requestContent": "독서모임 안내 메시지를 만들어주세요",
    "conversationContext": null
  }'

# 서버 상태 확인
curl http://localhost:8000/ai/health

# Swagger UI 접속
open http://localhost:8000/docs
```

---

## 📡 API 엔드포인트

### 🎯 Template Generation

#### `POST /ai/templates`
AI 기반 템플릿 생성 (재질문 지원)

**요청:**
```json
{
  "userId": 12345,
  "requestContent": "독서모임 안내 메시지 만들어주세요",
  "conversationContext": null
}
```

**성공 응답 (200):**
```json
{
  "id": 1,
  "userId": 12345,
  "categoryId": "004001",
  "title": "알림톡",
  "content": "안녕하세요 #{고객명}님!\n\n#{업체명}에서 #{서비스명} 관련하여 안내드립니다...",
  "imageUrl": null,
  "type": "MESSAGE",
  "buttons": [],
  "variables": [
    {
      "id": 1,
      "variableKey": "고객명",
      "placeholder": "#{고객명}",
      "inputType": "text",
      "value": "홍길동"
    }
  ],
  "industry": ["교육", "문화"],
  "purpose": ["안내", "예약확인"]
}
```

**정보 부족 응답 (400):**
```json
{
  "detail": {
    "error": {
      "code": "INCOMPLETE_INFORMATION",
      "message": "추가 정보가 필요합니다",
      "details": {
        "confirmed_variables": {"업체명": "ABC서점"},
        "missing_variables": ["일시", "장소", "연락처"],
        "contextual_question": "독서모임 일시와 장소, 문의 연락처를 알려주세요.",
        "original_input": "독서모임 안내 메시지 만들어줘"
      }
    },
    "timestamp": "2024-09-22T10:30:00Z"
  }
}
```

### 💬 Real-time Chat

#### `POST /ai/templates/{session_id}/variables`
세션 변수 업데이트

#### `GET /ai/templates/{session_id}/preview`
템플릿 미리보기 조회

#### `POST /ai/templates/{session_id}/complete`
템플릿 최종 완성

### 🔧 Session Management

#### `GET /ai/sessions/stats`
세션 통계 조회

#### `GET /ai/sessions`
세션 목록 조회

### 🛠️ System

#### `GET /ai/health`
기본 헬스체크

#### `GET /ai/health/detailed`
상세 헬스체크

#### `GET /ai/health/llm`
LLM 제공자 테스트

---

## 🧪 테스트

### API 통합 테스트
```bash
# 기본 템플릿 생성
curl -X POST http://localhost:8000/ai/templates \
  -H "Content-Type: application/json" \
  -d '{"userId": 123, "requestContent": "카페 이벤트 안내 메시지 만들어주세요"}'

# 재질문 테스트
curl -X POST http://localhost:8000/ai/templates \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 123,
    "requestContent": "일시는 3월 15일, 장소는 강남구입니다",
    "conversationContext": "카페 이벤트 안내 메시지 만들어주세요"
  }'

# 비속어 검출 테스트
curl -X POST http://localhost:8000/ai/templates \
  -H "Content-Type: application/json" \
  -d '{"userId": 123, "requestContent": "바보야"}'
```

### 개발 도구
```bash
# 코드 테스트 실행
poetry run pytest tests/

# 코드 품질 검사
poetry run ruff check app/
poetry run black app/

# 타입 검사
poetry run mypy app/
```

---

## 📁 프로젝트 구조

```
Jober_ai/
├── README.md                           # 프로젝트 설명서 ⭐
├── pyproject.toml                      # Poetry 의존성 설정
├── poetry.lock                         # Poetry 락 파일
├── .env.example                        # 환경변수 예시 ⭐
├── 백엔드.txt                          # 백엔드 연동 가이드 ⭐
├── app/                               # 메인 애플리케이션 ⭐
│   ├── main.py                        # FastAPI 진입점
│   ├── api/                           # API 엔드포인트
│   │   ├── templates.py               # 템플릿 생성 API ⭐
│   │   ├── sessions.py                # 세션 관리 API ⭐
│   │   └── health.py                  # 헬스체크 API
│   ├── agents/                        # AI 에이전트
│   │   ├── agent1.py                  # 변수 추출 + 의도 분석
│   │   └── agent2.py                  # 4-Tool 병렬 검증
│   ├── core/                          # 핵심 엔진
│   │   ├── template_selector.py       # 3단계 선택 시스템 ⭐
│   │   ├── public_template_manager.py # 168개 공용 템플릿 관리
│   │   ├── session_manager.py         # 세션 관리 ⭐
│   │   ├── template_preview.py        # 미리보기 생성 ⭐
│   │   └── session_models.py          # 세션 모델 ⭐
│   ├── tools/                         # 검증 도구
│   │   ├── blacklist_tool.py          # 금지어 검사
│   │   ├── whitelist_tool.py          # 허용 패턴 검사
│   │   ├── info_comm_law_tool.py      # 정보통신망법 검사
│   │   └── variable_mapper.py         # 변수 매핑 ⭐
│   └── utils/                         # 유틸리티
│       └── llm_provider_manager.py    # LLM Fallback 시스템
├── config/                            # 설정 파일
│   ├── settings.py                    # 애플리케이션 설정 ⭐
│   └── llm_providers.py               # LLM 제공자 설정
├── data/                              # 데이터
│   ├── templates/                     # 168개 공용 템플릿
│   ├── presets/                       # 전처리된 데이터
│   └── docs/                          # 정책 문서
├── tests/                             # 테스트
│   ├── test_template_selector.py      # 템플릿 선택 테스트
│   ├── test_template_validator.py     # 검증 시스템 테스트
│   └── test_integration.py            # 통합 테스트
└── scripts/                           # 유틸리티 스크립트
    └── debug_agent1_variables.py      # Agent1 디버깅
```

⭐ = 2024-09-22 업데이트

---

## 🛠️ 주요 기술 스택

### AI & ML
- **Google Gemini 2.0 Flash**: 최신 생성형 AI (주 LLM)
- **OpenAI GPT-4**: Fallback LLM
- **KoNLPy**: 한국어 자연어 처리
- **Chroma DB**: 벡터 데이터베이스 (임베딩 캐시)

### 백엔드
- **Python 3.11.13**: 안정적인 파이썬 버전
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **Pydantic**: 데이터 검증 (28개 응답 스키마)
- **Poetry**: 모던 의존성 관리
- **uvicorn**: ASGI 서버

### 데이터 & 캐싱
- **168개 공용 템플릿**: 전처리된 템플릿 DB
- **Chroma DB**: 벡터 검색 및 캐싱
- **768차원 임베딩**: Gemini 임베딩 모델
- **Thread-safe 세션**: 메모리 기반 세션 관리

---

## 🔒 보안 & 품질

### 보안 강화
- **SECRET_KEY 필수**: 환경변수 필수 설정
- **API 키 검증**: LLM 제공자 키 유효성 검사
- **CORS 정책**: 프로덕션 환경 도메인 제한 권장
- **에러 정보 제한**: DEBUG=false 시 상세 정보 숨김

### 품질 보증
- **4-Tool 병렬 검증**: BlackList, WhiteList, 가이드라인, 정보통신망법
- **자체 품질 검증**: TemplateValidator 시스템
- **재생성 로직**: 품질 미달 시 자동 재시도
- **완성도 추적**: 세션별 진행률 관리

### Fallback 시스템
- **LLM Fallback**: Gemini → OpenAI 자동 전환
- **실패 카운트**: 연속 실패 시 우선순위 조정
- **자동 복구**: 정상 작동 시 원래 설정 복원

---

## 🚀 배포 가이드

### 환경변수 필수 설정
```bash
# 필수 환경변수
GEMINI_API_KEY=your_actual_gemini_key
OPENAI_API_KEY=your_actual_openai_key
SECRET_KEY=your_super_secret_32_char_key

# 프로덕션 설정
DEBUG=false
CORS_ORIGINS=["https://yourdomain.com"]
LOG_LEVEL=WARNING
```

### Docker 배포 (권장)
```bash
# Dockerfile 생성 후
docker build -t jober-ai .
docker run -d -p 8000:8000 --env-file .env jober-ai
```

### 성능 최적화
- **메모리**: 최소 2GB RAM 권장
- **동시 연결**: MAX_CONCURRENT_REQUESTS=10 (기본)
- **세션 타임아웃**: 30분 (설정 가능)
- **캐시**: Chroma DB 자동 캐싱

---

## 📞 지원 & 기여

### 문제 해결
- **로그 확인**: `tail -f logs/app.log`
- **헬스체크**: `curl http://localhost:8000/ai/health/detailed`
- **API 문서**: http://localhost:8000/docs

### 기여하기
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### 라이선스
MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

---

## 코드베이스 분석 및 개선사항

### 현재 아키텍처 상태

이 프로젝트는 FastAPI 기반의 한국어 비즈니스 메시지 템플릿 생성 서비스로, AI 에이전트와 벡터 데이터베이스를 활용한 계층화된 구조를 가지고 있습니다.

**주요 구성요소:**
- **Agent1**: 입력 검증, 변수 추출, 의도 분류, 대화 상태 관리
- **Agent2**: 4개 도구를 활용한 템플릿 생성 및 컴플라이언스 검증
- **Core Layer**: 비즈니스 로직 및 템플릿 처리
- **API Layer**: FastAPI 라우트 핸들러

### 식별된 구조적 문제점

#### 중요 (Critical)
1. **Import 시스템 문제**: 수동 경로 조작 및 importlib 사용으로 인한 불안정성
2. **대형 클래스**: Agent1 (1,279줄), Agent2 (1,008줄) - 단일 책임 원칙 위반
3. **설정 시스템 분산**: 3개의 경쟁하는 설정 시스템

#### 보통 (Important)
4. **순환 의존성 위험**: 복잡한 교차 모듈 의존성
5. **오류 처리 불일치**: 중앙화된 오류 처리 전략 부재
6. **계층간 강한 결합**: API가 비즈니스 로직을 직접 인스턴스화

### 리팩토링 권장사항

#### 단계별 개선 계획
1. **Phase 1**: Import 시스템 안정화 (필수)
2. **Phase 2**: 서비스 레이어 추출 (필수)
3. **Phase 3**: 대형 클래스 분해 (중요)
4. **Phase 4**: 설정 시스템 통합 (중요)
5. **Phase 5**: 적절한 오류 처리 구현 (보통)

### LangGraph 마이그레이션 평가

#### LangGraph 적합성 - 권장

**현재 시스템의 복잡한 대화형 워크플로우는 LangGraph의 강점과 일치합니다:**

- **복잡한 상태 관리**: 다중 턴 대화에서 변수 수집
- **조건부 라우팅**: Agent1의 여러 검증 단계
- **도구 오케스트레이션**: Agent2의 4개 도구 병렬 실행
- **오류 처리**: 여러 실패 모드에 대한 복구 경로

#### 구현 전략
```python
# LangGraph 구현 예시
from langgraph.graph import StateGraph

class JoberState(TypedDict):
    user_input: str
    variables: dict
    validation_status: str
    template_result: dict

workflow = StateGraph(JoberState)
workflow.add_node("validate_input", agent1_validate)
workflow.add_node("generate_template", agent2_generate)
workflow.add_conditional_edges(
    "validate_input",
    check_completeness,
    {"generate_template": "generate_template", "reask": "reask_variables"}
)
```

#### 예상 효과
- 대화 상태 버그 30% 감소
- 새로운 에이전트 기능 추가 50% 용이
- 향상된 오류 복구 메커니즘
- 프로덕션 디버깅을 위한 관찰 가능성 개선

#### 마이그레이션 계획
1. **Phase 1** (3주): 핵심 Agent1→Agent2 플로우의 LangGraph 버전 구현
2. **Phase 2** (4-6주): 점진적 시스템 교체 및 A/B 테스트
3. **Phase 3** (2-3주): 고급 기능 (재시도 로직, 모니터링) 구현

---

**JOBER AI - AI로 완성하는 완벽한 알림톡 템플릿!**