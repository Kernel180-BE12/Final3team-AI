#  JOBER AI - 알림톡 템플릿 생성기

**기업용 AI 기반 알림톡 템플릿 자동 생성 시스템**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()
[![Backend](https://img.shields.io/badge/Backend-Ready-orange.svg)]()

##  핵심 기능

###  스마트 템플릿 생성
- ** AI 기반 생성**: Google Gemini 2.5 Flash를 활용한 고품질 템플릿 생성
- ** 완벽 정책 준수**: Agent2를 통한 가이드라인/법령/블랙리스트/화이트리스트 검증
- ** 벡터 검색**: FAISS 기반 유사도 검색으로 최적 템플릿 생성
- ** 스마트 날짜 처리**: 자연어 시간 표현을 실제 날짜로 자동 변환

### 백엔드 연동 최적화
- **RESTful API**: FastAPI 기반 완전한 백엔드 연동 인터페이스
- **JSON 출력**: 기업 DB 스키마 호환 JSON 형식 지원
- **업종별 분류**: 5개 업종(소매업, 부동산, 교육, 서비스업, 기타) 자동 분류
- **변수 추출**: 템플릿 내 #{변수} 자동 추출 및 매핑

---

## 🛠 시스템 아키텍처

###  핵심 컴포넌트

1. **Agent2 생성 엔진** (`src/agents/agent2.py`)
   - 4개 도구 병렬 분석 (가이드라인, 법령, 블랙리스트, 화이트리스트)
   - 완벽한 정책 준수 템플릿 생성
   - FAISS 벡터 검색 기반 컨텍스트 매칭

2. **템플릿 생성기** (`src/core/template_generator.py`)
   - prototype.py 로직 통합 완료
   - 변수 추출, 의도 파악, 정책 검증 파이프라인
   - 벡터 검색 기반 템플릿 생성

3. **Agent1 전용 도구들**
   - **날짜 전처리기** (`src/tools/date_preprocessor.py`): 자연어 날짜 표현 변환
   - **의도 분류기** (`src/tools/intent_classifier.py`): 사용자 의도 자동 분류
   - **변수 추출기** (`src/tools/variable_extractor.py`): 6W 변수 추출

4. **백엔드 연동 서버** (`server.py`)
   - FastAPI 기반 RESTful API
   - JSON 출력 및 DB 스키마 매칭
   - 실시간 템플릿 생성 서비스

---

##  빠른 시작

### 1. 환경 설정
```bash
# 레포지토리 클론
git clone https://github.com/your-username/Jober_ai.git
cd Jober_ai

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정 (.env 파일 생성)
echo "GOOGLE_API_KEY=your_gemini_api_key" > .env
```

### 2. 로컬 실행
```python
# 기본 CLI 모드
python main.py

# 백엔드 서버 실행 (FastAPI)
uvicorn server:app --reload --host 0.0.0.0 --port 5000
```

### 3. API 사용법
```bash
# 템플릿 생성 요청
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "쿠폰 발급 안내", "use_agent2": true}'
```

### 4. 응답 예시
```json
{
  "generated_template": "안녕하세요, #{고객명}님.\n...",
  "variables": ["#{고객명}", "#{쿠폰명}", "#{만료일}"],
  "entities": {...},
  "generation_method": "Agent2_Compliant_Generation",
  "quality_assured": true
}
```

---

## API 엔드포인트

### POST /api/v1/templates
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

##  기업 베스트 템플릿 호환성

### 지원하는 템플릿 유형 (20종)
- **소매업**: 쿠폰 발급, 가격 변경, 샘플 입고, 구매 안내
- **부동산**: 방문 예약, 상담 예약, 행사장 위치 안내  
- **교육**: 강의 일정, 학습 주요 사항 안내
- **서비스업**: PT 결과, 서비스 업데이트, 이용권 안내
- **기타**: 회원 혜택, 적립금, 등급, 이벤트, 인보이스

### 자동 분류 키워드
| 업종 | 자동 감지 키워드 |
|------|------------------|
| 소매업 | 쿠폰, 할인, 상품, 구매, 샘플, 브랜드 |
| 부동산 | 예약, 방문, 상담, 매물, 아파트 |
| 교육 | 강의, 학습, 수업, 교육, 학원, 과정 |
| 서비스업 | 뷰티, 건강, PT, 시술, 관리, 서비스 |
| 기타 | 회원, 적립금, 등급, 이벤트, 당첨 |

---

##  테스트

### 백엔드 연동 테스트
```bash
python test_backend_integration.py
```

### CLI 모드 테스트
```bash
python main.py
```

### 서버 실행 테스트
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 5000
# 또는
curl -X POST http://localhost:5000/api/v1/templates \
  -H "Content-Type: application/json" \
  -d '{"userId": 101, "requestContent": "쿠폰 발급 안내"}'
```

---

##  프로젝트 구조

```
Jober_ai/
├──  README.md                    # 프로젝트 설명서
├──  requirements.txt             # Python 의존성
├──  pyproject.toml              # 프로젝트 설정
├──  config.py                   # 설정 파일
├──  main.py                     # CLI 실행 파일
├── server.py                   # FastAPI 백엔드 서버
├──  api.py                      # API 유틸리티
├──  test_backend_integration.py # 백엔드 연동 테스트
├──  src/                        # 소스코드 패키지
│   ├──  __init__.py
│   ├──  core/                   # 핵심 컴포넌트  
│   │   ├──  __init__.py
│   │   ├──  base_processor.py  # 기본 프로세서
│   │   ├──  entity_extractor.py # 엔티티 추출
│   │   ├──  template_generator.py # 템플릿 생성 (prototype.py 통합)
│   │   └──  index_manager.py   # 벡터 인덱스 관리
│   ├──  agents/                 # AI 에이전트
│   │   └──  agent2.py          # Agent2 생성 엔진
│   ├──  tools/                  # 분석 도구
│   │   ├──  __init__.py
│   │   ├──  blacklist_tool.py  # 금지 키워드 검사
│   │   ├──  whitelist_tool.py  # 허용 패턴 검사
│   │   ├──  info_comm_law_tool.py # 정보통신법 검사
│   │   ├──  date_preprocessor.py # 날짜 전처리기 (Agent1용)
│   │   ├──  intent_classifier.py # 의도 분류기 (Agent1용)
│   │   └──  variable_extractor.py # 변수 추출기 (Agent1용)
│   └──  utils/                  # 유틸리티
│       ├──  __init__.py
│       └──  data_processor.py  # 데이터 처리
├──  data/                       # 원본 가이드라인 데이터
├──  predata/                    # 전처리된 가이드라인 데이터
├──  cache/                      # FAISS 인덱스 캐시
└──  docs/                       # 문서
    ├──  BACKEND_INTEGRATION_FINAL.md
    ├──  ENTERPRISE_SCHEMA_INTEGRATION.md
    └──  db_schema.md
```

---

##  기여하기

1. **Fork** 레포지토리
2. **브랜치** 생성 (`git checkout -b feature/amazing-feature`)
3. **커밋** 변경사항 (`git commit -m 'Add amazing feature'`)
4. **푸시** 브랜치 (`git push origin feature/amazing-feature`)
5. **Pull Request** 생성

---

##  버전 히스토리

### v2.1.0 (2025-09-01)
- prototype.py 로직 core/template_generator.py 통합
- FastAPI 기반 백엔드 서버 구현 (server.py)
- 프로젝트 구조 정리 및 src/ 패키지화
- 모든 이모지 제거 및 코드 정리 완료
- 백엔드 연동 최적화 완료

### v2.0.0 (2025-08-01)
-  Agent2 4도구 병렬 분석 시스템
-  업종별 자동 분류 시스템
-  FAISS 벡터 검색 및 캐시 시스템
-  완전한 정책 준수 파이프라인

### v1.0.0 (2025-07-01) 
-  기본 AI 템플릿 생성 기능
-  Google Gemini 2.5 Flash 통합
-  가이드라인 준수 시스템

---

##  지원 및 문의

- ** 이메일**: support@jober.ai
- ** 이슈 트래커**: [GitHub Issues](https://github.com/your-username/Jober_ai/issues)
- **📖 문서**: [상세 문서](./docs/)

---

##  라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

##  감사의 말

- **Google Gemini 2.5**: AI 템플릿 생성 엔진
- **FAISS**: 벡터 유사도 검색
- **LangChain**: AI 에이전트 프레임워크

---

<div align="center">

** 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! **

Made with  by JOBER Team

</div>