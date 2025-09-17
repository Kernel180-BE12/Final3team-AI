# 테스트 가이드

## 개요
Agent2 4개 Tools 기반 템플릿 검증 시스템의 테스트 코드입니다.

## 테스트 구조

### 📁 tests/
- `conftest.py` - pytest 설정 및 공통 픽스처
- `test_template_validator.py` - 검증 시스템 단위 테스트
- `test_template_selector.py` - 템플릿 선택기 통합 테스트
- `test_integration.py` - 전체 시스템 통합 테스트
- `README.md` - 이 파일

## 테스트 실행

### 전체 테스트 실행
```bash
pytest
```

### 특정 파일 테스트
```bash
pytest tests/test_template_validator.py
```

### 마커별 테스트
```bash
# 통합 테스트만 실행
pytest -m integration

# 성능 테스트만 실행
pytest -m performance

# 엣지 케이스 테스트만 실행
pytest -m edge_cases
```

### 상세 출력
```bash
pytest -v -s
```

### 커버리지 확인
```bash
pytest --cov=src --cov-report=term-missing
```

## 테스트 카테고리

### 🔧 단위 테스트 (test_template_validator.py)
- `TestTemplateValidator` - 검증기 핵심 기능
- `TestValidationResult` - 검증 결과 클래스
- `TestTemplateValidationReport` - 검증 보고서 클래스

**테스트 케이스:**
- BlackList 검증 (통과/실패)
- WhiteList 검증 (승인/부분승인/미승인)
- Guideline 검증 (높음/중간/낮음 준수도)
- Law 검증 (준수/부분준수/미준수)
- 전체 검증 플로우
- 가중치 기반 점수 계산
- 권장사항 생성
- 금지 패턴 감지
- 싱글톤 패턴

### 🔄 통합 테스트 (test_template_selector.py)
- `TestTemplateSelector` - 템플릿 선택기 + 검증 통합

**테스트 케이스:**
- 검증 성공하는 새 템플릿 생성
- 검증 실패로 재생성 (최대 3회)
- 경고 수준이지만 허용되는 케이스
- Agent2 생성 자체 실패
- 변수 표준화
- 재생성 시도 횟수 설정

### 🌐 시스템 통합 테스트 (test_integration.py)
- `TestFullSystemIntegration` - 전체 시스템 플로우
- `TestPerformance` - 성능 테스트
- `TestEdgeCases` - 엣지 케이스

**테스트 시나리오:**
- API → TemplateSelector → Agent2 → Validator 전체 플로우
- 검증 실패 → 재생성 플로우
- 공용 템플릿 선택시 검증 생략
- 성능 테스트 (평균 < 10ms)
- 메모리 사용량 테스트 (< 10MB)
- 빈 입력 처리
- 잘못된 Tools 결과 처리
- 유니코드/특수문자 처리
- 매우 긴 템플릿 처리

## 픽스처 (Fixtures)

### conftest.py에서 제공하는 픽스처:
- `sample_tools_results` - 성공하는 Tools 결과
- `failed_tools_results` - 실패하는 Tools 결과
- `sample_good_template` - 좋은 템플릿 샘플
- `sample_bad_template` - 나쁜 템플릿 샘플

## Mock 전략

### 테스트에서 Mock되는 컴포넌트:
- `IndexManager` - 무거운 ChromaDB 초기화 회피
- `Agent2` - LLM 호출 회피, 예측 가능한 결과 제공
- `get_llm_manager` - API 키 및 LLM 초기화 회피
- `get_public_template_manager` - 공용 템플릿 DB 접근 회피

### Mock 사용 이유:
1. **속도 향상** - 외부 의존성 제거로 빠른 테스트
2. **안정성** - 네트워크나 외부 서비스 장애 독립적
3. **예측가능성** - 일관된 테스트 결과
4. **격리성** - 각 테스트가 다른 테스트에 영향 없음

## 테스트 데이터

### 성공 케이스:
```python
sample_good_template = '''
안녕하세요, ${고객명}님!
${병원명}에서 건강검진 예약이 ${예약일시}에 예정되어 있습니다.
문의사항이 있으시면 ${연락처}로 연락주세요.
감사합니다.
'''
```

### 실패 케이스:
```python
sample_bad_template = '''
🎉 100% 무료 투자 기회! 🎉
대출 즉시 승인! 투자 수익 보장!
지금 바로 클릭하세요!
'''
```

## 성능 기준

### 검증 시스템 성능 목표:
- **평균 검증 시간**: < 10ms (0.01초)
- **메모리 사용량**: < 10MB (피크)
- **긴 템플릿 처리**: < 1초 (10KB 템플릿)

### 성능 테스트 실행:
```bash
pytest -m performance -s
```

## CI/CD 통합

### GitHub Actions 예시:
```yaml
- name: Run tests
  run: |
    poetry run pytest --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## 문제 해결

### 공통 문제:

1. **Import 에러**
   ```bash
   ModuleNotFoundError: No module named 'src'
   ```
   해결: `PYTHONPATH` 설정 또는 `sys.path.append()` 확인

2. **Mock 관련 에러**
   ```bash
   AttributeError: 'Mock' object has no attribute 'xxx'
   ```
   해결: Mock 객체의 return_value 설정 확인

3. **느린 테스트**
   - 실제 LLM 호출하고 있는지 확인
   - Mock이 제대로 적용되었는지 확인

### 디버깅:
```bash
# 실패한 테스트만 재실행
pytest --lf

# 특정 테스트 디버깅
pytest tests/test_template_validator.py::TestTemplateValidator::test_blacklist_validation_passed -v -s
```

## 기여하기

새로운 테스트 추가시:

1. **단위 테스트**: 새로운 기능에 대한 단위 테스트
2. **통합 테스트**: 기존 시스템과의 통합 검증
3. **엣지 케이스**: 예외 상황 처리 확인
4. **성능 테스트**: 성능 영향도 확인

### 테스트 작성 가이드라인:
- 테스트 이름은 명확하고 구체적으로
- Given-When-Then 패턴 사용
- 하나의 테스트에서는 하나의 케이스만
- Mock 사용시 명확한 이유 명시
- 성능에 민감한 코드는 성능 테스트 추가