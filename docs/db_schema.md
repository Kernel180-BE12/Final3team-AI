#  데이터베이스 스키마 제안

백엔드 DB 연동을 위한 테이블 구조와 JSON 형식을 제안합니다.

##  테이블 구조

### 1. `templates` 테이블 (메인 템플릿 데이터)

```sql
CREATE TABLE templates (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_input TEXT NOT NULL COMMENT '사용자 원본 입력',
    generated_template LONGTEXT NOT NULL COMMENT '생성된 템플릿',
    method VARCHAR(50) DEFAULT 'Agent2' COMMENT '생성 방법 (Agent2/Legacy)',
    quality_assured BOOLEAN DEFAULT FALSE COMMENT '품질 보증 여부',
    guidelines_compliant BOOLEAN DEFAULT FALSE COMMENT '가이드라인 준수 여부',
    legal_compliant BOOLEAN DEFAULT FALSE COMMENT '법령 준수 여부',
    status ENUM('draft', 'approved', 'rejected') DEFAULT 'draft' COMMENT '승인 상태',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_method (method),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);
```

### 2. `template_entities` 테이블 (추출된 엔티티)

```sql
CREATE TABLE template_entities (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    template_id BIGINT NOT NULL,
    entity_type VARCHAR(50) NOT NULL COMMENT 'dates/names/locations/events/others',
    entity_value VARCHAR(500) NOT NULL COMMENT '추출된 값',
    confidence_score DECIMAL(3,2) DEFAULT NULL COMMENT '신뢰도 점수',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE,
    INDEX idx_template_id (template_id),
    INDEX idx_entity_type (entity_type)
);
```

### 3. `template_variables` 테이블 (템플릿 변수)

```sql
CREATE TABLE template_variables (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    template_id BIGINT NOT NULL,
    variable_name VARCHAR(100) NOT NULL COMMENT '변수명 (예: #{수신자명})',
    variable_type VARCHAR(50) DEFAULT 'text' COMMENT 'text/date/number/email 등',
    is_required BOOLEAN DEFAULT TRUE COMMENT '필수 변수 여부',
    default_value VARCHAR(200) DEFAULT NULL COMMENT '기본값',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE,
    INDEX idx_template_id (template_id),
    INDEX idx_variable_name (variable_name)
);
```

### 4. `template_metadata` 테이블 (메타데이터)

```sql
CREATE TABLE template_metadata (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    template_id BIGINT NOT NULL,
    message_intent VARCHAR(100) DEFAULT NULL COMMENT '메시지 의도',
    message_type VARCHAR(50) DEFAULT NULL COMMENT '메시지 유형',
    urgency_level VARCHAR(20) DEFAULT 'normal' COMMENT '긴급도',
    estimated_length INT DEFAULT NULL COMMENT '예상 글자 수',
    agent2_data JSON DEFAULT NULL COMMENT 'Agent2 추가 데이터',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE,
    UNIQUE KEY uk_template_metadata (template_id)
);
```

##  JSON 구조

### API 응답 JSON 형식

```json
{
  "template_data": {
    "user_input": "내일 오후 2시에 카페에서 모임",
    "generated_template": "[모임 안내]\n\n#{수신자명}님, 안녕하세요...",
    "method": "Agent2",
    "quality_assured": true,
    "guidelines_compliant": true,
    "legal_compliant": true,
    "status": "draft"
  },
  "entities": [
    {
      "entity_type": "dates",
      "entity_value": "내일 오후 2시",
      "confidence_score": 0.95
    },
    {
      "entity_type": "locations", 
      "entity_value": "카페",
      "confidence_score": 0.87
    },
    {
      "entity_type": "events",
      "entity_value": "모임",
      "confidence_score": 0.92
    }
  ],
  "variables": [
    {
      "variable_name": "#{수신자명}",
      "variable_type": "text",
      "is_required": true,
      "default_value": null
    },
    {
      "variable_name": "#{모임장소}",
      "variable_type": "text", 
      "is_required": true,
      "default_value": "카페"
    }
  ],
  "metadata": {
    "message_intent": "모임안내",
    "message_type": "정보성",
    "urgency_level": "normal",
    "estimated_length": 245,
    "agent2_data": {
      "tools_used": ["blacklist", "whitelist", "guideline"],
      "validation_passed": true,
      "generation_time": 2.34
    }
  }
}
```

##  백엔드 연동 방법

### 1. 단순 JSON 전송

```python
# API에서 생성된 데이터를 JSON으로 백엔드에 전송
import requests

result = api.generate_template("사용자 입력")
json_data = api.export_to_json(result)

response = requests.post(
    "http://your-backend.com/api/templates",
    json=json_data,
    headers={"Content-Type": "application/json"}
)
```

### 2. 배치 처리

```python
# 여러 템플릿을 배치로 처리
templates = []
for user_input in user_inputs:
    result = api.generate_template(user_input)
    templates.append(api.export_to_json(result))

requests.post(
    "http://your-backend.com/api/templates/batch",
    json={"templates": templates}
)
```

##  백엔드 API 엔드포인트 제안

### 템플릿 저장
```
POST /api/templates
Content-Type: application/json

{
  "template_data": {...},
  "entities": [...],
  "variables": [...], 
  "metadata": {...}
}
```

### 템플릿 조회
```
GET /api/templates/{id}
GET /api/templates?status=draft&method=Agent2&page=1&limit=10
```

### 템플릿 승인/반려
```
PATCH /api/templates/{id}/status
{
  "status": "approved",
  "reviewer_comment": "검토 완료"
}
```

##  인덱스 최적화 제안

```sql
-- 자주 사용되는 검색 조건
CREATE INDEX idx_templates_method_status ON templates(method, status);
CREATE INDEX idx_templates_created_status ON templates(created_at, status);
CREATE INDEX idx_entities_template_type ON template_entities(template_id, entity_type);

-- 전문 검색이 필요한 경우
ALTER TABLE templates ADD FULLTEXT(user_input, generated_template);
```

이 구조로 하면 **확장 가능하고 효율적인 DB 설계**가 됩니다! 