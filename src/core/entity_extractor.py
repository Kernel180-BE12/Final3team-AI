import json
from typing import Dict, List
from .base_processor import BaseTemplateProcessor

class EntityExtractor(BaseTemplateProcessor):
    """엔티티 추출 전용 클래스"""
    
    def __init__(self, api_key: str, gemini_model: str = "gemini-2.0-flash-exp"):
        super().__init__(api_key, gemini_model)
    
    def extract_entities(self, user_input: str) -> Dict:
        """사용자 입력에서 엔티티 추출"""
        prompt = self._create_entity_extraction_prompt(user_input)
        
        try:
            response = self.generate_with_gemini(prompt)
            return self.parse_json_response(response)
        except Exception as e:
            print(f"엔티티 추출 오류: {e}")
            return self._create_fallback_entities(user_input)
    
    def _create_entity_extraction_prompt(self, user_input: str) -> str:
        """엔티티 추출 프롬프트 생성"""
        return f"""
다음 사용자 입력에서 구체적인 정보들을 추출해서 JSON 형태로 반환해주세요:

사용자 입력: "{user_input}"

다음 정보들을 찾아서 추출해주세요:
- 날짜/시간 정보 (예: 2025.8.26, 오후 2시 등)
- 사람 이름 (예: 홍길동 등)
- 장소/위치 (예: 강남점 등)
- 이벤트/행사명 (예: 세미나 등)
- 기타 중요 정보 (가격, 상품명, 서비스명 등)

JSON 형태:
{{
    "extracted_info": {{
        "dates": ["추출된 날짜들"],
        "names": ["추출된 이름들"], 
        "locations": ["추출된 장소들"],
        "events": ["추출된 이벤트들"],
        "others": ["기타 중요 정보들"]
    }},
    "message_intent": "메시지의 주요 목적 (예: 행사안내, 예약확인, 결제알림 등)",
    "context": "전체적인 상황/맥락 설명",
    "message_type": "메시지 유형 (정보성/광고성 판단)",
    "urgency_level": "긴급도 (높음/보통/낮음)",
    "target_audience": "대상 고객층"
}}
"""
    
    def _create_fallback_entities(self, user_input: str) -> Dict:
        """오류 시 기본 엔티티 반환"""
        return {
            "extracted_info": {
                "dates": [],
                "names": [],
                "locations": [],
                "events": [],
                "others": []
            },
            "message_intent": "일반안내",
            "context": user_input,
            "message_type": "정보성",
            "urgency_level": "보통",
            "target_audience": "일반고객"
        }
    
    def enhance_entities(self, entities: Dict, additional_context: str = "") -> Dict:
        """추출된 엔티티를 추가 컨텍스트로 보완"""
        if not additional_context:
            return entities
            
        prompt = f"""
기존 추출된 정보에 추가 컨텍스트를 반영하여 더 정확한 정보를 제공해주세요:

기존 정보: {json.dumps(entities, ensure_ascii=False, indent=2)}

추가 컨텍스트: "{additional_context}"

보완된 정보를 같은 JSON 형태로 반환해주세요.
"""
        
        try:
            response = self.generate_with_gemini(prompt)
            enhanced = self.parse_json_response(response)
            return enhanced if enhanced else entities
        except:
            return entities
    
    def validate_entities(self, entities: Dict) -> Dict:
        """엔티티 유효성 검증 및 점수화"""
        validation_result = {
            "is_valid": True,
            "completeness_score": 0,
            "quality_score": 0,
            "issues": []
        }
        
        extracted_info = entities.get("extracted_info", {})
        
        # 완성도 점수 계산
        total_categories = 5
        filled_categories = sum(1 for category in extracted_info.values() if category)
        validation_result["completeness_score"] = (filled_categories / total_categories) * 100
        
        # 품질 점수 계산
        quality_factors = []
        
        # 날짜 품질 체크
        dates = extracted_info.get("dates", [])
        if dates:
            # 날짜 형식 검증 로직
            quality_factors.append(0.8 if any(d for d in dates if len(d) > 4) else 0.3)
        
        # 이름 품질 체크
        names = extracted_info.get("names", [])
        if names:
            # 한국 이름 패턴 검증
            quality_factors.append(0.9 if any(len(n) >= 2 for n in names) else 0.4)
        
        # 기본 품질 점수
        validation_result["quality_score"] = (sum(quality_factors) / max(len(quality_factors), 1)) * 100 if quality_factors else 50
        
        # 이슈 체크
        if validation_result["completeness_score"] < 40:
            validation_result["issues"].append("추출된 정보가 부족합니다")
        
        if not entities.get("message_intent"):
            validation_result["issues"].append("메시지 의도가 불명확합니다")
        
        validation_result["is_valid"] = len(validation_result["issues"]) == 0
        
        return validation_result