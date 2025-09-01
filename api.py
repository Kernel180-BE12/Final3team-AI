"""
카카오 알림톡 템플릿 생성 API
백엔드 연동을 위한 깔끔한 API 인터페이스
"""
import json
import re
import requests
from datetime import datetime
from typing import Dict, Optional, List
from config import GEMINI_API_KEY
from src.core import EntityExtractor, TemplateGenerator
from src.core.index_manager import get_index_manager
from src.utils import DataProcessor
from src.agents.agent2 import Agent2


class TemplateAPI:
    """백엔드 연동을 위한 템플릿 생성 API"""
    
    def __init__(self):
        """API 초기화"""
        print(" Template API 초기화 중...")
        
        # 인덱스 매니저 초기화
        self.index_manager = get_index_manager()
        
        # 핵심 컴포넌트 초기화
        self.entity_extractor = EntityExtractor(GEMINI_API_KEY)
        self.template_generator = TemplateGenerator(GEMINI_API_KEY)
        self.data_processor = DataProcessor()
        self.agent2 = Agent2(GEMINI_API_KEY, index_manager=self.index_manager)
        
        # 템플릿 비교 학습 시스템 추가
        self.template_learner = TemplateComparisonLearner()
        
        # 가이드라인 및 인덱스 구축
        self._initialize_indexes()
        
        print(" Template API 준비 완료")
    
    def _initialize_indexes(self):
        """인덱스 초기화"""
        try:
            # 가이드라인 로드
            guidelines = self.index_manager.get_guidelines_chunks(
                chunk_func=self.entity_extractor.chunk_text,
                chunk_size=800,
                overlap=100
            )
            
            # 샘플 템플릿 로드
            templates = self._get_sample_templates()
            
            # 인덱스 구축
            if guidelines:
                self.entity_extractor.guideline_index = self.index_manager.get_faiss_index(
                    index_name="guidelines",
                    data=guidelines,
                    encode_func=self.entity_extractor.encode_texts,
                    build_func=self.entity_extractor.build_faiss_index
                )
                self.entity_extractor.guidelines = guidelines
            
            if templates:
                import re
                clean_templates = [re.sub(r"#\{[^}]+\}", "[VARIABLE]", t) for t in templates]
                self.template_generator.template_index = self.index_manager.get_faiss_index(
                    index_name="templates",
                    data=clean_templates,
                    encode_func=self.template_generator.encode_texts,
                    build_func=self.template_generator.build_faiss_index
                )
                self.template_generator.templates = templates
                
        except Exception as e:
            print(f" 인덱스 초기화 오류: {e}")
    
    def _get_sample_templates(self) -> list:
        """샘플 템플릿"""
        return [
            "[가격 변경 안내]\n\n안녕하세요, #{수신자명}님.\n#{서비스명} 서비스 가격 변경을 안내드립니다.\n\n 변경 적용일: #{적용일}\n 기존 가격: #{기존가격}원\n 변경 가격: #{변경가격}원\n\n[변경 사유 및 개선사항]\n#{변경사유}에 따라 서비스 품질 개선을 위해 가격을 조정합니다.\n주요 개선사항: #{개선사항}\n\n[기존 이용자 안내]\n- 현재 이용 중인 서비스: #{유예기간}까지 기존 가격 적용\n- 자동 연장 서비스: 변경된 가격으로 갱신\n- 서비스 해지 희망: #{해지마감일}까지 신청 가능\n\n[문의 및 지원]\n- 고객센터: #{고객센터번호}\n- 상담시간: 평일 09:00-18:00\n- 온라인 문의: #{문의링크}\n\n※ 본 메시지는 정보통신망법에 따라 서비스 약관 변경 안내를 위해 발송된 정보성 메시지입니다.",
            "[#{매장명} 방문 예약 확인]\n\n#{고객명}님, 안녕하세요.\n#{매장명} 방문 예약이 완료되었습니다.\n\n 예약 정보\n- 예약번호: #{예약번호}\n- 방문일시: #{방문일시}\n- 예상 소요시간: #{소요시간}\n- 담당 직원: #{담당자명}\n\n 매장 정보\n- 위치: #{매장주소}\n- 연락처: #{매장전화번호}\n- 주차: #{주차안내}\n\n[방문 전 준비사항]\n- 신분증 지참 필수 (본인 확인)\n- 예약 10분 전 도착 권장\n- 마스크 착용 협조\n- 예약 확인 문자 제시\n\n[교통 및 위치 안내]\n- 대중교통: #{교통편안내}\n- 자가용: #{길찾기정보}\n- 주변 랜드마크: #{랜드마크}\n\n[예약 변경 및 취소]\n방문 예정일 1일 전까지 변경/취소 가능\n- 전화: #{매장전화번호}\n- 온라인: #{변경링크}\n- 문자 회신으로도 변경 가능\n\n※ 본 메시지는 매장 방문 예약 신청고객에게 발송되는 예약 확인 메시지입니다.",
            "[#{행사명} 참가 안내]\n\n#{수신자명}님, 안녕하세요.\n#{주최기관}에서 개최하는 #{행사명} 참가를 안내드립니다.\n\n 행사 개요\n- 행사명: #{행사명}\n- 일시: #{행사일시}\n- 장소: #{행사장소}\n- 대상: #{참가대상}\n- 참가비: #{참가비}\n\n 프로그램 일정\n#{프로그램일정상세}\n\n 참가 신청\n- 신청 방법: #{신청방법}\n- 신청 마감: #{신청마감일}\n- 신청 문의: #{신청문의전화}\n- 온라인 신청: #{신청링크}\n\n[준비물 및 복장]\n- 필수 준비물: #{필수준비물}\n- 권장 복장: #{복장안내}\n- 개인 준비물: #{개인준비물}\n\n[행사장 안내]\n- 상세 주소: #{상세주소}\n- 교통편: #{교통편}\n- 주차 시설: #{주차정보}\n- 편의 시설: #{편의시설}\n\n[주의사항 및 안내]\n- 코로나19 방역수칙 준수\n- 행사 당일 발열체크 실시\n- 우천 시 일정: #{우천시대안}\n- 기타 문의: #{기타문의처}\n\n※ 본 메시지는 #{행사명} 관심 등록자에게 발송되는 행사 안내 메시지입니다."
        ]
    
    def generate_template(self, user_input: str, options: Optional[Dict] = None) -> Dict:
        """
        템플릿 생성 메인 API
        
        Args:
            user_input: 사용자 요청
            options: 생성 옵션 (use_agent2, method 등)
            
        Returns:
            생성 결과 딕셔너리
        """
        if not user_input or not user_input.strip():
            return {
                "success": False,
                "error": "입력이 비어있습니다.",
                "template": None,
                "metadata": None
            }
        
        try:
            # 1. 템플릿 비교 학습 - 새로운 유형인지 확인
            novelty_analysis = self.template_learner.analyze_input_novelty(user_input)
            
            # 기존 템플릿과 유사한 경우 경고 반환 (기업이 처리)
            if not novelty_analysis["is_new_type"]:
                return {
                    "success": False,
                    "is_duplicate": True,
                    "message": novelty_analysis["recommendation"]["message"],
                    "recommended_template_id": novelty_analysis["recommendation"].get("recommended_template_id"),
                    "similarity_score": novelty_analysis["recommendation"].get("similarity_score"),
                    "template": None,
                    "metadata": {"novelty_analysis": novelty_analysis}
                }
            
            # 2. 새로운 유형인 경우 생성 진행
            print(f" 새로운 유형 템플릿 생성: '{user_input}'")
            
            # 기본 옵션 설정
            if options is None:
                options = {}
            
            use_agent2 = options.get("use_agent2", True)
            method = options.get("method", "default")
            
            # 템플릿 생성
            if use_agent2:
                # Agent2 방식 (권장)
                template, metadata = self.agent2.generate_compliant_template(user_input)
                entities = self.entity_extractor.extract_entities(user_input)
                
                result = {
                    "success": True,
                    "template": template,
                    "is_new_type": True,
                    "novelty_analysis": novelty_analysis,
                    "metadata": {
                        "method": "Agent2",
                        "entities": entities,
                        "agent2_metadata": metadata,
                        "quality_assured": True,
                        "guidelines_compliant": True,
                        "template_learning": novelty_analysis
                    }
                }
            else:
                # 순수 신규 생성 방식 (기업 요구사항)
                entities = self.entity_extractor.extract_entities(user_input)
                
                # 기존 템플릿 검색 제거 - 기업이 별도 처리
                # 가이드라인만 검색 (규정 준수)
                relevant_guidelines = self.entity_extractor.search_similar(
                    user_input + " " + entities.get("message_intent", ""),
                    self.entity_extractor.guideline_index,
                    self.entity_extractor.guidelines,
                    top_k=3,
                )
                guidelines = [guideline for guideline, _ in relevant_guidelines]
                
                # 순수 신규 템플릿 생성 (기존 템플릿 참고 없음)
                template, filled_template = self.template_generator.generate_template(
                    user_input, entities, [], guidelines  # similar_templates를 빈 배열로
                )
                
                optimized_template = self.template_generator.optimize_template(
                    template, entities
                )
                
                variables = self.template_generator.extract_variables(optimized_template)
                
                result = {
                    "success": True,
                    "template": optimized_template,
                    "is_new_type": True,
                    "novelty_analysis": novelty_analysis,
                    "metadata": {
                        "method": "Pure_New_Generation",
                        "entities": entities,
                        "variables": variables,
                        "filled_template": filled_template,
                        "quality_assured": True,
                        "guidelines_compliant": True,
                        "template_learning": novelty_analysis
                    }
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "template": None,
                "metadata": None
            }
    
    def health_check(self) -> Dict:
        """API 상태 확인"""
        try:
            cache_info = self.index_manager.get_cache_info()
            return {
                "status": "healthy",
                "components": {
                    "entity_extractor": "ready",
                    "template_generator": "ready",
                    "agent2": "ready",
                    "index_manager": "ready"
                },
                "cache_info": cache_info,
                "indexes": {
                    "guidelines": self.entity_extractor.guideline_index is not None,
                    "templates": self.template_generator.template_index is not None
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def export_to_json(self, result: Dict, user_input: str = "", user_id: int = 101, category_id: int = None, industry_id: int = None) -> Dict:
        """템플릿 생성 결과를 DB 저장용 JSON으로 변환 (기업 스키마 맞춤)"""
        if not result.get("success"):
            return {
                "error": "템플릿 생성에 실패했습니다.",
                "data": None
            }
        
        template = result.get("template", "")
        metadata = result.get("metadata", {})
        entities_data = metadata.get("entities", {})
        # 사용자 입력 우선순위: 파라미터 > result > metadata
        final_user_input = user_input or result.get("user_input", "") or result.get("metadata", {}).get("user_input", "")
        
        # 변수 추출
        variables = self._extract_variables_from_template(template)
        
        # 엔티티 데이터 정규화
        entities = self._normalize_entities(entities_data)
        
        # 템플릿 제목 생성 (사용자 입력에서 추출)
        title = self._generate_title_from_input(final_user_input)
        
        # 자동 업종/카테고리 감지 (매개변수로 지정되지 않은 경우)
        if category_id is None or industry_id is None:
            auto_category, auto_industry = self._auto_detect_industry_and_category(final_user_input, title)
            category_id = category_id or auto_category
            industry_id = industry_id or auto_industry
        
        # field 값 생성 (템플릿 식별자)
        field_value = self._generate_field_value(title, final_user_input)
        
        # 기업 DB 스키마에 맞춘 JSON 구조
        json_data = {
            "template": {
                "user_id": user_id,
                "category_id": category_id,
                "title": title,
                "content": template,
                "status": "CREATE_REQUESTED",  # 초안 상태
                "type": "MESSAGE",
                "is_public": 1,  # 공개 (기업 베스트 템플릿처럼)
                "image_url": None,
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "updated_at": datetime.now().strftime("%Y-%m-%d"),
                "field": field_value
            },
            "entities": entities,  # template_entities 테이블용
            "variables": variables,  # template_variables 테이블용
            "industry_mapping": {  # template_industry 테이블용 (새로 추가)
                "industry_id": industry_id,
                "template_id": None,  # 백엔드에서 생성된 template_id 사용
                "created_at": datetime.now().strftime("%Y-%m-%d")
            } if industry_id else None,
            "metadata": {  # template_metadata 테이블용
                "generation_method": metadata.get("method", "Agent2"),
                "user_input": final_user_input,
                "quality_assured": metadata.get("quality_assured", False),
                "guidelines_compliant": metadata.get("guidelines_compliant", False),
                "legal_compliant": metadata.get("legal_compliant", False),
                "message_intent": entities_data.get("message_intent", "일반안내"),
                "message_type": entities_data.get("message_type", "정보성"),
                "urgency_level": entities_data.get("urgency_level", "normal"),
                "estimated_length": len(template),
                "agent2_metadata": metadata.get("agent2_metadata", {}),
                "created_at": datetime.now().isoformat()
            }
        }
        
        return {
            "success": True,
            "data": json_data
        }
    
    def _extract_variables_from_template(self, template: str) -> List[Dict]:
        """템플릿에서 변수 추출 (기업 형식에 맞춤)"""
        variables = []
        # 기업 베스트 템플릿은 #{변수명} 형식을 사용
        variable_pattern = r'#\{([^}]+)\}'
        matches = re.findall(variable_pattern, template)
        
        for var_name in set(matches):  # 중복 제거
            variable_info = {
                "template_id": None,  # 백엔드에서 생성된 template_id 사용
                "variable_key": var_name,  # 기업 형식: variable_key
                "placeholder": f"{{{var_name}}}",  # 기업 형식: {변수명}
                "input_type": "TEXT",  # 기업 형식: 모두 TEXT
                "created_at": datetime.now().strftime("%Y-%m-%d")
            }
            variables.append(variable_info)
        
        return variables
    
    def _infer_variable_type(self, var_name: str) -> str:
        """변수명으로 타입 추론"""
        var_name_lower = var_name.lower()
        
        if any(keyword in var_name_lower for keyword in ['일시', '날짜', '시간', '적용일']):
            return 'date'
        elif any(keyword in var_name_lower for keyword in ['번호', '가격', '금액', '수량']):
            return 'number'  
        elif any(keyword in var_name_lower for keyword in ['이메일', 'email']):
            return 'email'
        elif any(keyword in var_name_lower for keyword in ['전화', '연락처', '휴대폰']):
            return 'phone'
        else:
            return 'text'
    
    def _normalize_entities(self, entities_data: Dict) -> List[Dict]:
        """엔티티 데이터를 DB 저장용으로 정규화"""
        entities = []
        extracted_info = entities_data.get("extracted_info", {})
        
        for entity_type, values in extracted_info.items():
            if values and isinstance(values, list):
                for value in values:
                    entities.append({
                        "entity_type": entity_type,
                        "entity_value": str(value),
                        "confidence_score": 0.9  # 기본 신뢰도
                    })
        
        return entities
    
    def _generate_title_from_input(self, user_input: str) -> str:
        """사용자 입력에서 템플릿 제목 생성"""
        # 키워드 기반 제목 생성
        keywords = {
            '쿠폰': '쿠폰 발급 안내',
            '할인': '할인 혜택 안내', 
            '행사': '특별 행사 안내',
            '이벤트': '이벤트 참가 안내',
            '모임': '모임 참석 안내',
            '예약': '예약 확인 안내',
            '방문': '방문 예약 안내',
            '서비스': '서비스 이용 안내',
            'A/S': 'A/S 서비스 안내',
            '점검': '점검 일정 안내',
            '회원': '회원 혜택 안내',
            '치과': '치과 진료 안내',
            '가격': '가격 변경 안내'
        }
        
        for keyword, title in keywords.items():
            if keyword in user_input:
                return title
                
        # 기본 제목 (30자 제한)
        if len(user_input) > 30:
            return user_input[:27] + "..."
        return user_input
    
    def _generate_field_value(self, title: str, user_input: str) -> str:
        """field 값 생성 (템플릿 식별자)"""
        # 기업 베스트 템플릿의 field 패턴 분석하여 생성
        import hashlib
        import time
        
        # 제목과 입력을 기반으로 고유 식별자 생성
        base_string = f"{title}_{user_input}_{int(time.time())}"
        hash_value = hashlib.md5(base_string.encode()).hexdigest()[:10]
        
        # 카테고리별 접두사
        category_prefixes = {
            '쿠폰': 'coupon_noti',
            '할인': 'discount_noti', 
            '행사': 'event_noti',
            '예약': 'reservation',
            '방문': 'visit_noti',
            '가격': 'price_noti',
            '강의': 'lesson_info',
            '학습': 'lesson_info',
            '적립금': 'point_expired',
            '꿀팁': 'tip_received',
            '회원': 'signup_done',
            '업데이트': 'update_noti',
            '샘플': 'sample_noti',
            '결과': 'result_noti',
            '당첨': 'winner_noti',
            '응모권': 'ticket_noti',
            '이용권': 'usage_noti',
            '등급': 'grade_noti',
            '구매': 'purchase_noti',
            '인보이스': 'invoice_received'
        }
        
        prefix = "ai_generated"
        for keyword, pre in category_prefixes.items():
            if keyword in title or keyword in user_input:
                prefix = pre
                break
        
        return f"{prefix}_{hash_value}"
    
    def _auto_detect_industry_and_category(self, user_input: str, title: str) -> tuple:
        """사용자 입력을 기반으로 업종과 카테고리 자동 감지"""
        # 기업 베스트 템플릿 분석 결과
        industry_mapping = {
            # 소매업 (상품, 쿠폰, 할인)
            '소매업': ['쿠폰', '할인', '상품', '구매', '샘플', '브랜드', '매장', '상점'],
            # 부동산 (예약, 방문, 상담)
            '부동산': ['예약', '방문', '상담', '매물', '부동산', '아파트', '오피스텔'],
            # 교육 (강의, 학습, 수업)
            '교육': ['강의', '학습', '수업', '교육', '학원', '과정', '수강', '학습자'],
            # 서비스업 (뷰티, 건강, PT, 업데이트)
            '서비스업': ['뷰티', '건강', 'PT', '시술', '관리', '서비스', '업데이트', '이용권'],
            # 기타 (공통 - 회원, 적립금, 등급, 이벤트)
            '기타': ['회원', '적립금', '등급', '이벤트', '당첨', '응모권', '인보이스']
        }
        
        # 카테고리 매핑 (기업 스키마)
        category_mapping = {
            '이용안내': 9101,  # 가격변경, 행사안내, 강의일정, 학습사항, 적립금, 꿀팁, 회원혜택, 업데이트, 샘플, 쿠폰, 당첨, 응모권, 이용권, 등급, 구매
            '예약완료': 9301,  # 방문예약, 상담예약  
            '피드백': 9201    # PT결과, 인보이스
        }
        
        # 업종 감지
        detected_industry = '기타'  # 기본값
        for industry, keywords in industry_mapping.items():
            if any(keyword in user_input or keyword in title for keyword in keywords):
                detected_industry = industry
                break
        
        # 카테고리 감지
        detected_category = 9101  # 기본값: 이용안내
        if any(keyword in user_input or keyword in title for keyword in ['예약', '방문', '상담']):
            detected_category = 9301  # 예약완료
        elif any(keyword in user_input or keyword in title for keyword in ['결과', '인보이스', '피드백']):
            detected_category = 9201  # 피드백
            
        # 업종 ID 매핑 (실제로는 DB에서 조회해야 함)
        industry_id_mapping = {
            '소매업': 1,
            '부동산': 2, 
            '교육': 3,
            '서비스업': 4,
            '기타': 5
        }
        
        return detected_category, industry_id_mapping.get(detected_industry, 5)
    
    def send_to_backend(self, json_data: Dict, backend_url: str, 
                       headers: Optional[Dict] = None) -> Dict:
        """백엔드 서버로 JSON 데이터 전송 (기업 스키마 맞춤)"""
        if headers is None:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Template-Generator/1.0",
                "Accept": "application/json"
            }
        
        try:
            # 기업 백엔드 API 엔드포인트에 맞춤
            response = requests.post(
                f"{backend_url}/api/v1/templates/create",
                json=json_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                response_data = response.json() if response.content else {}
                return {
                    "success": True,
                    "message": "템플릿이 DB에 성공적으로 저장되었습니다",
                    "template_id": response_data.get("template_id"),
                    "status": response_data.get("status", "CREATE_REQUESTED"),
                    "response": response_data
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "message": "템플릿 저장에 실패했습니다",
                    "response": None
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"백엔드 통신 실패: {str(e)}",
                "message": "백엔드 서버와 연결할 수 없습니다",
                "response": None
            }
    
    def generate_and_send(self, user_input: str, backend_url: str, 
                         options: Optional[Dict] = None) -> Dict:
        """템플릿 생성 후 즉시 백엔드로 전송"""
        # 1. 템플릿 생성
        result = self.generate_template(user_input, options)
        
        if not result.get("success"):
            return result
        
        # 2. JSON 변환
        json_export = self.export_to_json(result, user_input)
        
        if not json_export.get("success"):
            return json_export
        
        # 3. 백엔드 전송
        send_result = self.send_to_backend(json_export["data"], backend_url)
        
        return {
            "template_generation": result,
            "json_export": json_export,
            "backend_send": send_result,
            "overall_success": send_result.get("success", False)
        }


# 전역 API 인스턴스
_api_instance = None

def get_template_api() -> TemplateAPI:
    """싱글톤 API 인스턴스 가져오기"""
    global _api_instance
    if _api_instance is None:
        _api_instance = TemplateAPI()
    return _api_instance


# 간편한 함수형 인터페이스
def generate_template(user_input: str, **options) -> Dict:
    """템플릿 생성 함수"""
    api = get_template_api()
    return api.generate_template(user_input, options)

def api_health_check() -> Dict:
    """API 상태 확인 함수"""
    api = get_template_api()
    return api.health_check()


if __name__ == "__main__":
    # API 테스트
    api = get_template_api()
    
    # 헬스 체크
    health = api.health_check()
    print("=== API Health Check ===")
    print(json.dumps(health, indent=2, ensure_ascii=False))
    
    # 템플릿 생성 테스트
    test_input = "내일 오후 2시에 카페에서 모임이 있다고 알림톡 보내줘"
    result = api.generate_template(test_input)
    
    print("\n=== Template Generation Test ===")
    print(f"Input: {test_input}")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Template:\n{result['template']}")
    else:
        print(f"Error: {result['error']}")