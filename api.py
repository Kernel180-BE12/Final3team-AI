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
from src.core import EntityExtractor
from src.core.index_manager import get_index_manager
from src.utils import DataProcessor
from src.agents.agent1 import Agent1
from src.agents.agent2 import Agent2
from src.evaluation.ragas_evaluator import TemplateRAGASEvaluator
from src.utils.llm_provider_manager import get_llm_manager, invoke_llm_with_fallback
from src.core.template_selector import get_template_selector


class TemplateAPI:
    """백엔드 연동을 위한 템플릿 생성 API"""
    
    def __init__(self):
        """API 초기화"""
        print(" Template API 초기화 중...")
        
        # 공통 초기화 모듈 사용
        from src.utils.common_init import initialize_core_components, setup_guidelines_and_indexes
        
        (self.index_manager, self.entity_extractor, 
         self.data_processor, self.agent2) = initialize_core_components()
        
        # Agent1 초기화 추가
        self.agent1 = Agent1()
        
        # LLM 공급자 관리자 초기화
        self.llm_manager = get_llm_manager()
        
        # RAGAS 평가기 초기화 추가
        self.evaluator = TemplateRAGASEvaluator(GEMINI_API_KEY)
        
        # 3단계 템플릿 선택기 초기화
        self.template_selector = get_template_selector()
        
        # TODO: 템플릿 비교 학습 시스템 구현 필요
        
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
                self.entity_extractor.guideline_collection = self.index_manager.get_chroma_collection(
                    collection_name="guidelines",
                    data=guidelines,
                    encode_func=self.entity_extractor.encode_texts
                )
                self.entity_extractor.guidelines = guidelines
            
            if templates:
                import re
                clean_templates = [re.sub(r"#\{[^}]+\}", "[VARIABLE]", t) for t in templates]
                self.template_generator.template_collection = self.index_manager.get_chroma_collection(
                    collection_name="templates",
                    data=clean_templates,
                    encode_func=self.template_generator.encode_texts
                )
                self.template_generator.templates = templates
                
        except Exception as e:
            print(f" 인덱스 초기화 오류: {e}")
    
    def _get_sample_templates(self) -> list:
        """샘플 템플릿"""
        from src.utils.sample_templates import get_sample_templates
        return get_sample_templates()
    
    def generate_template(self, user_input: str, options: Optional[Dict] = None, with_ragas_gate: bool = True) -> Dict:
        """
        템플릿 생성 메인 API (3단계 선택 시스템 적용)
        
        Args:
            user_input: 사용자 요청
            options: 생성 옵션 (use_agent2, method, force_generation 등)
            with_ragas_gate: RAGAS 품질 검증 게이트 사용 여부 (기본: True)
            
        Returns:
            생성 결과 딕셔너리 (검증 통과한 템플릿만 반환)
        """
        if not user_input or not user_input.strip():
            return {
                "success": False,
                "error": "입력이 비어있습니다.",
                "template": None,
                "metadata": None
            }
        
        try:
            print(f"🎯 3단계 템플릿 선택 시스템 시작: '{user_input}'")
            
            # 기본 옵션 설정
            if options is None:
                options = {}
            
            # 3단계 템플릿 선택 실행
            selection_result = self.template_selector.select_template(user_input, options)
            
            if not selection_result.success:
                return {
                    "success": False,
                    "error": f"템플릿 선택 실패: {selection_result.error}",
                    "template": None,
                    "metadata": {
                        "selection_path": selection_result.selection_path,
                        "source": selection_result.source
                    }
                }
            
            # 선택된 템플릿 정보
            template = selection_result.template
            variables = selection_result.variables or []
            source = selection_result.source
            
            print(f"✅ 템플릿 선택 완료: {source} (경로: {' -> '.join(selection_result.selection_path or [])})")
            
            # 메타데이터 구성
            metadata = {
                "source": source,
                "selection_path": selection_result.selection_path,
                "source_info": selection_result.source_info,
                "variables": variables,
                "created_at": datetime.now().isoformat(),
                "ragas_verified": False
            }
            
            # 생성된 템플릿인 경우에만 RAGAS 검증 적용
            if source == "generated" and with_ragas_gate:
                print("🔍 RAGAS 품질 검증 시작")
                ragas_result = self._apply_ragas_quality_gate(user_input, {
                    "success": True,
                    "template": template,
                    "metadata": metadata
                }, options)
                
                if not ragas_result.get("success"):
                    return ragas_result
                
                metadata["ragas_verified"] = True
                metadata["ragas_scores"] = ragas_result.get("metadata", {}).get("ragas_scores")
                print("✅ RAGAS 품질 검증 통과")
            
            # 최종 결과 반환
            return {
                "success": True,
                "template": template,
                "variables": variables,
                "metadata": metadata
            }
            
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
            llm_status = self.llm_manager.get_current_status()
            template_selector_stats = self.template_selector.get_selection_stats()
            
            return {
                "status": "healthy",
                "components": {
                    "entity_extractor": "ready",
                    "agent1": "ready",
                    "agent2": "ready",
                    "index_manager": "ready",
                    "llm_manager": "ready",
                    "ragas_evaluator": "ready",
                    "template_selector": "ready"
                },
                "template_selection": template_selector_stats,
                "cache_info": cache_info,
                "llm_providers": {
                    "current_provider": llm_status["current_provider"],
                    "current_model": llm_status["current_model"],
                    "available_providers": llm_status["available_providers"],
                    "failure_counts": llm_status["failure_counts"],
                    "total_providers": llm_status["total_providers"]
                },
                "indexes": {
                    "guidelines": self.entity_extractor.guideline_collection is not None
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
                "category_id": str(category_id),
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
                "placeholder": f"#{{{var_name}}}",  # 기업 형식: #{변수명}
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
        
        # 카테고리 매핑 (새로운 카테고리 체계)
        category_mapping = {
            # 대분류 (001-009)
            '회원': '001',
            '구매': '002', 
            '예약': '003',
            '서비스이용': '004',
            '리포팅': '005',
            '배송': '006',
            '법적고지': '007',
            '업무알림': '008',
            '쿠폰/포인트': '009',
            '기타': '999',
            
            # 소분류 (상세 매핑)
            '회원가입': '001001',
            '인증/비밀번호/로그인': '001002', 
            '회원정보/회원혜택': '001003',
            '구매완료': '002001',
            '상품가입': '002002',
            '진행상태': '002003',
            '구매취소': '002004',
            '구매예약/입고알림': '002005',
            '예약완료/예약내역': '003001',
            '예약상태': '003002',
            '예약취소': '003003', 
            '예약알림/리마인드': '003004',
            '이용안내/공지': '004001',
            '신청접수': '004002',
            '처리완료': '004003',
            '이용도구': '004004',
            '방문서비스': '004005',
            '피드백요청': '004006',
            '구매감사/이용확인': '004007',
            '리마인드': '004008',
            '피드백': '005001',
            '요금청구': '005002',
            '계약/견적': '005003',
            '안전/피해예방': '005004',
            '뉴스레터': '005005',
            '거래알림': '005006',
            '배송상태': '006001',
            '배송예정': '006002',
            '배송완료': '006003',
            '배송실패': '006004',
            '수신동의': '007001',
            '개인정보': '007002',
            '약관변경': '007003',
            '휴면관련': '007004',
            '주문/예약': '008001',
            '내부업무알림': '008002',
            '쿠폰발급': '009001',
            '쿠폰사용': '009002',
            '포인트적립': '009003',
            '포인트사용': '009004',
            '쿠폰/포인트안내': '009005',
            '기타': '999999'
        }
        
        # 업종 감지
        detected_industry = '기타'  # 기본값
        for industry, keywords in industry_mapping.items():
            if any(keyword in user_input or keyword in title for keyword in keywords):
                detected_industry = industry
                break
        
        # 카테고리 감지
        detected_category = '004001'  # 기본값: 이용안내/공지
        
        # 키워드 기반 상세 카테고리 분류
        if any(keyword in user_input or keyword in title for keyword in ['회원가입', '가입', '신규']):
            detected_category = '001001'  # 회원가입
        elif any(keyword in user_input or keyword in title for keyword in ['로그인', '비밀번호', '인증']):
            detected_category = '001002'  # 인증/비밀번호/로그인
        elif any(keyword in user_input or keyword in title for keyword in ['회원정보', '회원혜택', '등급']):
            detected_category = '001003'  # 회원정보/회원혜택
        elif any(keyword in user_input or keyword in title for keyword in ['구매완료', '주문완료', '결제완료']):
            detected_category = '002001'  # 구매완료
        elif any(keyword in user_input or keyword in title for keyword in ['상품가입', '서비스가입']):
            detected_category = '002002'  # 상품가입
        elif any(keyword in user_input or keyword in title for keyword in ['진행상태', '처리중']):
            detected_category = '002003'  # 진행상태
        elif any(keyword in user_input or keyword in title for keyword in ['구매취소', '주문취소', '결제취소']):
            detected_category = '002004'  # 구매취소
        elif any(keyword in user_input or keyword in title for keyword in ['입고알림', '재입고']):
            detected_category = '002005'  # 구매예약/입고알림
        elif any(keyword in user_input or keyword in title for keyword in ['예약완료', '예약확인']):
            detected_category = '003001'  # 예약완료/예약내역
        elif any(keyword in user_input or keyword in title for keyword in ['예약상태', '예약변경']):
            detected_category = '003002'  # 예약상태
        elif any(keyword in user_input or keyword in title for keyword in ['예약취소']):
            detected_category = '003003'  # 예약취소
        elif any(keyword in user_input or keyword in title for keyword in ['예약알림', '리마인드']):
            detected_category = '003004'  # 예약알림/리마인드
        elif any(keyword in user_input or keyword in title for keyword in ['공지', '안내', '이용방법']):
            detected_category = '004001'  # 이용안내/공지
        elif any(keyword in user_input or keyword in title for keyword in ['신청접수', '접수완료']):
            detected_category = '004002'  # 신청접수
        elif any(keyword in user_input or keyword in title for keyword in ['처리완료', '완료알림']):
            detected_category = '004003'  # 처리완료
        elif any(keyword in user_input or keyword in title for keyword in ['배송상태', '배송조회']):
            detected_category = '006001'  # 배송상태
        elif any(keyword in user_input or keyword in title for keyword in ['배송예정', '발송예정']):
            detected_category = '006002'  # 배송예정
        elif any(keyword in user_input or keyword in title for keyword in ['배송완료', '배달완료']):
            detected_category = '006003'  # 배송완료
        elif any(keyword in user_input or keyword in title for keyword in ['배송실패', '배송지연']):
            detected_category = '006004'  # 배송실패
        elif any(keyword in user_input or keyword in title for keyword in ['쿠폰발급', '쿠폰지급']):
            detected_category = '009001'  # 쿠폰발급
        elif any(keyword in user_input or keyword in title for keyword in ['포인트적립', '적립금']):
            detected_category = '009003'  # 포인트적립
        elif any(keyword in user_input or keyword in title for keyword in ['피드백', '만족도', '후기']):
            detected_category = '005001'  # 피드백
            
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
                f"{backend_url}/api/templates/create",
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
    
    def _apply_ragas_quality_gate(self, user_input: str, initial_result: Dict, options: Optional[Dict] = None) -> Dict:
        """
        RAGAS 품질 검증 게이트 적용
        품질 기준을 통과할 때까지 재생성하거나 최종 실패 처리
        
        Args:
            user_input: 사용자 입력
            initial_result: 초기 템플릿 생성 결과
            options: 생성 옵션
            
        Returns:
            검증 통과한 결과 또는 실패 결과
        """
        max_retries = self.evaluator.QUALITY_THRESHOLDS["max_retries"]
        current_result = initial_result
        
        for attempt in range(max_retries + 1):  # 최초 시도 + 재시도
            try:
                print(f"🔍 RAGAS 품질 검증 {attempt + 1}/{max_retries + 1} 시도...")
                
                # 평가용 데이터 준비
                evaluation_data = [{
                    "user_input": user_input,
                    "template": current_result["template"],
                    "metadata": current_result["metadata"]
                }]
                
                # 평가 데이터셋 생성
                dataset = self.evaluator.create_evaluation_dataset(evaluation_data)
                
                # RAGAS 평가 실행
                evaluation_results = self.evaluator.evaluate_templates(dataset)
                
                if evaluation_results:
                    # 품질 검증 확인
                    quality_check = self.evaluator.check_quality_pass(evaluation_results)
                    
                    # 평가 정보를 결과에 추가
                    current_result["ragas_evaluation"] = {
                        "attempt": attempt + 1,
                        "scores": evaluation_results,
                        "quality_check": quality_check,
                        "average_score": quality_check["average_score"],
                        "passed": quality_check["passed"]
                    }
                    
                    if quality_check["passed"]:
                        print(f"✅ RAGAS 검증 통과! (평균: {quality_check['average_score']:.3f})")
                        current_result["quality_verified"] = True
                        return current_result
                    else:
                        print(f"❌ RAGAS 검증 실패 (평균: {quality_check['average_score']:.3f})")
                        print(f"실패 이유: {quality_check['reason']}")
                        
                        # 최대 재시도 전이면 재생성
                        if attempt < max_retries:
                            print(f"🔄 품질 개선을 위해 템플릿 재생성 중... ({attempt + 2}/{max_retries + 1})")
                            
                            # 개선 제안을 반영한 재생성 옵션
                            improved_options = options.copy() if options else {}
                            improved_options["improvement_suggestions"] = quality_check["suggestions"]
                            improved_options["failed_metrics"] = quality_check["failed_metrics"]
                            
                            # 재생성 (검증 게이트 비활성화하여 무한루프 방지)
                            retry_result = self.generate_template(
                                user_input, 
                                improved_options, 
                                with_ragas_gate=False
                            )
                            
                            if retry_result.get("success"):
                                current_result = retry_result
                                continue
                            else:
                                print("❌ 재생성 실패")
                                break
                        else:
                            print("❌ 최대 재시도 횟수 초과")
                            break
                else:
                    print("❌ RAGAS 평가 실행 실패")
                    break
                    
            except Exception as e:
                print(f"❌ RAGAS 검증 오류: {e}")
                break
        
        # 검증 실패 시 최종 결과 반환
        current_result["quality_verified"] = False
        current_result["ragas_evaluation"]["final_status"] = "검증 실패"
        current_result["ragas_evaluation"]["reason"] = "품질 기준 미달 또는 평가 오류"
        
        return {
            "success": False,
            "error": "템플릿이 품질 기준을 통과하지 못했습니다.",
            "template": None,
            "quality_verification_failed": True,
            "ragas_evaluation": current_result.get("ragas_evaluation", {}),
            "suggestions": current_result.get("ragas_evaluation", {}).get("quality_check", {}).get("suggestions", []),
            "last_attempt": current_result
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