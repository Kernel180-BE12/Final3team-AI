"""
3단계 템플릿 선택 시스템
1. 기존 템플릿 검색 (IndexManager)
2. 공용 템플릿 검색 (PublicTemplateManager)  
3. 새 템플릿 생성 (Agent1)
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .index_manager import IndexManager
from .public_template_manager import PublicTemplateManager, get_public_template_manager
from .template_matcher import TemplateMatcher
from ..agents.agent2 import Agent2
from ..utils.llm_provider_manager import get_llm_manager

@dataclass
class TemplateSelectionResult:
    """템플릿 선택 결과"""
    success: bool
    template: Optional[str] = None
    variables: Optional[List[Dict[str, Any]]] = None
    source: str = "unknown"  # "existing", "public", "generated"
    source_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    selection_path: Optional[List[str]] = None  # 선택 경로 추적

class TemplateSelector:
    """
    3단계 템플릿 선택 시스템
    """
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 각 단계별 매니저 초기화
        self.index_manager = IndexManager()
        self.public_template_manager = get_public_template_manager()
        self.llm_manager = get_llm_manager()
        
        # TemplateMatcher 초기화 (1단계용)
        self._template_matcher = None
        
        # Agent2는 필요시에만 초기화 (무거운 객체)
        self._agent2 = None
        
        # 단계별 임계값 설정
        self.existing_similarity_threshold = 0.7  # 기존 템플릿 유사도 임계값
        self.public_similarity_threshold = 0.6   # 공용 템플릿 유사도 임계값
        
    @property
    def template_matcher(self) -> TemplateMatcher:
        """지연 로딩으로 TemplateMatcher 초기화"""
        if self._template_matcher is None:
            self._template_matcher = TemplateMatcher(
                api_key=self.llm_manager.get_current_api_key(),
                index_manager=self.index_manager
            )
        return self._template_matcher
    
    @property
    def agent2(self) -> Agent2:
        """지연 로딩으로 Agent2 초기화"""
        if self._agent2 is None:
            self._agent2 = Agent2(index_manager=self.index_manager)
        return self._agent2
    
    def select_template(self, user_input: str, options: Dict[str, Any] = None) -> TemplateSelectionResult:
        """
        3단계 템플릿 선택 프로세스 실행
        
        Args:
            user_input: 사용자 입력
            options: 선택 옵션
                - force_generation: True시 생성 단계로 바로 이동
                - existing_threshold: 기존 템플릿 임계값 오버라이드
                - public_threshold: 공용 템플릿 임계값 오버라이드
                
        Returns:
            TemplateSelectionResult
        """
        if options is None:
            options = {}
        
        selection_path = []
        
        try:
            # 강제 생성 모드인 경우 3단계로 바로 이동
            if options.get('force_generation', False):
                selection_path.append("forced_generation")
                return self._generate_new_template(user_input, selection_path)
            
            # 1단계: 기존 템플릿 검색
            selection_path.append("stage1_existing")
            existing_result = self._search_existing_templates(user_input, options)
            if existing_result.success:
                existing_result.selection_path = selection_path
                return existing_result
            
            # 2단계: 공용 템플릿 검색
            selection_path.append("stage2_public")
            public_result = self._search_public_templates(user_input, options)
            if public_result.success:
                public_result.selection_path = selection_path
                return public_result
            
            # 3단계: 새 템플릿 생성
            selection_path.append("stage3_generation")
            generation_result = self._generate_new_template(user_input, selection_path)
            return generation_result
            
        except Exception as e:
            self.logger.error(f"템플릿 선택 중 오류: {e}")
            return TemplateSelectionResult(
                success=False,
                error=f"템플릿 선택 실패: {str(e)}",
                selection_path=selection_path
            )
    
    def _search_existing_templates(self, user_input: str, options: Dict[str, Any]) -> TemplateSelectionResult:
        """1단계: 기존 승인된 템플릿 검색 (TemplateMatcher 사용)"""
        try:
            threshold = options.get('existing_threshold', self.existing_similarity_threshold)
            
            self.logger.info(f"1단계: 임베딩 기반 기존 템플릿 검색 (임계값: {threshold})")
            
            # TemplateMatcher를 통한 유사 템플릿 검색
            similar_templates = self.template_matcher.find_similar_templates(user_input, top_k=3)
            
            if not similar_templates:
                return TemplateSelectionResult(
                    success=False,
                    source="existing",
                    error="유사한 기존 템플릿을 찾을 수 없습니다"
                )
            
            # 가장 유사도가 높은 템플릿 선택
            best_template = similar_templates[0]
            similarity_score = best_template['similarity_score']
            
            if similarity_score >= threshold:
                # 기존 템플릿을 표준 형식으로 변환
                template_content = best_template['content']
                
                # 표준 변수 형식으로 변환
                standardized_template, variables = self._convert_to_standard_template(template_content)
                
                self.logger.info(f"기존 템플릿 발견: {best_template['category']} (유사도: {similarity_score:.3f})")
                
                return TemplateSelectionResult(
                    success=True,
                    template=standardized_template,
                    variables=variables,
                    source="existing",
                    source_info={
                        "template_id": best_template['id'],
                        "category": best_template['category'],
                        "similarity": similarity_score,
                        "recommendation_type": best_template['recommendation_type'],
                        "keywords": best_template['keywords']
                    }
                )
            
            return TemplateSelectionResult(
                success=False,
                source="existing",
                error=f"기존 템플릿 유사도 부족 ({similarity_score:.3f} < {threshold})"
            )
            
        except Exception as e:
            self.logger.warning(f"기존 템플릿 검색 실패: {e}")
            return TemplateSelectionResult(
                success=False,
                source="existing",
                error=f"기존 템플릿 검색 오류: {str(e)}"
            )
    
    def _calculate_predata_similarity(self, user_input: str, predata_content: str) -> float:
        """predata 내용과 사용자 입력 간 유사도 계산"""
        from difflib import SequenceMatcher
        
        # 키워드 추출
        keywords = self._extract_keywords(user_input)
        predata_lower = predata_content.lower()
        
        # 키워드 포함도 계산
        matching_keywords = sum(1 for keyword in keywords if keyword.lower() in predata_lower)
        keyword_ratio = matching_keywords / len(keywords) if keywords else 0
        
        # 텍스트 유사도 계산
        text_similarity = SequenceMatcher(None, user_input.lower(), predata_lower[:1000]).ratio()
        
        # 가중평균 (키워드 포함도에 더 높은 가중치)
        final_similarity = (text_similarity * 0.2) + (keyword_ratio * 0.8)
        
        return final_similarity
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출 (공용 템플릿 매니저와 동일한 로직)"""
        import re
        
        stopwords = {'을', '를', '이', '가', '은', '는', '의', '에', '로', '으로', '와', '과', '도', '만', '까지', '부터', '에서'}
        
        # 특수문자 제거 및 공백으로 분리
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        words = cleaned.split()
        
        # 불용어 제거 및 2글자 이상 키워드만 추출
        keywords = [word for word in words if len(word) >= 2 and word not in stopwords]
        
        return keywords
    
    def _search_public_templates(self, user_input: str, options: Dict[str, Any]) -> TemplateSelectionResult:
        """2단계: 공용 템플릿 검색"""
        try:
            threshold = options.get('public_threshold', self.public_similarity_threshold)
            
            self.logger.info(f"2단계: 공용 템플릿 검색 (임계값: {threshold})")
            
            # 공용 템플릿 검색
            matches = self.public_template_manager.search_by_keywords(user_input, threshold)
            
            if not matches:
                return TemplateSelectionResult(
                    success=False,
                    source="public",
                    error="매칭되는 공용 템플릿 없음"
                )
            
            # 가장 유사도가 높은 템플릿 선택
            best_template, similarity = matches[0]
            
            self.logger.info(f"공용 템플릿 발견: {best_template.template_name} (유사도: {similarity:.3f})")
            
            # 표준 형식으로 변환
            converted = self.public_template_manager.convert_to_standard_format(best_template)
            
            return TemplateSelectionResult(
                success=True,
                template=converted["template"],
                variables=converted["variables"],
                source="public",
                source_info={
                    "template_code": converted["source_code"],
                    "template_name": converted["source_name"],
                    "similarity": similarity,
                    "original_variables": len(best_template.variables)
                }
            )
            
        except Exception as e:
            self.logger.warning(f"공용 템플릿 검색 실패: {e}")
            return TemplateSelectionResult(
                success=False,
                source="public", 
                error=f"공용 템플릿 검색 오류: {str(e)}"
            )
    
    def _generate_new_template(self, user_input: str, selection_path: List[str]) -> TemplateSelectionResult:
        """3단계: 새 템플릿 생성"""
        try:
            self.logger.info("3단계: 새 템플릿 생성")
            
            # Agent2를 통한 템플릿 생성
            result, tools_data = self.agent2.generate_compliant_template(user_input)
            
            if not result.get("success", False):
                return TemplateSelectionResult(
                    success=False,
                    source="generated",
                    error=result.get("error", "템플릿 생성 실패"),
                    selection_path=selection_path
                )
            
            # 변수 형식을 표준 형식으로 변환
            template = result.get("template", "")
            variables = result.get("variables", [])
            
            # 표준 변수 형식으로 변환 (#{변수명} -> ${변수명})
            standardized_template, standardized_variables = self._standardize_variables(template, variables)
            
            return TemplateSelectionResult(
                success=True,
                template=standardized_template,
                variables=standardized_variables,
                source="generated",
                source_info={
                    "original_variables": len(variables),
                    "generation_method": "agent1"
                },
                selection_path=selection_path
            )
            
        except Exception as e:
            self.logger.error(f"새 템플릿 생성 실패: {e}")
            return TemplateSelectionResult(
                success=False,
                source="generated",
                error=f"템플릿 생성 오류: {str(e)}",
                selection_path=selection_path
            )
    
    def _standardize_variables(self, template: str, variables: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        변수 형식을 표준 형식으로 변환
        #{변수명} -> ${변수명} 형식
        """
        standardized_template = template
        standardized_variables = []
        
        for var in variables:
            old_format = var.get("name", f"#{{{var.get('description', '변수')}}}")
            # 의미있는 변수명 생성
            description = var.get("description", "변수")
            clean_name = description.replace(" ", "").replace("(", "").replace(")", "")
            new_format = f"${{{clean_name}}}"
            
            # 템플릿에서 변수 교체
            standardized_template = standardized_template.replace(old_format, new_format)
            
            # 변수 정보 표준화
            standardized_variables.append({
                "name": new_format,
                "description": var.get("description", "변수"),
                "required": var.get("required", True),
                "original_name": old_format
            })
        
        return standardized_template, standardized_variables
    
    def _convert_to_standard_template(self, template_content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        기존 템플릿을 표준 변수 형식으로 변환
        [제목] 형태나 특정 패턴을 ${변수명} 형식으로 변환
        """
        import re
        
        standardized_template = template_content
        variables = []
        
        # [제목] 패턴을 변수로 변환
        title_pattern = r'\[(.*?)\]'
        titles = re.findall(title_pattern, standardized_template)
        
        for title in titles:
            old_format = f"[{title}]"
            clean_name = title.replace(" ", "").replace("(", "").replace(")", "")
            new_format = f"${{{clean_name}}}"
            standardized_template = standardized_template.replace(old_format, new_format)
            
            variables.append({
                "name": new_format,
                "description": title,
                "required": True
            })
        
        # 기본 변수가 없으면 기본 템플릿 구조 추가
        if not variables:
            # 간단한 템플릿 구조로 변환
            standardized_template = f"안녕하세요, ${{고객명}}님!\n\n{template_content}\n\n자세한 내용은 ${{상세정보}}를 확인해주세요."
            
            variables = [
                {
                    "name": "${고객명}",
                    "description": "고객명",
                    "required": True
                },
                {
                    "name": "${상세정보}",
                    "description": "상세정보",
                    "required": True
                }
            ]
        
        return standardized_template, variables
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """선택 시스템 통계 정보"""
        return {
            "existing_templates": {
                "available": "predata 기반 검색",
                "threshold": self.existing_similarity_threshold
            },
            "public_templates": {
                "available": len(self.public_template_manager.get_all_templates()),
                "threshold": self.public_similarity_threshold,
                "stats": self.public_template_manager.get_template_stats()
            },
            "generation": {
                "agent_ready": self._agent2 is not None,
                "llm_status": self.llm_manager.get_current_status()
            }
        }


# 전역 인스턴스
_global_template_selector = None

def get_template_selector() -> TemplateSelector:
    """전역 템플릿 선택기 인스턴스 반환"""
    global _global_template_selector
    if _global_template_selector is None:
        _global_template_selector = TemplateSelector()
    return _global_template_selector


if __name__ == "__main__":
    # 테스트
    print("=== 3단계 템플릿 선택 시스템 테스트 ===")
    
    try:
        selector = TemplateSelector()
        
        # 통계 정보
        stats = selector.get_selection_stats()
        print(f"시스템 상태:")
        print(f"- 공용 템플릿: {stats['public_templates']['available']}개")
        print(f"- LLM 상태: {stats['generation']['llm_status']['available_providers']}")
        
        # 템플릿 선택 테스트
        print("\n=== 템플릿 선택 테스트 ===")
        test_input = "국가 건강검진 안내 메시지 만들어줘"
        
        result = selector.select_template(test_input)
        
        print(f"선택 결과: {result.success}")
        print(f"선택 경로: {' -> '.join(result.selection_path or [])}")
        print(f"소스: {result.source}")
        
        if result.success:
            print(f"템플릿: {result.template[:100]}...")
            print(f"변수 수: {len(result.variables or [])}")
        else:
            print(f"오류: {result.error}")
            
    except Exception as e:
        print(f" 테스트 실패: {e}")