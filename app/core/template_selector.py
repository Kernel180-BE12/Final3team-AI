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
from .basic_template_matcher import BasicTemplateMatcher
from .entity_extractor import EntityExtractor
from .template_validator import get_template_validator
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

    # 변수 수집 관련 필드들 (새로 추가)
    status: Optional[str] = None
    mapped_variables: Optional[Dict] = None
    missing_variables: Optional[List] = None
    partial_template: Optional[str] = None
    mapping_coverage: Optional[float] = None
    industry: Optional[List] = None
    purpose: Optional[List] = None

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

        # EntityExtractor 초기화 (BasicTemplateMatcher용)
        self._entity_extractor = None

        # BasicTemplateMatcher 초기화 (1단계용)
        self._basic_template_matcher = None
        
        # Agent2는 필요시에만 초기화 (무거운 객체)
        self._agent2 = None

        # 템플릿 검증 시스템
        self.template_validator = get_template_validator()

        # 단계별 임계값 설정
        self.existing_similarity_threshold = 0.85  # 기존 템플릿 유사도 임계값 (0.8→0.85 상향)
        self.public_similarity_threshold = 0.85  # 공용 템플릿 유사도 임계값 (0.75→0.85로 상향)

        # 재생성 설정
        self.max_regeneration_attempts = 3

    @property
    def entity_extractor(self) -> EntityExtractor:
        """지연 로딩으로 EntityExtractor 초기화"""
        if self._entity_extractor is None:
            self._entity_extractor = EntityExtractor(
                api_key=self.llm_manager.gemini_api_key
            )
        return self._entity_extractor

    @property
    def basic_template_matcher(self) -> BasicTemplateMatcher:
        """지연 로딩으로 BasicTemplateMatcher 초기화"""
        if self._basic_template_matcher is None:
            self._basic_template_matcher = BasicTemplateMatcher(
                index_manager=self.index_manager,
                entity_extractor=self.entity_extractor
            )
        return self._basic_template_matcher
    
    @property
    def agent2(self) -> Agent2:
        """지연 로딩으로 Agent2 초기화"""
        if self._agent2 is None:
            self._agent2 = Agent2(index_manager=self.index_manager)
        return self._agent2
    
    def select_template(self, user_input: str, options: Dict[str, Any] = None) -> TemplateSelectionResult:
        """
        3단계 템플릿 선택 프로세스 실행 - 동기 버전 (하위 호환성)

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

    async def select_template_async(self, user_input: str, options: Dict[str, Any] = None) -> TemplateSelectionResult:
        """
        3단계 템플릿 선택 프로세스 실행 - 비동기 버전

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
                return await self._generate_new_template_async(user_input, selection_path)

            # 1단계: 기존 템플릿 검색 (동기 - 빠른 처리)
            selection_path.append("stage1_existing")
            existing_result = self._search_existing_templates(user_input, options)
            if existing_result.success:
                existing_result.selection_path = selection_path
                return existing_result

            # 2단계: 공용 템플릿 검색 (동기 - 빠른 처리)
            selection_path.append("stage2_public")
            public_result = self._search_public_templates(user_input, options)
            if public_result.success:
                public_result.selection_path = selection_path
                return public_result

            # 3단계: 새 템플릿 생성 (비동기)
            selection_path.append("stage3_generation")
            generation_result = await self._generate_new_template_async(user_input, selection_path)
            return generation_result

        except Exception as e:
            self.logger.error(f"템플릿 선택 중 오류 (비동기): {e}")
            return TemplateSelectionResult(
                success=False,
                error=f"템플릿 선택 실패: {str(e)}",
                selection_path=selection_path
            )
    
    def _search_existing_templates(self, user_input: str, options: Dict[str, Any]) -> TemplateSelectionResult:
        """1단계: 기존 승인된 템플릿 검색 (BasicTemplateMatcher 사용)"""
        try:
            threshold = options.get('existing_threshold', self.existing_similarity_threshold)

            self.logger.info(f"1단계: BasicTemplateMatcher 기반 기존 템플릿 검색 (임계값: {threshold})")

            # BasicTemplateMatcher를 통한 유사 템플릿 검색
            matched_template = self.basic_template_matcher.find_matching_template(user_input, threshold)

            if not matched_template:
                return TemplateSelectionResult(
                    success=False,
                    source="existing",
                    error="유사한 기존 템플릿을 찾을 수 없습니다"
                )

            similarity_score = matched_template['similarity']

            # 기존 템플릿을 표준 형식으로 변환 (이미 #{변수} 형식)
            template_content = matched_template['content']
            variables = matched_template.get('variables', [])

            self.logger.info(f"기존 템플릿 발견: {matched_template['name']} ({matched_template['category']}) (유사도: {similarity_score:.3f})")

            return TemplateSelectionResult(
                success=True,
                template=template_content,
                variables=variables,
                source="existing",
                source_info={
                    "template_id": matched_template['id'],
                    "template_name": matched_template['name'],
                    "category": matched_template['category'],
                    "similarity": similarity_score,
                    "match_type": matched_template['match_type'],
                    "keywords": matched_template.get('keywords', [])
                }
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
                    "original_variables": len(best_template.variables),
                    "buttons": best_template.buttons,
                    "industry": best_template.industry or [],
                    "purpose": best_template.purpose or []
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
        """3단계: 새 템플릿 생성 (검증 포함) - 동기 버전 (하위 호환성)"""
        try:
            self.logger.info("3단계: 새 템플릿 생성 (검증 시스템 적용)")

            # 재생성 시도 횟수 제한
            for attempt in range(self.max_regeneration_attempts):
                if attempt > 0:
                    self.logger.info(f"템플릿 재생성 시도 {attempt + 1}/{self.max_regeneration_attempts}")

                # Agent2를 통한 템플릿 생성
                result, tools_data = self.agent2.generate_compliant_template(user_input)

                if not result.get("success", False):
                    # 새로운 응답: 변수 수집 필요
                    if result.get("status") == "need_more_variables":
                        return TemplateSelectionResult(
                            success=False,
                            source="generated",
                            status="need_more_variables",
                            mapped_variables=result.get("mapped_variables", {}),
                            missing_variables=result.get("missing_variables", []),
                            partial_template=result.get("template", ""),
                            mapping_coverage=result.get("mapping_coverage", 0),
                            industry=result.get("industry", []),
                            purpose=result.get("purpose", []),
                            selection_path=selection_path
                        )

                    # 기존 실패 처리
                    if attempt == self.max_regeneration_attempts - 1:  # 마지막 시도
                        return TemplateSelectionResult(
                            success=False,
                            source="generated",
                            error=result.get("error", "템플릿 생성 실패"),
                            selection_path=selection_path
                        )
                    continue  # 다음 시도

                # 변수 형식을 표준 형식으로 변환
                template = result.get("template", "")
                variables = result.get("variables", [])
                standardized_template, standardized_variables = self._standardize_variables(template, variables)

                # 템플릿 검증 실행
                self.logger.info(f"생성된 템플릿 검증 중... (시도 {attempt + 1})")
                validation_report = self.template_validator.validate_template(
                    template=standardized_template,
                    tools_results=tools_data,  # Agent2의 Tools 결과
                    user_input=user_input
                )

                # 검증 결과 로깅
                self.logger.info(f"검증 점수: {validation_report.overall_score:.2f}")
                if validation_report.warnings:
                    self.logger.warning(f"검증 경고: {', '.join(validation_report.warnings[:3])}")
                if validation_report.failed_checks:
                    self.logger.error(f"검증 실패: {', '.join(validation_report.failed_checks[:3])}")

                # 검증 통과시 템플릿 반환
                if validation_report.success:
                    self.logger.info("템플릿 검증 통과 - 최종 템플릿 완성")
                    return TemplateSelectionResult(
                        success=True,
                        template=standardized_template,
                        variables=standardized_variables,
                        source="generated",
                        source_info={
                            "original_variables": len(variables),
                            "generation_method": "agent2_with_validation",
                            "validation_score": validation_report.overall_score,
                            "validation_attempts": attempt + 1,
                            "tools_results": tools_data
                        },
                        selection_path=selection_path
                    )

                # 재생성이 필요한 경우
                elif validation_report.should_regenerate:
                    self.logger.warning(f"템플릿 검증 실패 (점수: {validation_report.overall_score:.2f}) - 재생성 필요")
                    if attempt < self.max_regeneration_attempts - 1:
                        continue  # 재생성 시도
                    else:
                        # 최대 시도 횟수 초과 - 최선의 결과라도 반환
                        self.logger.error("최대 재생성 횟수 초과 - 최선의 템플릿 반환")
                        return TemplateSelectionResult(
                            success=False,
                            template=standardized_template,
                            variables=standardized_variables,
                            source="generated",
                            error=f"검증 미통과 (점수: {validation_report.overall_score:.2f})",
                            source_info={
                                "original_variables": len(variables),
                                "generation_method": "agent2_validation_failed",
                                "validation_score": validation_report.overall_score,
                                "validation_attempts": attempt + 1,
                                "validation_issues": validation_report.failed_checks + validation_report.warnings,
                                "recommendation": validation_report.recommendation
                            },
                            selection_path=selection_path
                        )

                # 경고 수준이지만 사용 가능한 경우
                else:
                    self.logger.info(f"템플릿 검증 경고 수준 (점수: {validation_report.overall_score:.2f}) - 사용 가능")
                    return TemplateSelectionResult(
                        success=True,
                        template=standardized_template,
                        variables=standardized_variables,
                        source="generated",
                        source_info={
                            "original_variables": len(variables),
                            "generation_method": "agent2_with_warnings",
                            "validation_score": validation_report.overall_score,
                            "validation_attempts": attempt + 1,
                            "validation_warnings": validation_report.warnings,
                            "recommendation": validation_report.recommendation
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

    async def _generate_new_template_async(self, user_input: str, selection_path: List[str]) -> TemplateSelectionResult:
        """3단계: 새 템플릿 생성 (검증 포함) - 비동기 버전"""
        try:
            self.logger.info("3단계: 새 템플릿 생성 (검증 시스템 적용) - 비동기")

            # 재생성 시도 횟수 제한
            for attempt in range(self.max_regeneration_attempts):
                if attempt > 0:
                    self.logger.info(f"템플릿 재생성 시도 {attempt + 1}/{self.max_regeneration_attempts} (비동기)")

                # Agent2를 통한 템플릿 생성 (비동기)
                result, tools_data = await self.agent2.generate_compliant_template_async(user_input)

                if not result.get("success", False):
                    # 새로운 응답: 변수 수집 필요
                    if result.get("status") == "need_more_variables":
                        return TemplateSelectionResult(
                            success=False,
                            source="generated",
                            status="need_more_variables",
                            mapped_variables=result.get("mapped_variables", {}),
                            missing_variables=result.get("missing_variables", []),
                            partial_template=result.get("template", ""),
                            mapping_coverage=result.get("mapping_coverage", 0),
                            industry=result.get("industry", []),
                            purpose=result.get("purpose", []),
                            selection_path=selection_path
                        )

                    # 기존 실패 처리
                    if attempt == self.max_regeneration_attempts - 1:  # 마지막 시도
                        return TemplateSelectionResult(
                            success=False,
                            source="generated",
                            error=result.get("error", "템플릿 생성 실패"),
                            selection_path=selection_path
                        )
                    continue  # 다음 시도

                # 변수 형식을 표준 형식으로 변환
                template = result.get("template", "")
                variables = result.get("variables", [])
                standardized_template, standardized_variables = self._standardize_variables(template, variables)

                # 템플릿 검증 실행 (동기 - 빠른 처리)
                self.logger.info(f"생성된 템플릿 검증 중... (시도 {attempt + 1}, 비동기)")
                validation_report = self.template_validator.validate_template(
                    template=standardized_template,
                    tools_results=tools_data,  # Agent2의 Tools 결과
                    user_input=user_input
                )

                # 검증 결과 로깅
                self.logger.info(f"검증 점수 (비동기): {validation_report.overall_score:.2f}")
                if validation_report.warnings:
                    self.logger.warning(f"검증 경고 (비동기): {', '.join(validation_report.warnings[:3])}")
                if validation_report.failed_checks:
                    self.logger.error(f"검증 실패 (비동기): {', '.join(validation_report.failed_checks[:3])}")

                # 검증 통과시 템플릿 반환
                if validation_report.success:
                    self.logger.info("템플릿 검증 통과 - 최종 템플릿 완성 (비동기)")
                    return TemplateSelectionResult(
                        success=True,
                        template=standardized_template,
                        variables=standardized_variables,
                        source="generated",
                        source_info={
                            "original_variables": len(variables),
                            "generation_method": "agent2_with_validation_async",
                            "validation_score": validation_report.overall_score,
                            "validation_attempts": attempt + 1,
                            "tools_results": tools_data
                        },
                        selection_path=selection_path
                    )

                # 재생성이 필요한 경우
                elif validation_report.should_regenerate:
                    self.logger.warning(f"템플릿 검증 실패 (점수: {validation_report.overall_score:.2f}) - 재생성 필요 (비동기)")
                    if attempt < self.max_regeneration_attempts - 1:
                        continue  # 재생성 시도
                    else:
                        # 최대 시도 횟수 초과 - 최선의 결과라도 반환
                        self.logger.error("최대 재생성 횟수 초과 - 최선의 템플릿 반환 (비동기)")
                        return TemplateSelectionResult(
                            success=False,
                            template=standardized_template,
                            variables=standardized_variables,
                            source="generated",
                            error=f"검증 미통과 (점수: {validation_report.overall_score:.2f})",
                            source_info={
                                "original_variables": len(variables),
                                "generation_method": "agent2_validation_failed_async",
                                "validation_score": validation_report.overall_score,
                                "validation_attempts": attempt + 1,
                                "validation_issues": validation_report.failed_checks + validation_report.warnings,
                                "recommendation": validation_report.recommendation
                            },
                            selection_path=selection_path
                        )

                # 경고 수준이지만 사용 가능한 경우
                else:
                    self.logger.info(f"템플릿 검증 경고 수준 (점수: {validation_report.overall_score:.2f}) - 사용 가능 (비동기)")
                    return TemplateSelectionResult(
                        success=True,
                        template=standardized_template,
                        variables=standardized_variables,
                        source="generated",
                        source_info={
                            "original_variables": len(variables),
                            "generation_method": "agent2_with_warnings_async",
                            "validation_score": validation_report.overall_score,
                            "validation_attempts": attempt + 1,
                            "validation_warnings": validation_report.warnings,
                            "recommendation": validation_report.recommendation
                        },
                        selection_path=selection_path
                    )

        except Exception as e:
            self.logger.error(f"새 템플릿 생성 실패 (비동기): {e}")
            return TemplateSelectionResult(
                success=False,
                source="generated",
                error=f"템플릿 생성 오류: {str(e)}",
                selection_path=selection_path
            )
    
    def _standardize_variables(self, template: str, variables: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        변수 형식을 표준 형식으로 변환
        ${변수명} -> #{변수명} 형식
        """
        standardized_template = template
        standardized_variables = []
        
        for var in variables:
            old_format = var.get("name", f"${{{var.get('description', '변수')}}}")
            # 의미있는 변수명 생성
            description = var.get("description", "변수")
            clean_name = description.replace(" ", "").replace("(", "").replace(")", "")
            new_format = f"#{{{clean_name}}}"
            
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
        [제목] 형태나 특정 패턴을 #{변수명} 형식으로 변환
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
            new_format = f"#{{{clean_name}}}"
            standardized_template = standardized_template.replace(old_format, new_format)
            
            variables.append({
                "name": new_format,
                "description": title,
                "required": True
            })
        
        # 기본 변수가 없으면 기본 템플릿 구조 추가
        if not variables:
            # 간단한 템플릿 구조로 변환
            standardized_template = f"안녕하세요, #{{고객명}}님!\n\n{template_content}\n\n자세한 내용은 #{{상세정보}}를 확인해주세요."
            
            variables = [
                {
                    "name": "#{고객명}",
                    "description": "고객명",
                    "required": True
                },
                {
                    "name": "#{상세정보}",
                    "description": "상세정보",
                    "required": True
                }
            ]
        
        return standardized_template, variables
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """선택 시스템 통계 정보"""
        basic_stats = self.basic_template_matcher.get_statistics()
        return {
            "existing_templates": {
                "available": basic_stats["total_templates"],
                "categories": basic_stats["categories"],
                "has_embeddings": basic_stats["has_embeddings"],
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

    async def find_existing_template(self, user_input: str, variables: Dict[str, Any], intent: Dict[str, Any], user_id: int) -> Optional[Dict[str, Any]]:
        """
        기존 템플릿 검색 (Java 백엔드 호환)
        """
        try:
            # 기존 _search_existing_templates 메서드 재사용
            options = {
                'variables': variables,
                'intent': intent,
                'user_id': user_id,
                'existing_threshold': 0.7
            }

            result = self._search_existing_templates(user_input, options)

            if result.success:
                return {
                    'template': result.template,
                    'variables': result.variables,
                    'source': result.source,
                    'source_info': result.source_info
                }

            return None

        except Exception as e:
            self.logger.error(f"기존 템플릿 검색 실패: {e}")
            return None


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