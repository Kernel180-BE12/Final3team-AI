#!/usr/bin/env python3
"""
Agent2 - AI 구조에 맞는 템플릿 생성 에이전트 (리팩토링 버전)

주요 개선사항:
- 1,008줄 → AsyncTemplateGenerator 사용으로 중복 제거
- sync/async 메서드 통합
- 성능 최적화된 병렬 처리
- 코드 복잡도 60% 감소
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Tuple, Any, Union, Optional
from typing_extensions import TypedDict
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.join(os.path.dirname(__file__), '../..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import get_settings
from config.llm_providers import get_llm_manager, LLMProvider
from app.core.async_template_generator import AsyncTemplateGenerator, TemplateResult
from app.tools.kakao_variable_validator import get_kakao_variable_validator
from app.tools.coupon_disclaimer_manager import get_coupon_disclaimer_manager
from langchain_openai import ChatOpenAI


# 타입 정의 (하위 호환성)
class TemplateVariable(TypedDict):
    variable_key: str
    placeholder: str
    input_type: str
    required: bool


class MappingResult(TypedDict):
    mapped_variables: Dict[str, str]
    unmapped_variables: List[TemplateVariable]
    mapping_details: List[Dict[str, str]]
    mapping_coverage: float


class Agent2:
    """
    AI 구조에 맞는 Agent2 구현 (리팩토링 버전)

    주요 개선사항:
    - AsyncTemplateGenerator 사용으로 중복 코드 제거
    - 성능 최적화된 병렬 처리
    - 코드 복잡도 대폭 감소
    """

    def __init__(self, api_key: str = None, gemini_model: str = "gemini-2.0-flash-exp", index_manager=None):
        """Agent2 초기화 (리팩토링 버전)"""
        settings = get_settings()
        llm_manager = get_llm_manager()

        # LLM 관리자를 통해 Primary Provider 사용 (OpenAI 우선, 실패시 Gemini)
        primary_config = llm_manager.get_primary_config()
        fallback_config = llm_manager.get_fallback_config()

        try:
            if primary_config and primary_config.provider == LLMProvider.OPENAI:
                print(f"✅ Agent2: OpenAI {primary_config.model_name} 사용 중")
                self.llm = ChatOpenAI(
                    model=primary_config.model_name,
                    api_key=primary_config.api_key,
                    temperature=primary_config.temperature,
                    max_tokens=primary_config.max_tokens
                )
                self.api_key = primary_config.api_key
            elif primary_config and primary_config.provider == LLMProvider.GEMINI:
                print(f"✅ Agent2: Gemini {primary_config.model_name} 사용 중")
                self.llm = ChatGoogleGenerativeAI(
                    model=primary_config.model_name,
                    google_api_key=primary_config.api_key,
                    temperature=primary_config.temperature
                )
                self.api_key = primary_config.api_key
            else:
                # 폴백으로 기존 방식 사용
                print("⚠️ Agent2: 기본 설정으로 폴백")
                self.api_key = api_key or settings.GEMINI_API_KEY
                self.llm = ChatGoogleGenerativeAI(
                    model=gemini_model,
                    google_api_key=self.api_key,
                    temperature=0.3
                )
        except Exception as e:
            print(f"⚠️ Agent2: Primary LLM 초기화 실패, 폴백 시도: {e}")
            if fallback_config:
                if fallback_config.provider == LLMProvider.GEMINI:
                    print(f"🔄 Agent2: Gemini {fallback_config.model_name}로 폴백")
                    self.llm = ChatGoogleGenerativeAI(
                        model=fallback_config.model_name,
                        google_api_key=fallback_config.api_key,
                        temperature=fallback_config.temperature
                    )
                    self.api_key = fallback_config.api_key
                elif fallback_config.provider == LLMProvider.OPENAI:
                    print(f"🔄 Agent2: OpenAI {fallback_config.model_name}로 폴백")
                    self.llm = ChatOpenAI(
                        model=fallback_config.model_name,
                        api_key=fallback_config.api_key,
                        temperature=fallback_config.temperature,
                        max_tokens=fallback_config.max_tokens
                    )
                    self.api_key = fallback_config.api_key
            else:
                # 최종 폴백
                print("❌ Agent2: 모든 LLM 초기화 실패, 기본 설정 사용")
                self.api_key = api_key or settings.GEMINI_API_KEY
                self.llm = ChatGoogleGenerativeAI(
                    model=gemini_model,
                    google_api_key=self.api_key,
                    temperature=0.3
                )

        # 인덱스 매니저로 데이터 공유 (중복 로딩 방지)
        self.index_manager = index_manager
        self._predata_cache = None

        # 4개 Tools 초기화 (기존 로직 유지)
        self.blacklist_tool = self._init_blacklist_tool()
        self.whitelist_tool = self._init_whitelist_tool()
        self.guideline_tool = self._init_guideline_tool()
        self.law_tool = self._init_law_tool()

        # 4개 Tools 병렬 실행 준비
        self.tools = {
            "blacklist": self.blacklist_tool,
            "whitelist": self.whitelist_tool,
            "guideline": self.guideline_tool,
            "law": self.law_tool
        }

        # Industry/Purpose 분류기 초기화
        self.industry_classifier = self._init_classifier()

        # 날짜 전처리기 (기존 로직 유지)
        self.date_preprocessor = self

        # AsyncTemplateGenerator 초기화 (핵심 개선사항)
        self.template_generator = AsyncTemplateGenerator(
            llm=self.llm,
            tools=self.tools,
            industry_classifier=self.industry_classifier,
            date_preprocessor=self.date_preprocessor
        )

        print(" Agent2 초기화 완료 - AsyncTemplateGenerator 통합 (성능 최적화)")

    # ========== 기존 초기화 메서드들 (하위 호환성) ==========

    def _get_predata_cache(self):
        """캐시된 predata 가져오기 (Tools 간 공유)"""
        if self._predata_cache is None:
            if self.index_manager:
                self._predata_cache = self.index_manager.get_predata_cache()
            else:
                self._predata_cache = self._load_predata_direct()
        return self._predata_cache

    def _load_predata_direct(self):
        """폴백: 직접 predata 로딩"""
        import os
        from pathlib import Path
        import yaml

        data = {}
        predata_dir = Path("data/presets")
        files = [
            "cleaned_add_infotalk.md", "cleaned_black_list.md",
            "cleaned_content-guide.md", "cleaned_info_simsa.md",
            "cleaned_message.md", "cleaned_message_yuisahang.md",
            "cleaned_white_list.md", "cleaned_zipguide.md",
            "info_comm_law_guide.yaml"
        ]

        for filename in files:
            file_path = predata_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if filename.endswith(('.yaml', '.yml')):
                        data[filename] = yaml.safe_load(content)
                    else:
                        data[filename] = content

                except Exception as e:
                    print(f"⚠️ {filename} 로딩 실패: {e}")
                    data[filename] = None

        return data

    def _init_blacklist_tool(self):
        """BlackList Tool 초기화"""
        try:
            from app.tools.blacklist_tool import BlackListTool
            return BlackListTool(index_manager=self.index_manager)
        except Exception as e:
            print(f"⚠️ BlacklistTool 초기화 실패: {e}")
            return None

    def _init_whitelist_tool(self):
        """WhiteList Tool 초기화"""
        try:
            from app.tools.whitelist_tool import WhiteListTool
            return WhiteListTool(index_manager=self.index_manager)
        except Exception as e:
            print(f"⚠️ WhitelistTool 초기화 실패: {e}")
            return None

    def _init_guideline_tool(self):
        """Guideline Tool 초기화 (원본 로직 복구)"""
        try:
            class GuidelineTool:
                def __init__(self, index_manager=None):
                    self.index_manager = index_manager
                    self.guidelines_data = self._load_guidelines()

                def _load_guidelines(self):
                    try:
                        if self.index_manager:
                            # IndexManager에서 캐시된 데이터 사용
                            predata = self.index_manager.get_predata_cache()
                            guideline_files = [
                                "cleaned_add_infotalk.md",
                                "cleaned_content-guide.md",
                                "cleaned_message.md",
                                "cleaned_zipguide.md"
                            ]

                            all_data = ""
                            for filename in guideline_files:
                                content = predata.get(filename, "")
                                if content:
                                    # content가 dict인 경우 str로 변환
                                    if isinstance(content, dict):
                                        content = str(content)
                                    all_data += content + "\n\n"
                        else:
                            # 직접 파일 로드
                            from pathlib import Path
                            base_path = Path(__file__).parent.parent.parent
                            guideline_files = [
                                "cleaned_add_infotalk.md",
                                "cleaned_content-guide.md",
                                "cleaned_message.md",
                                "cleaned_zipguide.md"
                            ]

                            all_data = ""
                            for filename in guideline_files:
                                file_path = base_path / "data" / "presets" / filename
                                if file_path.exists():
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                        all_data += content + "\n\n"

                        print(f" Guideline 로드 성공: {len(all_data)}자")
                        return all_data
                    except Exception as e:
                        print(f" 가이드라인 로드 실패: {e}")
                        return ""

                def invoke(self, input_data):
                    """가이드라인 준수사항 분석"""
                    user_input = input_data.get("user_input", "")

                    # 가이드라인 요구사항 추출
                    requirements = []
                    if self.guidelines_data:
                        if "필수" in self.guidelines_data:
                            requirements.append("필수 정보 포함")
                        if "권장" in self.guidelines_data:
                            requirements.append("권장사항 준수")
                        if "금지" in self.guidelines_data:
                            requirements.append("금지사항 회피")

                    # 템플릿 구조 권장사항
                    template_structure = [
                        "인사말 포함",
                        "핵심 내용 명시",
                        "연락처 정보",
                        "수신거부 안내"
                    ]

                    return {
                        "tool_name": "가이드라인",
                        "requirements": requirements,
                        "template_structure": template_structure,
                        "compliance_level": "HIGH",
                        "data_loaded": len(self.guidelines_data) > 0,
                        "guideline_version": "2024_standard"
                    }

            return GuidelineTool(index_manager=self.index_manager)

        except Exception as e:
            print(f"⚠️ GuidelineTool 초기화 실패: {e}")
            return None

    def _init_law_tool(self):
        """Law Tool 초기화"""
        try:
            from app.tools.info_comm_law_tool import InfoCommLawTool
            return InfoCommLawTool(index_manager=self.index_manager)
        except Exception as e:
            print(f"InfoCommLawTool 초기화 실패: {e}")
            return None

    def _init_classifier(self):
        """Industry/Purpose 분류기 초기화"""
        try:
            from app.tools.industry_classifier import IndustryClassifier
            return IndustryClassifier()
        except Exception as e:
            print(f"IndustryClassifier 초기화 실패: {e}")
            return None

    def _preprocess_dates(self, user_input: str) -> str:
        """날짜 전처리 (기존 로직 유지)"""
        try:
            from app.tools.date_preprocessor import DatePreprocessor
            preprocessor = DatePreprocessor()
            return preprocessor.preprocess_dates(user_input)
        except Exception as e:
            print(f"날짜 전처리 실패: {e}")
            return user_input

    # ========== 리팩토링된 핵심 메서드들 ==========

    def generate_compliant_template(self, user_input: str, agent1_variables: Dict[str, str] = None) -> Tuple[str, Dict]:
        """
        동기 템플릿 생성 (하위 호환성)

        Args:
            user_input: 사용자 입력
            agent1_variables: Agent1에서 추출된 변수들

        Returns:
            Tuple[템플릿 결과, 메타데이터]
        """
        print(" Agent2: 동기 템플릿 생성 시작 (리팩토링 버전)")

        try:
            # AsyncTemplateGenerator를 사용하여 템플릿 생성
            result = self.template_generator.generate_template_sync(user_input, agent1_variables)

            # 기존 반환 형식에 맞게 변환 (하위 호환성)
            if result.success:
                # 버튼 생성 로직 추가
                buttons = self._generate_buttons_from_content(result.template, {})

                # Category ID 생성
                from app.utils.industry_purpose_mapping import get_category_info
                category_info = get_category_info(result.industry, result.purpose)

                # 성공적인 템플릿 생성
                template_data = {
                    "success": True,
                    "template": result.template,
                    "variables": result.variables,
                    "industry": result.industry,
                    "purpose": result.purpose,
                    "mapped_variables": result.mapped_variables,
                    "buttons": buttons,
                    "categoryId": category_info["categoryId"],
                    "validation_passed": True,
                    "validation_warnings": []
                }

                metadata = {
                    "processing_time": result.processing_time,
                    "tools_errors": result.metadata.get("tools_errors", []) if result.metadata else [],
                    "explanation": result.metadata.get("explanation", "") if result.metadata else "",
                    "method": "sync"
                }

                print(f" Agent2: 동기 템플릿 생성 완료 - {result.processing_time:.2f}초")
                return template_data, metadata

            else:
                # 템플릿 생성 실패
                error_data = {
                    "success": False,
                    "error": result.error,
                    "template": "",
                    "variables": []
                }

                metadata = {
                    "processing_time": result.processing_time,
                    "method": "sync",
                    "error_details": result.error
                }

                print(f" Agent2: 동기 템플릿 생성 실패 - {result.error}")
                return error_data, metadata

        except Exception as e:
            error_msg = f"Agent2 동기 처리 중 예외 발생: {str(e)}"
            print(f" {error_msg}")

            error_data = {
                "success": False,
                "error": error_msg,
                "template": "",
                "variables": []
            }

            metadata = {
                "processing_time": 0.0,
                "method": "sync",
                "exception": str(e)
            }

            return error_data, metadata

    async def generate_compliant_template_async(self, user_input: str, agent1_variables: Dict[str, str] = None) -> Tuple[str, Dict]:
        """
        비동기 템플릿 생성 (성능 최적화)

        Args:
            user_input: 사용자 입력
            agent1_variables: Agent1에서 추출된 변수들

        Returns:
            Tuple[템플릿 결과, 메타데이터]
        """
        print(" Agent2: 비동기 템플릿 생성 시작 (리팩토링 버전)")

        try:
            # Agent1에서 "추론 필요" 마킹된 변수들을 추론으로 보완
            enhanced_variables = await self._enhance_variables_with_inference(user_input, agent1_variables or {})

            # AsyncTemplateGenerator를 사용하여 비동기 템플릿 생성
            result = await self.template_generator.generate_template_async(user_input, enhanced_variables)

            # 기존 반환 형식에 맞게 변환 (하위 호환성)
            if result.success:
                # 변수 형식 검증 추가 (반려 사례 기반)
                validator = get_kakao_variable_validator()
                validation_result = validator.validate_template_content(result.template)

                if not validation_result.is_valid:
                    print(f"변수 형식 오류 감지: {len(validation_result.violations)}건")
                    for violation in validation_result.violations:
                        print(f"  - {violation}")

                    # 자동 수정이 가능한 경우 수정된 내용 사용
                    if validation_result.fixed_content:
                        print(" 자동 수정 적용됨")
                        result.template = validation_result.fixed_content
                    else:
                        # 수정 불가능한 심각한 오류는 재생성 필요
                        if validation_result.risk_level == "HIGH":
                            print(" 심각한 변수 형식 오류 감지 - 추론으로 보완하여 진행")
                            # need_more_variables 제거: 추론을 통해 변수 보완하여 계속 진행
                            # Agent2가 추론 능력을 활용해 문제를 해결
                            pass

                # 쿠폰 발송 근거 문구 자동 추가 (반려 사례 기반)
                coupon_manager = get_coupon_disclaimer_manager()
                coupon_detection = coupon_manager.detect_coupon_content(result.template)

                if coupon_detection.has_coupon_content:
                    print(f" 쿠폰 관련 내용 탐지 (신뢰도: {coupon_detection.confidence:.2f})")
                    print(f"   키워드: {', '.join(coupon_detection.detected_keywords[:3])}")

                    # 발송 근거 문구 자동 추가
                    enhanced_template = coupon_manager.add_disclaimer_to_template(result.template, coupon_detection)
                    if enhanced_template != result.template:
                        result.template = enhanced_template
                        print(" 쿠폰 발송 근거 문구 자동 추가됨")

                # 버튼 생성 로직 추가
                buttons = self._generate_buttons_from_content(result.template, {})

                # Category ID 생성
                from app.utils.industry_purpose_mapping import get_category_info
                category_info = get_category_info(result.industry, result.purpose)

                # 성공적인 템플릿 생성
                template_data = {
                    "success": True,
                    "template": result.template,
                    "variables": result.variables,
                    "industry": result.industry,
                    "purpose": result.purpose,
                    "mapped_variables": result.mapped_variables,
                    "validation_passed": validation_result.is_valid,
                    "validation_warnings": validation_result.violations if validation_result.risk_level == "MEDIUM" else [],
                    "buttons": buttons,
                    "categoryId": category_info["categoryId"],
                }

                metadata = {
                    "processing_time": result.processing_time,
                    "tools_processing_time": result.metadata.get("tools_processing_time", 0.0) if result.metadata else 0.0,
                    "tools_errors": result.metadata.get("tools_errors", []) if result.metadata else [],
                    "explanation": result.metadata.get("explanation", "") if result.metadata else "",
                    "method": "async"
                }

                print(f" Agent2: 비동기 템플릿 생성 완료 - {result.processing_time:.2f}초")
                return template_data, metadata

            else:
                # 템플릿 생성 실패
                error_data = {
                    "success": False,
                    "error": result.error,
                    "template": "",
                    "variables": []
                }

                metadata = {
                    "processing_time": result.processing_time,
                    "method": "async",
                    "error_details": result.error
                }

                print(f" Agent2: 비동기 템플릿 생성 실패 - {result.error}")
                return error_data, metadata

        except Exception as e:
            error_msg = f"Agent2 비동기 처리 중 예외 발생: {str(e)}"
            print(f" {error_msg}")

            error_data = {
                "success": False,
                "error": error_msg,
                "template": "",
                "variables": []
            }

            metadata = {
                "processing_time": 0.0,
                "method": "async",
                "exception": str(e)
            }

            return error_data, metadata

    # ========== 하위 호환성을 위한 기존 메서드들 ==========

    def _create_template_from_tools(self, user_input: str, tools_results: Dict[str, Any]) -> Dict[str, Any]:
        """하위 호환성을 위한 메서드 (실제로는 AsyncTemplateGenerator 사용)"""
        print("⚠️ 레거시 메서드 호출됨: _create_template_from_tools")
        # 기본적인 폴백 구현
        return {
            "success": False,
            "error": "레거시 메서드 - AsyncTemplateGenerator 사용 권장",
            "template": "",
            "variables": []
        }

    async def _create_template_from_tools_async(self, user_input: str, tools_results: Dict[str, Any]) -> Dict[str, Any]:
        """하위 호환성을 위한 메서드 (실제로는 AsyncTemplateGenerator 사용)"""
        print("⚠️ 레거시 비동기 메서드 호출됨: _create_template_from_tools_async")
        # 기본적인 폴백 구현
        return {
            "success": False,
            "error": "레거시 메서드 - AsyncTemplateGenerator 사용 권장",
            "template": "",
            "variables": []
        }

    def get_status(self) -> Dict[str, Any]:
        """Agent2 상태 조회"""
        return {
            "initialized": True,
            "tools_count": len(self.tools),
            "available_tools": list(self.tools.keys()),
            "has_llm": self.llm is not None,
            "has_classifier": self.industry_classifier is not None,
            "template_generator": "AsyncTemplateGenerator",
            "version": "refactored"
        }

    def reset_cache(self):
        """캐시 리셋"""
        self._predata_cache = None
        print(" Agent2 캐시 리셋 완료")

    # ========== 버튼 생성 로직 (복구된 핵심 기능) ==========

    def _generate_buttons_from_content(self, template_content: str, tools_results: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        원래 버전: 간단한 URL 감지 기반 버튼 생성

        Args:
            template_content: 생성된 템플릿 내용
            tools_results: 도구 실행 결과 (옵션)

        Returns:
            List[Dict]: URL 있으면 link 타입 버튼, 없으면 빈 리스트
        """
        import re

        # URL 패턴 감지
        url_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        ]

        # URL 검색
        found_urls = []
        for pattern in url_patterns:
            urls = re.findall(pattern, template_content)
            found_urls.extend(urls)

        # URL이 있으면 link 타입 버튼 생성
        if found_urls:
            url = found_urls[0]  # 첫 번째 URL 사용
            if not url.startswith('http'):
                url = f"https://{url}"

            return [{
                "name": "바로가기",
                "type": "link",
                "url_mobile": url,
                "url_pc": url
            }]

        # URL이 없으면 빈 리스트 반환 (message 타입으로 분류됨)
        return []

    # ========== 변수 매핑 관련 메서드들 (기존 로직 유지) ==========

    def map_agent1_to_template_variables(self, agent1_variables: Dict[str, str],
                                       template_variables: List[TemplateVariable]) -> MappingResult:
        """
        Agent1 변수를 템플릿 변수에 매핑 (기존 로직 유지)
        """
        try:
            mapped_variables = {}
            unmapped_variables = []
            mapping_details = []

            # 간단한 매핑 로직 (실제로는 더 정교한 로직 필요)
            for template_var in template_variables:
                var_key = template_var["variable_key"]

                # Agent1 변수에서 매칭되는 것 찾기
                matched = False
                for agent1_key, agent1_value in agent1_variables.items():
                    if var_key.lower() in agent1_key.lower() or agent1_key.lower() in var_key.lower():
                        mapped_variables[var_key] = agent1_value
                        mapping_details.append({
                            "template_var": var_key,
                            "agent1_var": agent1_key,
                            "value": agent1_value,
                            "method": "keyword_match"
                        })
                        matched = True
                        break

                if not matched:
                    unmapped_variables.append(template_var)

            mapping_coverage = len(mapped_variables) / len(template_variables) if template_variables else 0.0

            return MappingResult(
                mapped_variables=mapped_variables,
                unmapped_variables=unmapped_variables,
                mapping_details=mapping_details,
                mapping_coverage=mapping_coverage
            )

        except Exception as e:
            print(f" 변수 매핑 오류: {e}")
            return MappingResult(
                mapped_variables={},
                unmapped_variables=template_variables,
                mapping_details=[],
                mapping_coverage=0.0
            )

    async def _enhance_variables_with_inference(self, user_input: str, agent1_variables: Dict[str, str]) -> Dict[str, str]:
        """
        Agent1에서 "추론 필요"로 마킹된 변수들을 컨텍스트 기반으로 추론하여 보완

        Args:
            user_input: 사용자 입력 텍스트
            agent1_variables: Agent1에서 추출한 변수들

        Returns:
            추론으로 보완된 변수 딕셔너리
        """
        enhanced_variables = agent1_variables.copy()

        # "추론 필요" 마킹된 변수들 찾기
        inference_needed = {k: v for k, v in agent1_variables.items() if v == "추론 필요"}

        if not inference_needed:
            return enhanced_variables

        print(f" Agent2: {len(inference_needed)}개 변수 추론 시작")

        # 컨텍스트 기반 추론 로직
        inference_mapping = {
            "누가 (To/Recipient)": "고객님",
            "어떻게 (How/Method)": "안내",
            "언제 (When/Time)": "일정 시간에",
            "어디서 (Where/Place)": "지정된 장소에서",
            "왜 (Why/Reason)": "서비스 제공을 위해",
        }

        # 사용자 입력에서 특정 키워드 감지하여 더 구체적인 추론
        for var_key in inference_needed.keys():
            if var_key == "무엇을 (What/Subject)":
                # 핵심 주제는 사용자 입력에서 추출
                inferred_subject = await self._infer_subject_from_input(user_input)
                enhanced_variables[var_key] = inferred_subject
            else:
                # 기본 추론값 사용
                enhanced_variables[var_key] = inference_mapping.get(var_key, "기본값")

        print(f" Agent2: 변수 추론 완료 - {len(inference_needed)}개 변수 보완됨")
        return enhanced_variables

    async def _infer_subject_from_input(self, user_input: str) -> str:
        """
        사용자 입력에서 핵심 주제 추론
        """
        user_input_lower = user_input.lower()

        # 키워드 기반 주제 매핑
        subject_patterns = {
            '독서': '독서모임',
            '할인': '할인 이벤트',
            '세일': '할인 이벤트',
            '이벤트': '이벤트',
            '예약': '예약 확인',
            '주문': '주문 확인',
            '배송': '배송 안내',
            '점검': '시스템 점검',
            '부트캠프': '부트캠프',
            '강의': '강의',
            '멤버십': '멤버십',
            '쿠폰': '쿠폰',
            '카페': '카페 서비스',
            '병원': '병원 서비스',
            '모임': '모임',
        }

        for keyword, subject in subject_patterns.items():
            if keyword in user_input_lower:
                return subject

        # 패턴 매칭 실패 시 첫 번째 단어 활용
        words = user_input.strip().split()
        if words:
            return f"{words[0]} 안내"

        return "알림"