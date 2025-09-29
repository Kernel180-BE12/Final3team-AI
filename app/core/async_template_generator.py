#!/usr/bin/env python3
"""
Async Template Generator - Agent2의 중복 코드를 제거하는 베이스 클래스
동기/비동기 템플릿 생성 로직을 통합하고 성능을 최적화
"""

import asyncio
import time
import concurrent.futures
from typing import Dict, List, Tuple, Any, Optional
from typing_extensions import TypedDict
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass
class TemplateResult:
    """템플릿 생성 결과를 담는 데이터 클래스"""
    success: bool
    template: Optional[str] = None
    variables: Optional[List[Dict[str, Any]]] = None
    industry: Optional[List[Dict[str, Any]]] = None
    purpose: Optional[List[Dict[str, Any]]] = None
    _mapped_variables: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0

    @property
    def mapped_variables(self) -> Optional[Dict[str, str]]:
        """mapped_variables 속성에 대한 접근자"""
        return self._mapped_variables


@dataclass
class ToolsResult:
    """4개 도구 실행 결과를 담는 데이터 클래스"""
    blacklist: Dict[str, Any]
    whitelist: Dict[str, Any]
    guideline: Dict[str, Any]
    law: Dict[str, Any]
    processing_time: float
    errors: List[str]


class AsyncTemplateGenerator:
    """
    Agent2의 중복 코드를 제거하는 베이스 클래스
    동기/비동기 템플릿 생성 로직을 통합
    """

    def __init__(self, llm: ChatGoogleGenerativeAI, tools: Dict[str, Any],
                 industry_classifier: Any = None, date_preprocessor: Any = None):
        """
        초기화

        Args:
            llm: LangChain LLM 인스턴스
            tools: 4개 도구 딕셔너리 (blacklist, whitelist, guideline, law)
            industry_classifier: 업종 분류기
            date_preprocessor: 날짜 전처리기
        """
        self.llm = llm
        self.tools = tools
        self.industry_classifier = industry_classifier
        self.date_preprocessor = date_preprocessor

    def preprocess_input(self, user_input: str) -> str:
        """입력 전처리 (날짜 변환 등)"""
        if self.date_preprocessor:
            try:
                preprocessed = self.date_preprocessor._preprocess_dates(user_input)
                if preprocessed != user_input:
                    print(f"날짜 전처리: '{user_input}' → '{preprocessed}'")
                return preprocessed
            except Exception as e:
                print(f"날짜 전처리 오류: {e}")

        return user_input

    def run_tools_sync(self, input_data: Dict[str, str]) -> ToolsResult:
        """
        4개 도구 동기 병렬 실행
        """
        start_time = time.time()
        print("4개 Tools 병렬 실행 시작 (동기)")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # 모든 도구를 병렬로 실행 (None인 tool 제외)
                future_to_tool = {
                    executor.submit(tool.invoke, input_data): tool_name
                    for tool_name, tool in self.tools.items()
                    if tool is not None
                }

                tools_results = {}
                errors = []

                for future in concurrent.futures.as_completed(future_to_tool):
                    tool_name = future_to_tool[future]
                    try:
                        result = future.result()
                        tools_results[tool_name] = result
                        print(f"{tool_name} 완료")
                    except Exception as exc:
                        error_msg = f"{tool_name} 오류: {exc}"
                        print(f"{error_msg}")
                        tools_results[tool_name] = {"error": str(exc)}
                        errors.append(error_msg)

            processing_time = time.time() - start_time
            print(f"4개 Tools 병렬 분석 완료 - {processing_time:.2f}초")

            return ToolsResult(
                blacklist=tools_results.get("blacklist", {}),
                whitelist=tools_results.get("whitelist", {}),
                guideline=tools_results.get("guideline", {}),
                law=tools_results.get("law", {}),
                processing_time=processing_time,
                errors=errors
            )

        except Exception as e:
            error_msg = f"Tools 실행 오류: {str(e)}"
            print(f" {error_msg}")
            return ToolsResult(
                blacklist={"error": error_msg},
                whitelist={"error": error_msg},
                guideline={"error": error_msg},
                law={"error": error_msg},
                processing_time=time.time() - start_time,
                errors=[error_msg]
            )

    async def run_tools_async(self, input_data: Dict[str, str]) -> ToolsResult:
        """
        4개 도구 비동기 병렬 실행
        """
        start_time = time.time()
        print(" 4개 Tools 비동기 병렬 실행 시작")

        try:
            async def run_tool(tool_name: str, tool: Any, input_data: Dict[str, str]):
                try:
                    result = await asyncio.to_thread(tool.invoke, input_data)
                    print(f" {tool_name} 비동기 완료")
                    return tool_name, result
                except Exception as e:
                    error_msg = f"{tool_name} 비동기 오류: {e}"
                    print(f" {error_msg}")
                    return tool_name, {"error": str(e)}

            # 모든 도구를 병렬로 실행 (None인 tool 제외)
            tasks = [
                run_tool(tool_name, tool, input_data)
                for tool_name, tool in self.tools.items()
                if tool is not None
            ]
            results = await asyncio.gather(*tasks)

            # 결과를 구조화
            tools_results = {tool_name: result for tool_name, result in results}
            errors = [
                f"{tool_name}: {result.get('error', '')}"
                for tool_name, result in results
                if isinstance(result, dict) and result.get('error')
            ]

            processing_time = time.time() - start_time
            print(f" 4개 Tools 비동기 병렬 분석 완료 - {processing_time:.2f}초")

            return ToolsResult(
                blacklist=tools_results.get("blacklist", {}),
                whitelist=tools_results.get("whitelist", {}),
                guideline=tools_results.get("guideline", {}),
                law=tools_results.get("law", {}),
                processing_time=processing_time,
                errors=errors
            )

        except Exception as e:
            error_msg = f"Tools 실행 오류 (비동기): {str(e)}"
            print(f" {error_msg}")
            return ToolsResult(
                blacklist={"error": error_msg},
                whitelist={"error": error_msg},
                guideline={"error": error_msg},
                law={"error": error_msg},
                processing_time=time.time() - start_time,
                errors=[error_msg]
            )

    def create_template_from_tools_sync(self, user_input: str, tools_result: ToolsResult) -> Dict[str, Any]:
        """
        도구 결과를 기반으로 템플릿 생성 (동기)
        """
        try:
            # 도구 결과를 종합하여 프롬프트 생성
            tools_summary = self._summarize_tools_results(tools_result)

            # LLM을 사용하여 템플릿 생성
            template_prompt = self._build_template_prompt(user_input, tools_summary)

            print(" AI 템플릿 생성 시작 (동기)")
            response = self.llm.invoke(template_prompt)

            # 응답 파싱
            parsed_result = self._parse_template_response(response.content)
            print(" AI 템플릿 생성 완료 (동기)")

            return parsed_result

        except Exception as e:
            error_msg = f"템플릿 생성 오류 (동기): {str(e)}"
            print(f" {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "template": "",
                "variables": []
            }

    async def create_template_from_tools_async(self, user_input: str, tools_result: ToolsResult) -> Dict[str, Any]:
        """
        도구 결과를 기반으로 템플릿 생성 (비동기)
        """
        try:
            # 도구 결과를 종합하여 프롬프트 생성
            tools_summary = self._summarize_tools_results(tools_result)

            # LLM을 사용하여 템플릿 생성
            template_prompt = self._build_template_prompt(user_input, tools_summary)

            print(" AI 템플릿 생성 시작 (비동기)")
            response = await asyncio.to_thread(self.llm.invoke, template_prompt)

            # 응답 파싱
            parsed_result = self._parse_template_response(response.content)
            print(" AI 템플릿 생성 완료 (비동기)")

            return parsed_result

        except Exception as e:
            error_msg = f"템플릿 생성 오류 (비동기): {str(e)}"
            print(f" {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "template": "",
                "variables": []
            }

    def classify_industry_purpose(self, user_input: str, agent1_variables: Optional[Dict[str, str]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        업종/목적 분류
        """
        if not self.industry_classifier:
            return [], []

        try:
            print(" Industry/Purpose 분류 시작")
            industry, purpose = self.industry_classifier.classify_industry_purpose(
                user_input, agent1_variables or {}
            )
            print(" Industry/Purpose 분류 완료")
            return industry, purpose
        except Exception as e:
            print(f" Industry/Purpose 분류 오류: {e}")
            return [], []

    def _summarize_tools_results(self, tools_result: ToolsResult) -> str:
        """도구 결과를 요약하여 프롬프트용 텍스트 생성"""
        summary_parts = []

        # 각 도구별 결과 요약
        for tool_name in ["blacklist", "whitelist", "guideline", "law"]:
            result = getattr(tools_result, tool_name, {})
            if isinstance(result, dict) and not result.get('error'):
                summary_parts.append(f" {tool_name}: 검증 통과")
            else:
                error = result.get('error', '알 수 없는 오류') if isinstance(result, dict) else '결과 없음'
                summary_parts.append(f" {tool_name}: {error}")

        # 처리 시간 추가
        summary_parts.append(f" 처리 시간: {tools_result.processing_time:.2f}초")

        return "\n".join(summary_parts)

    def _build_template_prompt(self, user_input: str, tools_summary: str) -> str:
        """템플릿 생성용 프롬프트 구성"""
        return f"""
카카오 알림톡 템플릿을 생성해주세요.

[사용자 요청]
{user_input}

[4개 도구 검증 결과]
{tools_summary}

[요구사항]
1. 카카오 알림톡 가이드라인 준수
2. 정보통신망법 준수
3. 비즈니스 목적에 적합한 내용
4. 변수는 #{{변수명}} 형식 사용
5. 200자 이내 권장

[알림톡 템플릿 구조 가이드]
1. 정중한 인사말로 시작
2. 핵심 정보를 명확하게 구조화 (▶ 기호 사용 권장)
3. 감사 인사로 마무리
4. 문의사항 안내 포함

[변수 사용 규칙]
- 형식: #{{변수명}} (한글 변수명 사용)
- 예시: #{{고객명}}, #{{예약일시}}, #{{예약인원}}, #{{카페연락처}}
- 링크나 버튼 관련 내용 포함 금지

[좋은 예시]
안녕하세요. #{{카페이름}} 예약 확인 안내입니다.

▶ 예약자명 : #{{고객명}}
▶ 예약일시 : #{{예약일시}}
▶ 예약인원 : #{{예약인원}}명

예약해 주셔서 감사합니다.
문의사항은 #{{카페연락처}}로 연락 주세요.

[출력 형식]
템플릿: [실제 템플릿 내용]
변수: [변수1, 변수2, ...]
설명: [생성 근거]
"""

    def _generate_buttons_from_content(self, template_content: str) -> List[Dict[str, str]]:
        """
        원래 버전: 간단한 URL 감지 기반 버튼 생성

        Args:
            template_content: 생성된 템플릿 내용

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

    def _parse_template_response(self, response_content: str) -> Dict[str, Any]:
        """LLM 응답을 파싱하여 구조화된 결과 반환"""
        try:
            # 간단한 파싱 로직 (실제로는 더 정교하게 구현 필요)
            lines = response_content.strip().split('\n')
            template = ""
            variables = []
            explanation = ""

            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('템플릿:'):
                    current_section = 'template'
                    template = line.replace('템플릿:', '').strip()
                elif line.startswith('변수:'):
                    current_section = 'variables'
                    var_text = line.replace('변수:', '').strip()
                    if var_text:
                        variables = [v.strip() for v in var_text.split(',') if v.strip()]
                elif line.startswith('설명:'):
                    current_section = 'explanation'
                    explanation = line.replace('설명:', '').strip()
                elif current_section == 'template' and line:
                    template += ' ' + line
                elif current_section == 'explanation' and line:
                    explanation += ' ' + line

            return {
                "success": True,
                "template": template.strip(),
                "variables": [{"variable_key": var, "placeholder": f"#{{{var}}}"} for var in variables],
                "explanation": explanation.strip(),
                "raw_response": response_content
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"응답 파싱 오류: {str(e)}",
                "template": "",
                "variables": [],
                "raw_response": response_content
            }

    async def generate_template_async(self, user_input: str, agent1_variables: Optional[Dict[str, str]] = None) -> TemplateResult:
        """
        비동기 템플릿 생성 (메인 진입점)
        """
        start_time = time.time()

        try:
            # 1. 입력 전처리
            preprocessed_input = self.preprocess_input(user_input)

            # 2. 4개 도구 병렬 실행
            input_data = {"user_input": preprocessed_input}
            tools_result = await self.run_tools_async(input_data)

            # 3. 템플릿 생성
            template_result = await self.create_template_from_tools_async(preprocessed_input, tools_result)

            # 4. 업종/목적 분류
            industry, purpose = self.classify_industry_purpose(preprocessed_input, agent1_variables)

            # 5. 버튼 생성
            buttons = self._generate_buttons_from_content(template_result.get("template", ""))

            # 6. 결과 통합
            total_time = time.time() - start_time

            return TemplateResult(
                success=template_result.get("success", False),
                template=template_result.get("template", ""),
                variables=template_result.get("variables", []),
                industry=industry,
                purpose=purpose,
                _mapped_variables=agent1_variables or {},
                error=template_result.get("error"),
                metadata={
                    "tools_processing_time": tools_result.processing_time,
                    "total_processing_time": total_time,
                    "tools_errors": tools_result.errors,
                    "explanation": template_result.get("explanation", ""),
                    "buttons": buttons
                },
                processing_time=total_time
            )

        except Exception as e:
            error_msg = f"템플릿 생성 실패: {str(e)}"
            print(f" {error_msg}")

            return TemplateResult(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )

    def generate_template_sync(self, user_input: str, agent1_variables: Optional[Dict[str, str]] = None) -> TemplateResult:
        """
        동기 템플릿 생성 (메인 진입점)
        """
        start_time = time.time()

        try:
            # 1. 입력 전처리
            preprocessed_input = self.preprocess_input(user_input)

            # 2. 4개 도구 병렬 실행
            input_data = {"user_input": preprocessed_input}
            tools_result = self.run_tools_sync(input_data)

            # 3. 템플릿 생성
            template_result = self.create_template_from_tools_sync(preprocessed_input, tools_result)

            # 4. 업종/목적 분류
            industry, purpose = self.classify_industry_purpose(preprocessed_input, agent1_variables)

            # 5. 버튼 생성
            buttons = self._generate_buttons_from_content(template_result.get("template", ""))

            # 6. 결과 통합
            total_time = time.time() - start_time

            return TemplateResult(
                success=template_result.get("success", False),
                template=template_result.get("template", ""),
                variables=template_result.get("variables", []),
                industry=industry,
                purpose=purpose,
                _mapped_variables=agent1_variables or {},
                error=template_result.get("error"),
                metadata={
                    "tools_processing_time": tools_result.processing_time,
                    "total_processing_time": total_time,
                    "tools_errors": tools_result.errors,
                    "explanation": template_result.get("explanation", ""),
                    "buttons": buttons
                },
                processing_time=total_time
            )

        except Exception as e:
            error_msg = f"템플릿 생성 실패: {str(e)}"
            print(f" {error_msg}")

            return TemplateResult(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )