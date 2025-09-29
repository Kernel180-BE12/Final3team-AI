"""
Template API Router
템플릿 생성 및 관리 API 엔드포인트
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Union

# 모듈 임포트
from app.agents.agent1 import Agent1
from app.agents.agent2 import Agent2
from app.core.template_selector import TemplateSelector
from app.utils.llm_provider_manager import get_llm_manager
from app.dto.api_result import ApiResult, ErrorResponse as ApiErrorResponse
from app.utils.language_detector import validate_input_language, ValidationError
from app.utils.industry_purpose_mapping import get_category_info
from app.utils.performance_logger import get_performance_logger, TimingContext

# LangGraph 통합
from app.core.langgraph_integration import (
    process_template_with_langgraph,
    is_langgraph_enabled
)

import time
import uuid

router = APIRouter()


class TemplateRequest(BaseModel):
    """템플릿 생성 요청 모델"""
    userId: int
    requestContent: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="구체적인 알림톡 템플릿 요청 내용 (최소 10자 이상)",
        examples=["카페 예약 확인 알림톡 템플릿을 만들어주세요"]
    )
    conversationContext: Optional[str] = Field(
        None,
        description="재질문 컨텍스트",
        examples=["이전 대화 내용"]
    )

    @validator('requestContent')
    @classmethod
    def validate_request_content(cls, v):
        """requestContent 유효성 검증"""
        if not v or v.strip() == "":
            raise ValueError("템플릿 요청 내용을 입력해주세요")

        # 언어별 검증을 먼저 실행 (영어 입력 감지)
        is_valid, error_type, message = validate_input_language(v)
        if not is_valid:
            if error_type == ValidationError.ENGLISH_ONLY:
                raise ValueError("Please enter in Korean. English-only input cannot generate KakaoTalk templates.")
            else:
                raise ValueError(message)

        # 한국어 기본값이나 의미없는 텍스트 필터링 (영어는 위에서 이미 처리됨)
        invalid_inputs = ["샘플", "테스트", "없음", "기본값"]
        if v.strip().lower() in invalid_inputs:
            raise ValueError("구체적인 템플릿 요청 내용을 입력해주세요")

        # 최소 단어 수 검증 (2단어 이상) - 언어 검증 통과 후에만 체크
        words = v.strip().split()
        if len(words) < 2:
            raise ValueError("더 구체적인 요청 내용을 입력해주세요 (최소 2단어 이상)")

        return v.strip()


class ErrorDetail(BaseModel):
    """에러 상세 정보 모델"""
    code: str
    message: str
    details: Optional[str] = None


class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    error: ErrorDetail
    timestamp: str


class Variable(BaseModel):
    """변수 정보 모델"""
    id: int
    variableKey: str = Field(..., alias='variableKey')
    placeholder: str
    inputType: str = Field(..., alias='inputType')
    value: str


class IndustryPurposeItem(BaseModel):
    """Industry/Purpose 아이템 모델 (ID + 이름)"""
    id: int
    name: str


class TemplateSuccessData(BaseModel):
    """Java AiTemplateResponse와 호환되는 데이터 구조"""
    id: Optional[int]  # 부분 완성 시 null
    userId: int
    categoryId: str
    title: str
    content: str
    imageUrl: Optional[str] = None
    type: str
    buttons: List[dict] = []
    variables: List[Variable]
    industries: List[IndustryPurposeItem] = []  # [{"id": 1, "name": "학원"}]
    purposes: List[IndustryPurposeItem] = []    # [{"id": 2, "name": "공지/안내"}]
    _mapped_variables: Dict[str, str] = {}  # FastAPI 전용 필드

class TemplateSuccessResponse(BaseModel):
    """ApiResult로 래핑된 템플릿 응답 모델"""
    data: Optional[TemplateSuccessData] = None
    message: Optional[str] = None
    error: Optional[ErrorResponse] = None


class IncompleteInfoDetails(BaseModel):
    """추가 정보 필요 상세 모델"""
    confirmed_variables: Optional[Dict[str, str]] = None
    missing_variables: Optional[List[str]] = None
    contextual_question: Optional[str] = None
    original_input: Optional[str] = None
    validation_status: Optional[str] = None
    reasoning: Optional[str] = None
    mapped_variables: Optional[Dict[str, str]] = None
    partial_template: Optional[str] = None
    mapping_coverage: Optional[float] = None
    industry: Optional[List[dict]] = None
    purpose: Optional[List[dict]] = None


class ErrorResponseWithDetails(BaseModel):
    """상세 에러 응답 모델"""
    detail: Dict[str, Any]


class PartialTemplateResponse(BaseModel):
    """부분 완성 템플릿 응답 모델"""
    status: str = "PARTIAL_COMPLETION"
    message: str
    data: Dict[str, Any]
    timestamp: str


def create_error_response(error_code: str, message: str, details: Any = None, status_code: int = 400) -> JSONResponse:
    """Java 호환 에러 응답 생성"""
    # ApiResult의 클래스 메소드를 사용하여 에러 응답 객체를 생성
    api_result = ApiResult.error(code=error_code, message=message)
    return JSONResponse(
        status_code=status_code,
        # Pydantic v1의 dict() 메소드를 사용하여 직렬화
        content=api_result.dict()
    )


def determine_template_type(buttons: List[dict] = None) -> str:
    """
    버튼 존재 여부에 따라 템플릿 타입 결정

    Args:
        buttons: 버튼 리스트

    Returns:
        "LINK" if buttons exist, else "MESSAGE"
    """
    if buttons and len(buttons) > 0:
        return "LINK"
    return "MESSAGE"


def convert_industry_purpose_data(industry_list: List[dict] = None, purpose_list: List[dict] = None) -> Dict[str, List]:
    """
    Industry/Purpose 데이터를 ID+이름 객체 배열 형식으로 변환

    Args:
        industry_list: Agent2에서 받은 industry 데이터 [{"id": 1, "name": "학원", "confidence": 0.9}, ...]
        purpose_list: Agent2에서 받은 purpose 데이터 [{"id": 2, "name": "공지/안내", "confidence": 0.8}, ...]

    Returns:
        {
            "industries": [{"id": 1, "name": "학원"}],
            "purposes": [{"id": 2, "name": "공지/안내"}]
        }
    """
    result = {
        "industries": [],
        "purposes": []
    }

    # Industry 처리
    for item in (industry_list or []):
        if isinstance(item, dict):
            if "id" in item and "name" in item:
                # Agent2에서 오는 정상적인 형식
                result["industries"].append({"id": item["id"], "name": item["name"]})
            elif "name" in item:
                # name만 있는 경우
                result["industries"].append({"id": 9, "name": item["name"]})  # 기타 ID
            else:
                # 기타 dict 형식
                name = str(item.get('name', item))
                result["industries"].append({"id": 9, "name": name})
        else:
            # 문자열인 경우
            name = str(item)
            result["industries"].append({"id": 9, "name": name})

    # Purpose 처리
    for item in (purpose_list or []):
        if isinstance(item, dict):
            if "id" in item and "name" in item:
                # Agent2에서 오는 정상적인 형식
                result["purposes"].append({"id": item["id"], "name": item["name"]})
            elif "name" in item:
                # name만 있는 경우
                result["purposes"].append({"id": 11, "name": item["name"]})  # 기타 ID
            else:
                # 기타 dict 형식
                name = str(item.get('name', item))
                result["purposes"].append({"id": 11, "name": name})
        else:
            # 문자열인 경우
            name = str(item)
            result["purposes"].append({"id": 11, "name": name})

    return result


def create_partial_response(user_id: int, partial_template: str, missing_variables: List[dict], mapped_variables: Dict[str, str], industry: List[dict], purpose: List[dict]) -> JSONResponse:
    """부분 완성 응답 생성 (202 상태코드) - Java 호환 구조"""
    # Industry/Purpose 데이터를 기존 및 새로운 형식 둘 다 생성
    converted_data = convert_industry_purpose_data(industry, purpose)

    # 동적 카테고리 결정
    category_info = get_category_info(converted_data["industries"], converted_data["purposes"])

    template_data = TemplateSuccessData(
        id=None,  # 부분 완성 상태 (아직 DB에 저장되지 않음)
        userId=user_id,
        categoryId=category_info["categoryId"],
        title=f"{category_info['title']} (부분 완성)",
        content=partial_template,
        imageUrl=None,
        type=determine_template_type([]),  # 부분완성은 버튼이 없으므로 MESSAGE
        buttons=[],
        variables=missing_variables,  # 누락된 변수들
        industries=converted_data["industries"],
        purposes=converted_data["purposes"],
        _mapped_variables=mapped_variables  # 이미 매핑된 변수들
    )

    # ApiResult로 래핑
    result = ApiResult.ok(template_data)
    return JSONResponse(
        status_code=202,
        content=result.model_dump()
    )



@router.post("/templates", tags=["Template Generation"],
            responses={
                200: {
                    "model": TemplateSuccessResponse,
                    "description": "템플릿 생성 완료"
                },
                202: {
                    "model": TemplateSuccessResponse,
                    "description": "부분 완성 - 추가 정보 필요 (200과 동일한 구조)"
                },
                400: {"model": ErrorResponseWithDetails},
                500: {"model": ErrorResponseWithDetails}
            })
async def create_template(request: TemplateRequest):
    """
    AI 기반 알림톡 템플릿 생성

    Args:
        request: 템플릿 생성 요청 (userId, requestContent)

    Returns:
        생성된 템플릿 정보 또는 에러 응답
    """
    # 성능 로깅 초기화
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    perf_logger = get_performance_logger()
    stage_times = {}

    print(f"[REQUEST START] {request_id} - User: {request.userId} - Content: '{request.requestContent[:50]}...'")

    try:
        # LangGraph 활성화 여부 확인 후 자동 선택
        if is_langgraph_enabled():
            print(f"LangGraph 자동 선택 - 성능 최적화 모드")
            # LangGraph 워크플로우로 처리
            api_response, processing_time, metadata = await process_template_with_langgraph(
                user_id=request.userId,
                request_content=request.requestContent,
                conversation_context=request.conversationContext
            )

            # 성능 개선 정보 출력
            improvement_info = metadata.get("performance_improvement", {})
            if improvement_info.get("improvement_percentage", 0) > 0:
                print(f"성능 개선: {improvement_info['improvement_percentage']}% 단축 "
                      f"({improvement_info.get('time_saved', 0):.2f}초 절약)")

            # LangGraph 결과를 기존 API 형식으로 변환
            if api_response.get("success"):
                template_data = api_response.get("data", {})
                print(f"[REQUEST SUCCESS] {request_id} - Total: {processing_time:.2f}s (LangGraph)")
                return ApiResult.ok(template_data)

            elif api_response.get("status") == "need_more_info":
                # 부분 완성 응답 처리
                template_data = api_response.get("data", {})
                result = ApiResult.ok(template_data)
                return JSONResponse(status_code=202, content=result.dict())

            else:
                # LangGraph 오류 시 기존 방식으로 폴백
                print(f"LangGraph 처리 실패, 기존 방식으로 폴백: {api_response.get('error', 'Unknown error')}")

        # 기존 방식 처리 (LangGraph 비활성화 또는 폴백)
        print(f"기존 Agent 방식으로 처리")

        # 1. Agent1 초기화 및 분석
        with TimingContext(perf_logger, "Agent1_Initialization", request_id) as ctx:
            agent1 = Agent1()
        stage_times['agent1_init'] = ctx.duration

        # 2. 사용자 입력 처리
        with TimingContext(perf_logger, "Agent1_Processing", request_id) as ctx:
            agent1_result = await agent1.process_query_async(
                request.requestContent,
                conversation_context=request.conversationContext
            )
        stage_times['agent1_processing'] = ctx.duration

        # 3. Agent1 처리 결과에 따른 분기
        if agent1_result['status'] == 'inappropriate_request':
            # 부적절한 요청 검출 (비즈니스 알림톡에 적합하지 않음)
            return create_error_response(
                "INAPPROPRIATE_REQUEST",
                agent1_result['message']
            )

        elif agent1_result['status'] == 'profanity_retry':
            # 비속어 검출
            return create_error_response(
                "PROFANITY_DETECTED",
                agent1_result['message']
            )

        elif agent1_result['status'] == 'reask_required':
            # 추가 정보 필요
            missing_vars = agent1_result.get('missing_variables', [])
            # missing_variables 형태를 Variable 형태로 변환
            formatted_missing_vars = []
            for i, var in enumerate(missing_vars):
                if isinstance(var, str):
                    formatted_missing_vars.append({
                        "id": i+1,
                        "variableKey": var,
                        "placeholder": f"#{{{var}}}",
                        "inputType": "TEXT",
                        "value": ""
                    })
                else:
                    formatted_missing_vars.append(var)

            return create_partial_response(
                user_id=request.userId,
                partial_template="",  # Agent1 단계에서는 템플릿이 없음
                missing_variables=formatted_missing_vars,
                mapped_variables=agent1_result.get('analysis', {}).get('variables', {}),
                industry=[],
                purpose=[]
            )

        elif agent1_result['status'] == 'policy_violation':
            # 정책 위반
            return create_error_response(
                "POLICY_VIOLATION",
                agent1_result['message']
            )

        elif agent1_result['status'] not in ['complete', 'success']:
            # 기타 Agent1 오류
            return create_error_response(
                "AGENT1_ERROR",
                f"Agent1 처리 실패: {agent1_result.get('message', '알 수 없는 오류')}"
            )

        # 4. 템플릿 선택
        with TimingContext(perf_logger, "Template_Selection", request_id) as ctx:
            template_selector = TemplateSelector()
            analysis = agent1_result.get('analysis', {})

            selected_template = await template_selector.select_template_async(
                user_input=request.requestContent,
                options={
                    'variables': analysis.get('variables', {}),
                    'intent': analysis.get('intent', {}),
                    'user_id': request.userId
                }
            )
        stage_times['template_selection'] = ctx.duration

        if not selected_template:
            return create_error_response(
                "TEMPLATE_SELECTION_FAILED",
                "적합한 템플릿을 찾을 수 없습니다"
            )

        # 5. Agent2로 최종 템플릿 생성
        with TimingContext(perf_logger, "Agent2_Initialization", request_id) as ctx:
            agent2 = Agent2()
        stage_times['agent2_init'] = ctx.duration

        with TimingContext(perf_logger, "Agent2_Template_Generation", request_id) as ctx:
            final_template_result, metadata = await agent2.generate_compliant_template_async(
                user_input=request.requestContent,
                agent1_variables=analysis.get('variables', {})
            )
        stage_times['agent2_generation'] = ctx.duration

        if not final_template_result:
            return create_error_response(
                "TEMPLATE_GENERATION_FAILED",
                "템플릿 생성에 실패했습니다"
            )

        # Check if more variables are needed
        if final_template_result.get('status') == 'need_more_variables':
            # Convert TemplateVariable format to Variable format
            missing_vars = final_template_result.get('missing_variables', [])
            formatted_missing_vars = []
            for i, var in enumerate(missing_vars):
                if isinstance(var, dict) and 'variable_key' in var:
                    # TemplateVariable format from Agent2
                    formatted_missing_vars.append({
                        "id": i+1,
                        "variableKey": var["variable_key"],
                        "placeholder": var["placeholder"],
                        "inputType": var["input_type"],
                        "value": ""
                    })
                elif isinstance(var, dict):
                    # Already in Variable format
                    if 'value' not in var:
                        var['value'] = ""
                    formatted_missing_vars.append(var)
                else:
                    # String format (fallback)
                    formatted_missing_vars.append({
                        "id": i+1,
                        "variableKey": str(var),
                        "placeholder": f"#{{{var}}}",
                        "inputType": "TEXT",
                        "value": ""
                    })

            return create_partial_response(
                user_id=request.userId,
                partial_template=final_template_result.get('template', ''),
                missing_variables=formatted_missing_vars,
                mapped_variables=final_template_result.get('mapped_variables', {}),
                industry=final_template_result.get('industry', []),  # Agent2에서 오는 dict 형태 그대로 전달
                purpose=final_template_result.get('purpose', [])    # Agent2에서 오는 dict 형태 그대로 전달
            )

        # Check if template generation failed
        if not final_template_result.get('success'):
            return create_error_response(
                "TEMPLATE_GENERATION_FAILED",
                "템플릿 생성에 실패했습니다"
            )

        # 6. 성공 응답 반환 (Java 호환 구조)
        # Ensure variables have required value field
        variables_list = final_template_result.get('variables', [])
        formatted_variables = []
        for i, var in enumerate(variables_list):
            if isinstance(var, dict):
                if 'value' not in var:
                    var['value'] = ""  # Add missing value field
                formatted_variables.append(var)
            else:
                # Handle other formats if needed
                formatted_variables.append({
                    "id": i+1,
                    "variableKey": str(var),
                    "placeholder": f"#{{{var}}}",
                    "inputType": "TEXT",
                    "value": ""
                })

        # Industry/Purpose 데이터를 기존 및 새로운 형식 둘 다 생성
        converted_data = convert_industry_purpose_data(
            final_template_result.get('industry', []),
            final_template_result.get('purpose', [])
        )

        # 동적 카테고리 결정
        category_info = get_category_info(converted_data["industries"], converted_data["purposes"])

        template_data = TemplateSuccessData(
            id=None,  # Java 백엔드에서 DB 자동생성 ID 사용
            userId=request.userId,
            categoryId=category_info["categoryId"],
            title=category_info["title"],
            content=final_template_result.get('template', ''),
            imageUrl=None,
            type=determine_template_type(final_template_result.get('buttons', [])),
            buttons=final_template_result.get('buttons', []),
            variables=formatted_variables,
            industries=converted_data["industries"],
            purposes=converted_data["purposes"],
            _mapped_variables={}  # 완성된 템플릿은 빈 객체
        )

        # 최종 성능 로깅
        total_time = time.time() - start_time
        stage_times['total'] = total_time
        stage_times['response_formatting'] = total_time - sum([v for k, v in stage_times.items() if k != 'total'])

        # 상세 로깅
        perf_logger.log_request_timing(
            request_id=request_id,
            user_id=request.userId,
            request_content=request.requestContent,
            total_time=total_time,
            stage_times=stage_times,
            metadata={
                "status": "success",
                "template_length": len(str(template_data.content or '')),
                "variables_count": len(template_data.variables or []),
                "has_conversation_context": bool(request.conversationContext)
            }
        )

        print(f"[REQUEST SUCCESS] {request_id} - Total: {total_time:.2f}s")

        # ApiResult로 래핑하여 반환
        return ApiResult.ok(template_data)

    except Exception as e:
        # 에러 성능 로깅
        total_time = time.time() - start_time
        error_message = str(e)

        perf_logger.log_error(
            error_msg=f"템플릿 생성 실패: {error_message}",
            request_id=request_id,
            duration=total_time
        )

        print(f"[REQUEST ERROR] {request_id} - Duration: {total_time:.2f}s - Error: {error_message}")

        # 예상치 못한 오류 - 디버그 로깅 추가
        print(f"DEBUG: 템플릿 생성 중 예외 발생: {error_message}")
        import traceback
        traceback.print_exc()

        # 특정 오류에 대한 세부 처리
        if "Please enter in Korean" in error_message:
            return create_error_response(
                "LANGUAGE_VALIDATION_ERROR",
                error_message,
                None,
                400
            )
        elif "quota" in error_message.lower() or "rate limit" in error_message.lower():
            return create_error_response(
                "API_QUOTA_EXCEEDED",
                "API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요.",
                error_message,
                429
            )
        elif "timeout" in error_message.lower():
            return create_error_response(
                "PROCESSING_TIMEOUT",
                "템플릿 생성 시간이 초과되었습니다. 다시 시도해주세요.",
                "AI 처리 시간 초과 (30초 제한)",
                408
            )
        else:
            return create_error_response(
                "INTERNAL_SERVER_ERROR",
                "서버 내부 오류가 발생했습니다.",
                error_message,
                500
            )


@router.get("/templates/test", tags=["Template Generation"],
           responses={
               200: {"model": TemplateSuccessResponse},
               400: {"model": ErrorResponseWithDetails},
               500: {"model": ErrorResponseWithDetails}
           })
async def test_template_generation() -> Dict[str, Any]:
    """
    템플릿 생성 기능 테스트용 엔드포인트
    """
    test_request = TemplateRequest(
        userId=999,
        requestContent="독서모임 안내 메시지를 만들어주세요"
    )

    return await create_template(test_request)


class LLMStatusResponse(BaseModel):
    """LLM 상태 응답 모델"""
    status: str
    llm_status: Dict[str, Any]
    timestamp: str

@router.get("/llm/status", tags=["Template Generation"])
async def get_llm_status() -> Dict[str, Any]:
    """
    LLM 제공자 상태 간단 확인 (AI명세서.txt 호환)
    """
    try:
        llm_manager = get_llm_manager()
        llm_status = llm_manager.get_status()

        # AI명세서.txt 형식에 맞는 응답 구조
        result_data = {
            "available_providers": llm_status.get("available_providers", []),
            "primary_provider": llm_status.get("primary_provider", "unknown"),
            "failure_counts": llm_status.get("failure_counts", {}),
            "gemini_configured": llm_status.get("gemini_configured", False),
            "openai_configured": llm_status.get("openai_configured", False)
        }

        return ApiResult.ok(result_data)

    except Exception as e:
        return create_error_response(
            "LLM_STATUS_ERROR",
            f"LLM 상태 확인 실패: {e}"
        )

@router.get("/templates/llm/status", tags=["Template Generation"],
           response_model=LLMStatusResponse)
async def get_llm_status_detailed() -> Dict[str, Any]:
    """
    템플릿 생성에 사용되는 LLM 상태 확인 (상세 버전)
    """
    try:
        llm_manager = get_llm_manager()
        return {
            "status": "success",
            "llm_status": llm_manager.get_status(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return create_error_response(
            "LLM_STATUS_ERROR",
            f"LLM 상태 확인 실패: {e}"
        )