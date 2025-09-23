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
from pydantic import BaseModel, Field, field_validator
from typing import List, Union

# 모듈 임포트
from app.agents.agent1 import Agent1
from app.agents.agent2 import Agent2
from app.core.template_selector import TemplateSelector
from app.utils.llm_provider_manager import get_llm_manager
from app.dto.api_result import ApiResult

router = APIRouter()


class TemplateRequest(BaseModel):
    """템플릿 생성 요청 모델"""
    userId: int
    requestContent: str = Field(..., min_length=10, max_length=1000, description="구체적인 알림톡 템플릿 요청 내용 (최소 10자 이상)")
    conversationContext: Optional[str] = None  # 재질문 컨텍스트

    @field_validator('requestContent')
    @classmethod
    def validate_request_content(cls, v):
        """requestContent 유효성 검증"""
        if not v or v.strip() == "":
            raise ValueError("템플릿 요청 내용을 입력해주세요")

        # 기본값이나 의미없는 텍스트 필터링
        invalid_inputs = ["string", "test", "example", "샘플", "테스트", "없음", "기본값"]
        if v.strip().lower() in invalid_inputs:
            raise ValueError("구체적인 템플릿 요청 내용을 입력해주세요")

        # 최소 단어 수 검증 (2단어 이상)
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
    industry: List[str]
    purpose: List[str]
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
    error_detail = ErrorDetail(code=error_code, message=message, details=details)
    error_response = ErrorResponse(error=error_detail, timestamp=datetime.now().isoformat() + "Z")
    error_result = ApiResult(data=None, message=None, error=error_response)
    return JSONResponse(
        status_code=status_code,
        content=error_result.model_dump()
    )


def create_partial_response(user_id: int, partial_template: str, missing_variables: List[dict], mapped_variables: Dict[str, str], industry: List[dict], purpose: List[dict]) -> JSONResponse:
    """부분 완성 응답 생성 (202 상태코드) - Java 호환 구조"""
    template_data = TemplateSuccessData(
        id=None,  # 부분 완성 상태 (아직 DB에 저장되지 않음)
        userId=user_id,
        categoryId="004001",
        title="알림톡 (부분 완성)",
        content=partial_template,
        imageUrl=None,
        type="MESSAGE",
        buttons=[],
        variables=missing_variables,  # 누락된 변수들
        industry=[item.get('name', item) if isinstance(item, dict) else str(item) for item in (industry or [])],
        purpose=[item.get('name', item) if isinstance(item, dict) else str(item) for item in (purpose or [])],
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
    try:
        # 1. Agent1 초기화 및 분석
        agent1 = Agent1()

        # 2. 사용자 입력 처리
        agent1_result = await agent1.process_query_async(
            request.requestContent,
            conversation_context=request.conversationContext
        )

        # 3. Agent1 처리 결과에 따른 분기
        if agent1_result['status'] == 'profanity_retry':
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

        if not selected_template:
            return create_error_response(
                "TEMPLATE_SELECTION_FAILED",
                "적합한 템플릿을 찾을 수 없습니다"
            )

        # 5. Agent2로 최종 템플릿 생성
        agent2 = Agent2()

        final_template_result, metadata = await agent2.generate_compliant_template_async(
            user_input=request.requestContent,
            agent1_variables=analysis.get('variables', {})
        )

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
                industry=[item.get('name', item) if isinstance(item, dict) else str(item) for item in final_template_result.get('industry', [])],
                purpose=[item.get('name', item) if isinstance(item, dict) else str(item) for item in final_template_result.get('purpose', [])]
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

        template_data = TemplateSuccessData(
            id=1,
            userId=request.userId,
            categoryId='004001',
            title='알림톡',
            content=final_template_result.get('template', ''),
            imageUrl=None,
            type='MESSAGE',
            buttons=[],
            variables=formatted_variables,
            industry=[item.get('name', item) if isinstance(item, dict) else str(item) for item in final_template_result.get('industry', [])],
            purpose=[item.get('name', item) if isinstance(item, dict) else str(item) for item in final_template_result.get('purpose', [])],
            _mapped_variables={}  # 완성된 템플릿은 빈 객체
        )

        # ApiResult로 래핑하여 반환
        return ApiResult.ok(template_data)

    except Exception as e:
        # 예상치 못한 오류
        error_message = str(e)

        # 특정 오류에 대한 세부 처리
        if "quota" in error_message.lower() or "rate limit" in error_message.lower():
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

@router.get("/templates/llm/status", tags=["Template Generation"],
           response_model=LLMStatusResponse)
async def get_llm_status() -> Dict[str, Any]:
    """
    템플릿 생성에 사용되는 LLM 상태 확인
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