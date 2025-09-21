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
from pydantic import BaseModel, Field
from typing import List, Union

# 모듈 임포트
from app.agents.agent1 import Agent1
from app.agents.agent2 import Agent2
from app.core.template_selector import TemplateSelector
from app.utils.llm_provider_manager import get_llm_manager

router = APIRouter()


class TemplateRequest(BaseModel):
    """템플릿 생성 요청 모델"""
    userId: int
    requestContent: str
    conversationContext: Optional[str] = None  # 재질문 컨텍스트


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


class TemplateSuccessResponse(BaseModel):
    """템플릿 생성 성공 응답 모델"""
    id: int
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


def create_error_response(error_code: str, message: str, details: Any = None) -> Dict[str, Any]:
    """표준화된 에러 응답 생성"""
    return {
        "detail": {
            "error": {
                "code": error_code,
                "message": message,
                "details": details
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    }


@router.post("/templates", tags=["Template Generation"],
            responses={
                200: {"model": TemplateSuccessResponse},
                400: {"model": ErrorResponseWithDetails},
                500: {"model": ErrorResponseWithDetails}
            })
async def create_template(request: TemplateRequest) -> Dict[str, Any]:
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
            return create_error_response(
                "INCOMPLETE_INFORMATION",
                "추가 정보가 필요합니다",
                {
                    "confirmed_variables": agent1_result.get('analysis', {}).get('variables', {}),
                    "missing_variables": agent1_result.get('missing_variables', []),
                    "contextual_question": agent1_result.get('message', ''),
                    "original_input": request.requestContent,
                    "validation_status": "incomplete",
                    "reasoning": agent1_result.get('reasoning', '')
                }
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
            return create_error_response(
                "INCOMPLETE_INFORMATION",
                "추가 정보가 필요합니다",
                {
                    "mapped_variables": final_template_result.get('mapped_variables', {}),
                    "missing_variables": final_template_result.get('missing_variables', []),
                    "partial_template": final_template_result.get('template', ''),
                    "mapping_coverage": final_template_result.get('mapping_coverage', 0.0),
                    "industry": final_template_result.get('industry', []),
                    "purpose": final_template_result.get('purpose', [])
                }
            )

        # Check if template generation failed
        if not final_template_result.get('success'):
            return create_error_response(
                "TEMPLATE_GENERATION_FAILED",
                "템플릿 생성에 실패했습니다"
            )

        # 6. 성공 응답 반환
        return {
            "id": 1,
            "userId": request.userId,
            "categoryId": '004001',
            "title": '알림톡',
            "content": final_template_result.get('template', ''),
            "imageUrl": None,
            "type": 'MESSAGE',
            "buttons": [],
            "variables": final_template_result.get('variables', []),
            "industry": final_template_result.get('industry', []),
            "purpose": final_template_result.get('purpose', [])
        }

    except Exception as e:
        # 예상치 못한 오류
        error_message = str(e)

        # 특정 오류에 대한 세부 처리
        if "quota" in error_message.lower() or "rate limit" in error_message.lower():
            return create_error_response(
                "API_QUOTA_EXCEEDED",
                "API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요.",
                error_message
            )
        elif "timeout" in error_message.lower():
            return create_error_response(
                "PROCESSING_TIMEOUT",
                "템플릿 생성 시간이 초과되었습니다. 다시 시도해주세요.",
                "AI 처리 시간 초과 (30초 제한)"
            )
        else:
            return create_error_response(
                "INTERNAL_SERVER_ERROR",
                "서버 내부 오류가 발생했습니다.",
                error_message
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