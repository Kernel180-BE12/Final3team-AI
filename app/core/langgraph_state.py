#!/usr/bin/env python3
"""
LangGraph State 정의 - JOBER_AI 워크플로우 상태 관리

백엔드 API 호환성 100% 보장하면서 성능 60-85% 향상을 위한 상태 기반 워크플로우
"""

from typing import TypedDict, List, Dict, Optional, Any, Union
from typing_extensions import Annotated
from enum import Enum


class ProcessingStatus(str, Enum):
    """처리 상태 열거형"""
    PROCESSING = "processing"
    NEED_INFO = "need_info"
    COMPLETED = "completed"
    ERROR = "error"


class TemplateSource(str, Enum):
    """템플릿 소스 유형"""
    PREDATA = "predata"      # 기존 predata 폴더 템플릿
    PUBLIC = "public"        # 공개 템플릿
    GENERATED = "generated"  # AI 생성 템플릿


class JoberState(TypedDict):
    """
    JOBER_AI LangGraph 워크플로우 상태

    백엔드 API와 100% 호환성을 유지하면서 내부 처리 최적화
    """

    # ========== 입력 데이터 ==========
    user_input: str
    conversation_context: Optional[str]
    user_id: int

    # ========== Agent1 처리 결과 ==========
    # 입력 검증 결과
    validation_result: Optional[Dict[str, Any]]
    """
    {
        "is_valid": bool,
        "message": str,
        "category": str,  # "language", "business", "profanity" 등
        "confidence": float
    }
    """

    # 추출된 변수들
    extracted_variables: Optional[Dict[str, Any]]
    """
    {
        "variables": Dict[str, str],  # 실제 추출된 변수
        "intent": Dict[str, Any],     # 분류된 의도
        "confidence": float,          # 변수 추출 신뢰도
        "missing_variables": List[str] # 부족한 변수들
    }
    """

    # 정책 검사 결과
    policy_check: Optional[Dict[str, Any]]
    """
    {
        "is_compliant": bool,
        "violations": List[str],
        "risk_level": str,  # "LOW", "MEDIUM", "HIGH"
        "details": Dict[str, Any]
    }
    """

    # ========== 템플릿 선택 결과 ==========
    selected_template: Optional[Dict[str, Any]]
    """
    기존 템플릿이 발견된 경우:
    {
        "template_id": str,
        "similarity_score": float,
        "template_content": str,
        "metadata": Dict[str, Any]
    }
    """

    template_source: Optional[TemplateSource]
    """템플릿 출처: predata, public, generated"""

    # ========== Agent2 처리 결과 ==========
    # 4개 도구 병렬 실행 결과
    tools_results: Optional[Dict[str, Any]]
    """
    {
        "blacklist_result": Dict[str, Any],
        "whitelist_result": Dict[str, Any],
        "guideline_result": Dict[str, Any],
        "law_result": Dict[str, Any],
        "processing_time": float,
        "errors": List[str]
    }
    """

    # 최종 생성된 템플릿
    final_template: Optional[Dict[str, Any]]
    """
    백엔드 API 응답 형식과 동일:
    {
        "template": str,
        "variables": List[Dict],
        "industry": str,
        "purpose": str,
        "metadata": Dict[str, Any]
    }
    """

    # 컴플라이언스 검사 결과
    compliance_check: Optional[Dict[str, Any]]
    """
    {
        "is_compliant": bool,
        "issues": List[str],
        "recommendations": List[str]
    }
    """

    # ========== 상태 관리 ==========
    status: ProcessingStatus
    """현재 처리 상태"""

    error_info: Optional[Dict[str, Any]]
    """
    오류 발생 시 정보:
    {
        "code": str,        # "LANGUAGE_VALIDATION_ERROR", "PROFANITY_DETECTED" 등
        "message": str,
        "details": Dict[str, Any],
        "retry_possible": bool
    }
    """

    completion_percentage: float
    """처리 완료율 (0.0 ~ 100.0)"""

    # ========== 성능 추적 ==========
    processing_times: Optional[Dict[str, float]]
    """
    각 단계별 처리 시간:
    {
        "validation": float,
        "variable_extraction": float,
        "policy_check": float,
        "template_selection": float,
        "template_generation": float,
        "total": float
    }
    """

    # ========== 디버깅 및 로깅 ==========
    debug_info: Optional[Dict[str, Any]]
    """디버깅을 위한 추가 정보"""

    workflow_path: Optional[List[str]]
    """실행된 워크플로우 경로 추적"""


class ConversationState(TypedDict):
    """대화 상태 (Agent1 내부 사용)"""

    # 대화 기본 정보
    session_id: str
    user_input: str
    conversation_context: Optional[str]

    # 변수 관리
    variables: Dict[str, str]           # 확정된 변수들
    candidate_variables: Dict[str, str] # 후보 변수들
    missing_variables: List[str]        # 부족한 변수들

    # 완성도 판단
    completion_percentage: float
    confidence_score: float

    # 질문 생성을 위한 컨텍스트
    previous_questions: List[str]
    user_responses: List[str]

    # 메타데이터
    created_at: str
    last_updated: str


class TemplateGenerationRequest(TypedDict):
    """템플릿 생성 요청 (AsyncTemplateGenerator 용)"""

    user_input: str
    agent1_variables: Optional[Dict[str, str]]
    industry: Optional[str]
    purpose: Optional[str]

    # 성능 옵션
    enable_cache: bool
    parallel_processing: bool
    timeout_seconds: Optional[int]


class WorkflowResult(TypedDict):
    """워크플로우 실행 결과"""

    # 최종 결과
    success: bool
    result_data: Optional[Dict[str, Any]]

    # 오류 정보
    error: Optional[str]
    error_code: Optional[str]

    # 성능 메트릭
    total_processing_time: float
    step_times: Dict[str, float]

    # 백엔드 API 호환성
    api_response: Dict[str, Any]
    """백엔드 API 형식으로 변환된 응답"""


# ========== 상태 조작 헬퍼 함수들 ==========

def initialize_jober_state(
    user_input: str,
    conversation_context: Optional[str] = None,
    user_id: int = 1
) -> JoberState:
    """JoberState 초기화"""
    return JoberState(
        # 입력 데이터
        user_input=user_input,
        conversation_context=conversation_context,
        user_id=user_id,

        # Agent1 결과 (초기값)
        validation_result=None,
        extracted_variables=None,
        policy_check=None,

        # 템플릿 선택 결과 (초기값)
        selected_template=None,
        template_source=None,

        # Agent2 결과 (초기값)
        tools_results=None,
        final_template=None,
        compliance_check=None,

        # 상태 관리
        status=ProcessingStatus.PROCESSING,
        error_info=None,
        completion_percentage=0.0,

        # 성능 추적
        processing_times=None,

        # 디버깅
        debug_info=None,
        workflow_path=[]
    )


def update_completion_percentage(state: JoberState) -> JoberState:
    """완료율 자동 계산 및 업데이트"""
    completed_steps = 0
    total_steps = 6  # validation, extraction, policy, template_selection, generation, compliance

    if state["validation_result"] is not None:
        completed_steps += 1
    if state["extracted_variables"] is not None:
        completed_steps += 1
    if state["policy_check"] is not None:
        completed_steps += 1
    if state["selected_template"] is not None or state["template_source"] is not None:
        completed_steps += 1
    if state["final_template"] is not None:
        completed_steps += 1
    if state["compliance_check"] is not None:
        completed_steps += 1

    state["completion_percentage"] = (completed_steps / total_steps) * 100.0
    return state


def add_workflow_step(state: JoberState, step_name: str) -> JoberState:
    """워크플로우 경로에 단계 추가"""
    if state.get("workflow_path") is None:
        state["workflow_path"] = []
    state["workflow_path"].append(step_name)
    return state


def is_ready_for_template_generation(state: JoberState) -> bool:
    """템플릿 생성 준비 완료 여부 확인"""
    return (
        state["validation_result"] is not None and
        state["validation_result"]["is_valid"] and
        state["extracted_variables"] is not None and
        state["policy_check"] is not None and
        state["policy_check"]["is_compliant"]
    )


def should_use_existing_template(state: JoberState) -> bool:
    """기존 템플릿 사용 여부 판단"""
    return (
        state["selected_template"] is not None and
        state["selected_template"]["similarity_score"] > 0.85
    )


def convert_to_api_response(state: JoberState) -> Dict[str, Any]:
    """JoberState를 백엔드 API 응답 형식으로 변환"""

    if state["status"] == ProcessingStatus.ERROR:
        return {
            "success": False,
            "error": state["error_info"]["message"] if state["error_info"] else "Unknown error",
            "error_code": state["error_info"]["code"] if state["error_info"] else "UNKNOWN_ERROR"
        }

    if state["status"] == ProcessingStatus.NEED_INFO:
        return {
            "success": False,
            "status": "need_more_info",
            "completion_percentage": state["completion_percentage"],
            "message": "추가 정보가 필요합니다.",
            "missing_variables": state["extracted_variables"]["missing_variables"] if state["extracted_variables"] else []
        }

    if state["status"] == ProcessingStatus.COMPLETED and state["final_template"]:
        return {
            "success": True,
            "data": state["final_template"],
            "processing_time": state["processing_times"]["total"] if state["processing_times"] else 0.0,
            "template_source": state["template_source"]
        }

    # 처리 중
    return {
        "success": False,
        "status": "processing",
        "completion_percentage": state["completion_percentage"]
    }