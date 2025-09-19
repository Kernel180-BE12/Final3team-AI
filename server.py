
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import time
from datetime import datetime
import re
from collections import defaultdict
import threading

# api.py에서 기존 로직을 가져옵니다.
from api import get_template_api

# 세션 관리 시스템 추가
from src.core.session_manager import get_session_manager, create_session_for_user, get_user_session
from src.core.template_preview import get_preview_generator
from src.core.session_models import SessionStatus

# --- Pydantic 모델 정의 ---
# FastAPI가 JSON의 snake_case <-> camelCase 변환을 자동으로 처리해줍니다.

class TemplateCreationRequest(BaseModel):
    """Spring Boot에서 오는 요청을 위한 모델"""
    user_id: int = Field(..., alias='userId')
    request_content: str = Field(..., alias='requestContent')
    conversation_context: Optional[str] = Field(None, alias='conversationContext')  # 재질문 컨텍스트

    class Config:
        allow_population_by_field_name = True  # snake_case와 camelCase 둘 다 허용
        populate_by_name = True

# 응답 모델 (Response)
class Button(BaseModel):
    id: int
    name: str
    ordering: int
    link_pc: Optional[str] = Field(None, alias='linkPc')
    link_and: Optional[str] = Field(None, alias='linkAnd')
    link_ios: Optional[str] = Field(None, alias='linkIos')

class Variable(BaseModel):
    id: int
    variable_key: str = Field(..., alias='variableKey')
    placeholder: str
    input_type: str = Field(..., alias='inputType')

class Industry(BaseModel):
    id: int
    name: str

class Purpose(BaseModel):
    id: int
    name: str

# 세션 관련 새로운 모델들
class VariableUpdateRequest(BaseModel):
    """변수 업데이트 요청 모델"""
    variables: Dict[str, str] = Field(..., description="업데이트할 변수들")
    force_update: bool = Field(False, description="강제 업데이트 여부")

class SessionVariableInfo(BaseModel):
    """세션 변수 정보 모델"""
    variable_key: str = Field(..., alias='variableKey')
    placeholder: str
    variable_type: str = Field(..., alias='variableType')
    required: bool = True
    description: Optional[str] = None
    example: Optional[str] = None
    input_hint: Optional[str] = Field(None, alias='inputHint')
    priority: int = 0

class SessionPreviewResponse(BaseModel):
    """세션 미리보기 응답 모델"""
    success: bool
    session_id: str = Field(..., alias='sessionId')
    preview_template: str = Field(..., alias='previewTemplate')
    completion_percentage: float = Field(..., alias='completionPercentage')
    total_variables: int = Field(..., alias='totalVariables')
    completed_variables: int = Field(..., alias='completedVariables')
    missing_variables: List[SessionVariableInfo] = Field(..., alias='missingVariables')
    next_suggested_variables: List[SessionVariableInfo] = Field(..., alias='nextSuggestedVariables')
    quality_score: float = Field(..., alias='qualityScore')
    estimated_final_length: int = Field(..., alias='estimatedFinalLength')
    readiness_for_completion: bool = Field(..., alias='readinessForCompletion')

class CompleteTemplateRequest(BaseModel):
    """템플릿 완료 요청 모델"""
    force_complete: bool = Field(False, alias='forceComplete')
    final_adjustments: Optional[Dict[str, str]] = Field(None, alias='finalAdjustments')

class TemplateResponse(BaseModel):
    """FastAPI가 Spring Boot로 보낼 응답을 위한 모델"""
    id: int
    user_id: int = Field(..., alias='userId')
    category_id: str = Field(..., alias='categoryId')
    title: str
    content: str
    image_url: Optional[str] = Field(None, alias='imageUrl')
    type: str
    buttons: List[Button]
    variables: List[Variable]
    industry: List[Industry]
    purpose: List[Purpose]


# --- 유틸리티 함수 ---

# Rate limiting을 위한 메모리 저장소 (개선된 버전)
request_counts = defaultdict(list)
RATE_LIMIT_PER_MINUTE = 10
RATE_LIMIT_WINDOW = 60  # seconds
rate_limit_lock = threading.Lock()  # 동시성 보호
last_cleanup_time = time.time()     # 마지막 정리 시간
CLEANUP_INTERVAL = 300              # 5분마다 정리

def create_error_response(code: str, message: str, details: Optional[Any] = None, retry_after: Optional[int] = None) -> Dict[str, Any]:
    """표준화된 에러 응답 생성"""
    error_response = {
        "error": {
            "code": code,
            "message": message
        },
        "timestamp": datetime.now().isoformat() + "Z"
    }

    # 재질문 데이터는 별도 필드로 처리
    if code == "INCOMPLETE_INFORMATION" and details:
        error_response["reask_data"] = details
    elif code == "PROFANITY_RETRY" and details:
        error_response["retry_data"] = details
    elif details:
        error_response["error"]["details"] = details

    if retry_after:
        error_response["retryAfter"] = retry_after
    return error_response

# Issue 5 해결: 3단계 입력 검증 시스템 import
try:
    from src.utils.comprehensive_validator import validate_input_quick, get_comprehensive_validator
    ADVANCED_VALIDATION_AVAILABLE = True
    print("🛡️ 고급 3단계 입력 검증 시스템 로드 완료")
except ImportError as e:
    print(f"⚠️ 고급 검증 시스템 로드 실패: {e}")
    ADVANCED_VALIDATION_AVAILABLE = False

def is_meaningful_text(text: str) -> bool:
    """
    의미있는 텍스트인지 판별 (Issue 5 해결: 고급 검증 시스템 적용)
    """
    if ADVANCED_VALIDATION_AVAILABLE:
        # 3단계 고급 검증 시스템 사용
        is_valid, error_code, error_message = validate_input_quick(text)
        if not is_valid:
            print(f"🚫 입력 차단 [{error_code}]: {error_message}")
        return is_valid
    else:
        # 폴백: 기존 간단한 검증
        if not text or not text.strip():
            return False

        # 특수문자와 공백만 있는지 체크
        cleaned = re.sub(r'[^\w\s가-힣]', '', text.strip())
        if not cleaned:
            return False

        # 너무 짧은 텍스트
        if len(cleaned) < 2:
            return False

        return True

def validate_input_comprehensive_api(text: str) -> tuple[bool, dict]:
    """
    API용 종합 입력 검증 (Issue 5 해결)

    Returns:
        tuple[bool, dict]: (유효여부, 에러응답객체)
    """
    if not ADVANCED_VALIDATION_AVAILABLE:
        # 폴백: 기존 검증만 사용
        is_valid = is_meaningful_text(text)
        if not is_valid:
            return False, create_error_response(
                "INVALID_INPUT",
                "입력 내용이 올바르지 않습니다. 의미있는 텍스트를 입력해주세요.",
                "특수문자만 입력되었거나 공백만 포함된 요청입니다."
            )
        return True, {}

    # 고급 검증 시스템 사용
    validator = get_comprehensive_validator()
    result = validator.validate_input_comprehensive(text)

    if result.is_valid:
        return True, {}
    else:
        # 검증 단계별 상세 에러 응답 생성
        error_response = create_error_response(
            result.error_code,
            result.error_message,
            {
                "failed_stage": result.failed_stage.value if result.failed_stage else None,
                "suggestion": result.suggestion,
                "validation_details": result.stage_results
            }
        )
        return False, error_response

def cleanup_old_requests():
    """오래된 요청 기록 정리하여 메모리 누수 방지"""
    global last_cleanup_time
    current_time = time.time()

    # 5분마다만 정리 실행
    if current_time - last_cleanup_time < CLEANUP_INTERVAL:
        return

    with rate_limit_lock:
        # 모든 사용자의 오래된 요청 제거
        for user_id in list(request_counts.keys()):
            request_counts[user_id] = [
                req_time for req_time in request_counts[user_id]
                if current_time - req_time < RATE_LIMIT_WINDOW
            ]
            # 빈 리스트면 사용자 기록 완전 삭제
            if not request_counts[user_id]:
                del request_counts[user_id]

        last_cleanup_time = current_time
        print(f"🧹 Rate limit 메모리 정리 완료: {len(request_counts)}명 활성 사용자")

def check_rate_limit(user_id: int) -> bool:
    """사용자별 요청 제한 확인 (동시성 보호 + 메모리 누수 방지)"""

    # 주기적 메모리 정리
    cleanup_old_requests()

    current_time = time.time()

    with rate_limit_lock:  # 동시성 보호
        user_requests = request_counts[user_id]

        # 1분 이전 요청들 제거
        valid_requests = [
            req_time for req_time in user_requests
            if current_time - req_time < RATE_LIMIT_WINDOW
        ]
        request_counts[user_id] = valid_requests

        # 현재 요청 수 확인
        if len(valid_requests) >= RATE_LIMIT_PER_MINUTE:
            print(f"🚫 Rate limit 초과: user_id={user_id}, 요청수={len(valid_requests)}")
            return False

        # 현재 요청 추가
        request_counts[user_id].append(current_time)
        print(f"✅ Rate limit 통과: user_id={user_id}, 요청수={len(valid_requests)+1}")
        return True

# --- FastAPI 애플리케이션 설정 ---

app = FastAPI()
# api.py에 정의된 싱글톤 인스턴스를 가져옵니다.
template_api = get_template_api()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/ai/templates" and request.method == "POST":
        body = await request.body()
        print(f" Content-Type: {request.headers.get('content-type')}")
        print(f" Content-Length: {request.headers.get('content-length')}")
        print(f" Body Length: {len(body)} bytes")
        print(f" 원시 요청 데이터: '{body.decode('utf-8')}'")
        
        # body를 다시 사용할 수 있도록 설정
        async def receive():
            return {"type": "http.request", "body": body}
        request._receive = receive
        
    response = await call_next(request)
    return response

@app.post("/ai/templates", response_model=TemplateResponse, status_code=200)
async def create_template(request: TemplateCreationRequest):
    """
    AI를 사용하여 템플릿을 생성하고 Spring Boot가 요구하는 형식으로 반환합니다.
    """
    try:
        # 요청 데이터 로깅
        print(f" 받은 요청: user_id={request.user_id}, content={request.request_content}")
        
        # Rate limiting 확인
        if not check_rate_limit(request.user_id):
            raise HTTPException(
                status_code=429, 
                detail=create_error_response(
                    "RATE_LIMIT_EXCEEDED",
                    "너무 많은 요청을 보내셨습니다. 잠시 후 다시 시도해주세요.",
                    "1분당 최대 10회 요청 가능합니다.",
                    30
                )
            )
        
        # 요청 필드 검증
        if not request.request_content or not request.request_content.strip():
            raise HTTPException(
                status_code=400, 
                detail=create_error_response(
                    "INVALID_INPUT",
                    "requestContent is required and cannot be empty"
                )
            )
        
        if not request.user_id or request.user_id <= 0:
            raise HTTPException(
                status_code=400, 
                detail=create_error_response(
                    "INVALID_INPUT",
                    "userId is required and must be greater than 0"
                )
            )
            
        # 콘텐츠 길이 검증
        if len(request.request_content.strip()) > 500:
            raise HTTPException(
                status_code=413, 
                detail=create_error_response(
                    "CONTENT_TOO_LARGE",
                    "입력 텍스트가 너무 깁니다. 500자 이하로 입력해주세요.",
                    f"현재 입력: {len(request.request_content.strip())}자, 최대 허용: 500자"
                )
            )
            
        # Issue 5 해결: 3단계 종합 입력 검증
        validation_passed, validation_error = validate_input_comprehensive_api(request.request_content)
        if not validation_passed:
            raise HTTPException(
                status_code=400,
                detail=validation_error
            )
        
        # 텍스트 전처리 (긴 텍스트에 적절한 공백 추가)
        processed_content = request.request_content.replace(".", ". ").replace("  ", " ").strip()
        print(f" 전처리된 내용: {processed_content[:100]}...")
        # 1. api.py의 비동기 템플릿 생성 함수 호출 (전처리된 내용 + 컨텍스트 사용)
        generation_result = await template_api.generate_template_async(
            user_input=processed_content,
            conversation_context=request.conversation_context
        )

        if not generation_result.get("success"):
            # 템플릿 생성 실패 시 세분화된 처리
            if generation_result.get("is_duplicate"):
                raise HTTPException(
                    status_code=409, 
                    detail=create_error_response(
                        "DUPLICATE_TEMPLATE",
                        "유사한 템플릿이 이미 존재합니다.",
                        generation_result.get("error", "")
                    )
                )
            
            # 에러 코드 및 타입별 처리
            error_code = generation_result.get("error_code", "unknown")
            error_type = generation_result.get("error_type", "unknown")
            error_message = generation_result.get("error", "Template generation failed")
            
            # API에서 제공하는 error_code 우선 처리
            if error_code == "INCOMPLETE_INFORMATION":
                raise HTTPException(
                    status_code=400,
                    detail=create_error_response(
                        "INCOMPLETE_INFORMATION",
                        "추가 정보가 필요합니다",
                        generation_result.get("reask_data")
                    )
                )
            elif error_code == "PROFANITY_RETRY":
                raise HTTPException(
                    status_code=400,
                    detail=create_error_response(
                        "PROFANITY_RETRY",
                        "비속어가 감지되었습니다. 다시 입력해주세요",
                        generation_result.get("retry_data")
                    )
                )
            elif error_code == "MISSING_VARIABLES":
                raise HTTPException(
                    status_code=400,
                    detail=create_error_response(
                        "MISSING_VARIABLES",
                        "필요한 변수가 부족합니다.",
                        generation_result.get("message", error_message)
                    )
                )
            elif error_code == "EMPTY_INPUT":
                raise HTTPException(
                    status_code=400,
                    detail=create_error_response(
                        "EMPTY_INPUT",
                        "입력이 비어있습니다.",
                        error_message
                    )
                )
            elif error_code == "TEMPLATE_SELECTION_FAILED":
                raise HTTPException(
                    status_code=500,
                    detail=create_error_response(
                        "TEMPLATE_SELECTION_FAILED",
                        "템플릿 선택에 실패했습니다.",
                        error_message
                    )
                )
            elif error_code == "QUALITY_VERIFICATION_FAILED":
                raise HTTPException(
                    status_code=422,
                    detail=create_error_response(
                        "QUALITY_VERIFICATION_FAILED",
                        "템플릿 품질 검증에 실패했습니다.",
                        error_message
                    )
                )
            elif error_code == "BACKEND_HTTP_ERROR":
                raise HTTPException(
                    status_code=502,
                    detail=create_error_response(
                        "BACKEND_HTTP_ERROR",
                        "백엔드 서버 오류가 발생했습니다.",
                        error_message
                    )
                )
            elif error_code == "BACKEND_CONNECTION_ERROR":
                raise HTTPException(
                    status_code=503,
                    detail=create_error_response(
                        "BACKEND_CONNECTION_ERROR",
                        "백엔드 서버에 연결할 수 없습니다.",
                        error_message
                    )
                )
            elif error_code == "INTERNAL_ERROR":
                raise HTTPException(
                    status_code=500,
                    detail=create_error_response(
                        "INTERNAL_ERROR",
                        "서버 내부 오류가 발생했습니다.",
                        error_message
                    )
                )
            elif error_type == "profanity":
                raise HTTPException(
                    status_code=422,
                    detail=create_error_response(
                        "INAPPROPRIATE_CONTENT",
                        "부적절한 내용이 감지되었습니다.",
                        "입력 내용을 수정하여 다시 시도해주세요."
                    )
                )
            elif error_type == "policy_violation":
                raise HTTPException(
                    status_code=422,
                    detail=create_error_response(
                        "POLICY_VIOLATION",
                        "정책 위반 내용이 감지되었습니다.",
                        error_message
                    )
                )
            elif "timeout" in error_message.lower() or "시간" in error_message:
                raise HTTPException(
                    status_code=504,
                    detail=create_error_response(
                        "PROCESSING_TIMEOUT",
                        "템플릿 생성 시간이 초과되었습니다. 다시 시도해주세요.",
                        "AI 처리 시간 초과 (30초 제한)"
                    )
                )
            elif "api" in error_message.lower() or "service" in error_message.lower():
                raise HTTPException(
                    status_code=503,
                    detail=create_error_response(
                        "AI_SERVICE_ERROR",
                        "AI 서비스에 일시적 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
                        "Gemini API 또는 OpenAPI 연결 오류"
                    )
                )
            else:
                # 기타 내부 서버 오류
                raise HTTPException(
                    status_code=500,
                    detail=create_error_response(
                        "INTERNAL_SERVER_ERROR",
                        "서버 내부 오류가 발생했습니다.",
                        error_message
                    )
                )

        # 2. 생성된 결과를 DB 저장용 JSON 포맷으로 변환
        json_export = template_api.export_to_json(
            result=generation_result,
            user_input=processed_content,  # 전처리된 내용 사용
            user_id=request.user_id
        )

        if not json_export.get("success"):
            raise HTTPException(
                status_code=500, 
                detail=create_error_response(
                    "JSON_EXPORT_ERROR",
                    "템플릿 데이터 변환에 실패했습니다.",
                    json_export.get("error", "Failed to export template to JSON")
                )
            )

        exported_data = json_export["data"]
        template_data = exported_data["template"]
        variable_data = exported_data["variables"]

        # 3. 세션 생성 및 템플릿 데이터 설정
        session_manager = get_session_manager()
        session_id = session_manager.create_session(
            user_id=request.user_id,
            original_request=processed_content,
            conversation_context=request.conversation_context
        )

        # 세션에 템플릿 데이터 설정
        session_manager.set_template_data(
            session_id=session_id,
            template=generation_result["template"],
            variables=generation_result.get("variables", []),
            source=generation_result.get("metadata", {}).get("source", "generated")
        )

        # Agent 결과 저장 (있는 경우)
        if generation_result.get("metadata"):
            session_manager.update_session(session_id, {
                "agent1_result": generation_result["metadata"].get("agent1_result"),
                "agent2_result": generation_result["metadata"].get("agent2_result")
            })

        # 4. 최종 응답 모델(TemplateResponse)에 맞춰 데이터 매핑
        response_data = {
            "id": 1, # 임시 ID
            "userId": template_data["user_id"],
            "categoryId": template_data["category_id"],
            "title": template_data["title"],
            "content": template_data["content"],
            "imageUrl": template_data["image_url"],
            "type": detect_content_type(template_data["content"]),
            "buttons": _get_metadata_buttons(generation_result, template_data["content"]),
            "variables": [
                {
                    "id": i + 1, # 임시 ID
                    "variableKey": var.get("variable_key"),
                    "placeholder": var.get("placeholder"),
                    "inputType": var.get("input_type", "TEXT")  # 기본값 설정
                } for i, var in enumerate(variable_data)
            ],
            "industry": _get_metadata_industries(generation_result, template_data["content"], template_data["category_id"]),
            "purpose": _get_metadata_purpose(generation_result, template_data["content"], processed_content)
        }

        # 5. 세션 정보 추가 - 챗봇 연동을 위한 핵심 데이터
        session = session_manager.get_session(session_id)
        response_data["session_data"] = {
            "session_id": session_id,
            "has_missing_variables": len(session.missing_variables) > 0,
            "completion_percentage": round(session.completion_percentage, 1),
            "missing_variables": session.missing_variables,
            "total_variables": len(session.template_variables),
            "session_endpoints": {
                "update_variables": f"/ai/templates/{session_id}/variables",
                "preview": f"/ai/templates/{session_id}/preview",
                "complete": f"/ai/templates/{session_id}/complete"
            },
            "expires_in_minutes": max(0, int((session.expires_at - datetime.now()).total_seconds() / 60))
        }

        return response_data

    except HTTPException:
        # 이미 처리된 HTTPException은 다시 발생
        raise
    except ConnectionError as e:
        # DB나 외부 서비스 연결 오류
        raise HTTPException(
            status_code=502,
            detail=create_error_response(
                "CONNECTION_ERROR",
                "데이터베이스 또는 외부 서비스 연결에 실패했습니다.",
                f"연결 오류: {str(e)}"
            )
        )
    except TimeoutError as e:
        # 타임아웃 오류
        raise HTTPException(
            status_code=504,
            detail=create_error_response(
                "TIMEOUT_ERROR",
                "요청 처리 시간이 초과되었습니다.",
                f"타임아웃: {str(e)}"
            )
        )
    except Exception as e:
        # 예상치 못한 기타 오류
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "UNEXPECTED_ERROR",
                "예상치 못한 오류가 발생했습니다.",
                str(e)
            )
        )

def detect_content_type(content: str) -> str:
    """템플릿 내용을 분석하여 적절한 타입 결정"""
    content_lower = content.lower()

    # 1. 문서 키워드 확장 (우선순위 높음)
    doc_keywords = ['pdf', 'hwp', 'docx', 'xlsx', 'pptx', '문서', '파일', '첨부', '다운로드', '업로드']
    doc_extensions = ['.pdf', '.hwp', '.doc', '.xls', '.ppt']

    if (any(keyword in content_lower for keyword in doc_keywords) or
        any(ext in content_lower for ext in doc_extensions)):
        return 'DOCS'

    # 2. URL 패턴 확장 (중간 우선순위)
    url_patterns = ['http://', 'https://', 'www.', '.com', '.co.kr', '.net', '.org']
    link_keywords = ['링크', '바로가기', '홈페이지', '사이트', '클릭']

    if (any(pattern in content_lower for pattern in url_patterns) or
        any(keyword in content_lower for keyword in link_keywords)):
        return 'LINK'

    # 3. 기본값
    return 'MESSAGE'


def _get_metadata_buttons(generation_result: Dict, content: str) -> List[Dict]:
    """메타데이터에서 버튼 정보 추출 또는 추론"""
    # 1. 직접 매칭된 템플릿의 버튼 사용
    buttons = generation_result.get("metadata", {}).get("source_info", {}).get("buttons", [])
    if buttons:
        return [{
            "id": i + 1,
            "name": btn.get("name", "버튼"),
            "ordering": btn.get("ordering", i + 1),
            "linkPc": btn.get("linkPc"),
            "linkAnd": btn.get("linkAnd"),
            "linkIos": btn.get("linkIos")
        } for i, btn in enumerate(buttons)]
    
    # 2. 생성된 경우 버튼 필요성 추론 (보수적 접근)
    button_keywords = ["자세히", "확인", "신청", "예약", "문의", "링크", "URL", "클릭"]

    # 더 명확한 버튼 필요성 판단 (2개 이상 키워드 또는 URL 포함 시에만)
    keyword_count = sum(1 for keyword in button_keywords if keyword in content)
    has_url = any(url_pattern in content for url_pattern in ["http://", "https://", "www.", ".com", ".co.kr"])

    if keyword_count >= 2 or has_url:
        return [{
            "id": 1,
            "name": "자세히 보기",
            "ordering": 1,
            "linkPc": "https://example.com",
            "linkAnd": None,
            "linkIos": None
        }]

    # 기본값: 빈 리스트 (프론트에서 버튼 비활성화)
    return []

def _get_metadata_industries(generation_result: Dict, content: str, category_id: str) -> List[Dict]:
    """메타데이터에서 업종 정보 추출 또는 추론"""
    # 1. 직접 매칭된 템플릿의 업종 사용
    industries = generation_result.get("metadata", {}).get("source_info", {}).get("industries", [])
    if industries:
        return [{"id": i + 1, "name": industry} for i, industry in enumerate(industries)]
    
    # 2. 카테고리 기반 추론 (GroupName 기반)
    category_industry_map = {
        # 회원 관련
        "001001": ["서비스업"], "001002": ["서비스업"], "001003": ["서비스업"],
        # 구매 관련  
        "002001": ["소매업"], "002002": ["소매업"], "002003": ["소매업"], 
        "002004": ["소매업"], "002005": ["소매업"],
        # 예약 관련
        "003001": ["서비스업"], "003002": ["서비스업"], "003003": ["서비스업"], "003004": ["서비스업"],
        # 서비스이용 관련
        "004001": ["서비스업"], "004002": ["서비스업"], "004003": ["서비스업"],
        "004004": ["서비스업"], "004005": ["서비스업"], "004006": ["서비스업"],
        "004007": ["서비스업"], "004008": ["서비스업"],
        # 리포팅 관련
        "005001": ["서비스업"], "005002": ["서비스업"], "005003": ["서비스업"],
        "005004": ["서비스업"], "005005": ["서비스업"], "005006": ["서비스업"],
        # 배송 관련
        "006001": ["배송업"], "006002": ["배송업"], "006003": ["배송업"], "006004": ["배송업"],
        # 법적고지 관련
        "007001": ["기타"], "007002": ["기타"], "007003": ["기타"], "007004": ["기타"],
        # 업무알림 관련  
        "008001": ["서비스업"], "008002": ["서비스업"],
        # 쿠폰/포인트 관련
        "009001": ["소매업"], "009002": ["소매업"], "009003": ["소매업"],
        "009004": ["소매업"], "009005": ["소매업"],
        # 기타
        "999999": ["기타"]
    }
    
    if category_id in category_industry_map:
        industries = category_industry_map[category_id]
        return [{"id": i + 1, "name": industry} for i, industry in enumerate(industries)]
    
    # 3. 내용 기반 키워드 추론
    if any(word in content for word in ["교육", "강의", "수업", "학원"]):
        return [{"id": 1, "name": "학원"}]
    elif any(word in content for word in ["온라인", "인터넷", "웹"]):
        return [{"id": 2, "name": "온라인 강의"}]
    elif any(word in content for word in ["운동", "헬스", "피트니스", "요가"]):
        return [{"id": 3, "name": "피트니스"}]
    elif any(word in content for word in ["공연", "행사", "이벤트", "콘서트"]):
        return [{"id": 4, "name": "공연/행사"}]
    elif any(word in content for word in ["모임", "동호회"]):
        return [{"id": 5, "name": "모임"}]
    elif any(word in content for word in ["동문", "동창", "졸업"]):
        return [{"id": 6, "name": "동문회"}]
    elif any(word in content for word in ["병원", "의료", "진료", "건강"]):
        return [{"id": 7, "name": "병원"}]
    elif any(word in content for word in ["부동산", "집", "아파트", "매매", "임대"]):
        return [{"id": 8, "name": "부동산"}]
    
    # 4. 어떤 업종도 매칭되지 않으면 "기타"로 분류
    return [{"id": 9, "name": "기타"}]

def _get_metadata_purpose(generation_result: Dict, content: str, original_input: str) -> List[Dict]:
    """메타데이터에서 목적 정보 추출 또는 추론"""
    # 1. 직접 매칭된 템플릿의 목적 사용  
    purpose = generation_result.get("metadata", {}).get("source_info", {}).get("purpose", [])
    if purpose:
        return [{"id": i + 1, "name": p} for i, p in enumerate(purpose)]
    
    # 2. 내용 기반 목적 추론
    purpose_keywords = {
        "공지/안내": ["공지", "안내", "알림", "설명회", "공지사항"],
        "예약알림/리마인드": ["예약", "리마인드", "일정", "예정", "신청"],
        "할인/혜택": ["할인", "혜택", "특가", "세일", "쿠폰"],
        "이벤트/프로모션": ["이벤트", "프로모션", "경품", "참여"],
        "회원 관리": ["회원", "가입", "등급", "포인트"]
    }
    
    detected_purpose = []
    combined_text = content + " " + original_input
    
    for purpose, keywords in purpose_keywords.items():
        if any(keyword in combined_text for keyword in keywords):
            detected_purpose.append(purpose)
    
    if not detected_purpose:
        detected_purpose = ["공지/안내"]
    
    return [{"id": i + 1, "name": purpose} for i, purpose in enumerate(detected_purpose)]


@app.get("/health")
async def health_check():
    """API 상태를 확인합니다."""
    return template_api.health_check()

@app.post("/debug/request")
async def debug_request(request: TemplateCreationRequest):
    """요청 데이터 디버깅용"""
    return {
        "received": {
            "user_id": request.user_id,
            "request_content": request.request_content,
            "content_length": len(request.request_content) if request.request_content else 0
        },
        "status": "OK"
    }

@app.post("/debug/raw")
async def debug_raw(request_data: dict):
    """원시 요청 데이터 디버깅용"""
    return {
        "raw_data": request_data,
        "keys": list(request_data.keys()),
        "status": "OK"
    }


# ===========================================
# 새로운 세션 기반 챗봇 API 엔드포인트들
# ===========================================

@app.post("/ai/templates/{session_id}/variables")
async def update_session_variables(session_id: str, request: VariableUpdateRequest):
    """
    세션의 변수를 개별 업데이트

    Args:
        session_id: 세션 ID
        request: 변수 업데이트 요청

    Returns:
        업데이트 결과 및 현재 세션 상태
    """
    try:
        session_manager = get_session_manager()

        # 세션 유효성 검증
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "SESSION_NOT_FOUND",
                    "세션을 찾을 수 없거나 만료되었습니다.",
                    f"세션 ID: {session_id}"
                )
            )

        # 변수 유효성 검증
        if not request.variables:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "EMPTY_VARIABLES",
                    "업데이트할 변수가 없습니다.",
                    "최소 1개 이상의 변수를 제공해야 합니다."
                )
            )

        # 변수 업데이트
        success = session_manager.update_user_variables(session_id, request.variables)
        if not success:
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "UPDATE_FAILED",
                    "변수 업데이트에 실패했습니다.",
                    "세션 상태를 확인해주세요."
                )
            )

        # 업데이트된 세션 정보 조회
        updated_session = session_manager.get_session(session_id)

        # 미리보기 생성
        preview_generator = get_preview_generator()
        preview_result = preview_generator.generate_preview(updated_session)

        return {
            "success": True,
            "session_id": session_id,
            "updated_variables": list(request.variables.keys()),
            "completion_percentage": updated_session.completion_percentage,
            "remaining_variables": updated_session.missing_variables,
            "preview_snippet": preview_result.get("preview_template", "")[:100] + "..." if len(preview_result.get("preview_template", "")) > 100 else preview_result.get("preview_template", ""),
            "quality_score": preview_result.get("quality_analysis", {}).get("quality_score", 0),
            "next_suggested_variables": preview_result.get("next_suggested_variables", []),
            "session_status": updated_session.status.value,
            "update_count": updated_session.update_count,
            "last_updated": updated_session.last_updated.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "INTERNAL_ERROR",
                "변수 업데이트 중 오류가 발생했습니다.",
                str(e)
            )
        )


@app.get("/ai/templates/{session_id}/preview")
async def get_template_preview(session_id: str, style: str = "missing"):
    """
    부분 완성 템플릿 미리보기 조회

    Args:
        session_id: 세션 ID
        style: 미리보기 스타일 ("missing", "partial", "preview")

    Returns:
        미리보기 템플릿 및 완성도 정보
    """
    try:
        session_manager = get_session_manager()

        # 세션 조회
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "SESSION_NOT_FOUND",
                    "세션을 찾을 수 없거나 만료되었습니다.",
                    f"세션 ID: {session_id}"
                )
            )

        # 템플릿이 설정되어 있는지 확인
        if not session.template_content:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "NO_TEMPLATE",
                    "세션에 템플릿이 설정되지 않았습니다.",
                    "먼저 템플릿을 생성해주세요."
                )
            )

        # 미리보기 생성
        preview_generator = get_preview_generator()
        preview_result = preview_generator.generate_preview(session, style)

        if not preview_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "PREVIEW_GENERATION_FAILED",
                    "미리보기 생성에 실패했습니다.",
                    preview_result.get("error", "알 수 없는 오류")
                )
            )

        # 변수 정보 포맷팅
        missing_variables = []
        for var_info in preview_result.get("next_suggested_variables", []):
            missing_variables.append(SessionVariableInfo(
                variableKey=var_info["variable_key"],
                placeholder=var_info["placeholder"],
                variableType=var_info["variable_type"],
                required=var_info["required"],
                description=var_info.get("description"),
                example=var_info.get("example"),
                inputHint=var_info.get("input_hint"),
                priority=var_info.get("priority", 0)
            ))

        next_suggested = []
        for var_info in preview_result.get("next_suggested_variables", []):
            next_suggested.append(SessionVariableInfo(
                variableKey=var_info["variable_key"],
                placeholder=var_info["placeholder"],
                variableType=var_info["variable_type"],
                required=var_info["required"],
                description=var_info.get("description"),
                example=var_info.get("example"),
                inputHint=var_info.get("input_hint"),
                priority=var_info.get("priority", 0)
            ))

        return SessionPreviewResponse(
            success=True,
            sessionId=session_id,
            previewTemplate=preview_result["preview_template"],
            completionPercentage=round(preview_result["completion_percentage"], 1),
            totalVariables=preview_result["total_variables"],
            completedVariables=preview_result["completed_variables"],
            missingVariables=missing_variables,
            nextSuggestedVariables=next_suggested,
            qualityScore=round(preview_result.get("quality_analysis", {}).get("quality_score", 0), 1),
            estimatedFinalLength=preview_result.get("preview_metadata", {}).get("estimated_final_length", 0),
            readinessForCompletion=preview_result.get("quality_analysis", {}).get("readiness_for_completion", False)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "INTERNAL_ERROR",
                "미리보기 조회 중 오류가 발생했습니다.",
                str(e)
            )
        )


@app.post("/ai/templates/{session_id}/complete")
async def complete_template_session(session_id: str, request: CompleteTemplateRequest):
    """
    세션 템플릿 최종 완성

    Args:
        session_id: 세션 ID
        request: 완성 요청

    Returns:
        최종 완성된 템플릿 데이터
    """
    try:
        session_manager = get_session_manager()

        # 세션 조회
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "SESSION_NOT_FOUND",
                    "세션을 찾을 수 없거나 만료되었습니다.",
                    f"세션 ID: {session_id}"
                )
            )

        # 템플릿 존재 확인
        if not session.template_content:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "NO_TEMPLATE",
                    "완성할 템플릿이 없습니다.",
                    "먼저 템플릿을 생성해주세요."
                )
            )

        # 최종 조정사항 적용 (있는 경우)
        if request.final_adjustments:
            session_manager.update_user_variables(session_id, request.final_adjustments)
            session = session_manager.get_session(session_id)  # 업데이트된 세션 다시 조회

        # 완성도 검증 (강제 완성이 아닌 경우)
        if not request.force_complete and session.completion_percentage < 100.0:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "INCOMPLETE_TEMPLATE",
                    "템플릿이 아직 완성되지 않았습니다.",
                    {
                        "completion_percentage": session.completion_percentage,
                        "missing_variables": session.missing_variables,
                        "suggestion": "누락된 변수를 모두 입력하거나 force_complete=true로 강제 완성하세요."
                    }
                )
            )

        # 최종 템플릿 생성
        preview_generator = get_preview_generator()
        final_preview = preview_generator.generate_preview(session, "missing")
        final_template = final_preview["preview_template"]

        # 기존 TemplateAPI의 export_to_json 활용하여 DB 저장용 데이터 생성
        template_api = get_template_api()

        # 변수 목록을 올바른 형식으로 변환
        variables_for_export = []
        for var_key, var_info in session.template_variables.items():
            variables_for_export.append({
                "variable_key": var_key,
                "placeholder": var_info.placeholder,
                "input_type": var_info.variable_type
            })

        # 생성 결과 구성 (기존 API 형식에 맞춤)
        generation_result = {
            "success": True,
            "template": final_template,
            "variables": variables_for_export,
            "metadata": {
                "source": session.template_source or "generated",
                "session_id": session_id,
                "completion_percentage": session.completion_percentage,
                "quality_score": final_preview.get("quality_analysis", {}).get("quality_score", 0),
                "update_count": session.update_count,
                "creation_time": (session.last_updated - session.created_at).total_seconds() / 60
            }
        }

        # JSON 내보내기
        export_result = template_api.export_to_json(
            result=generation_result,
            user_input=session.original_request,
            user_id=session.user_id
        )

        if not export_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "EXPORT_FAILED",
                    "템플릿 데이터 변환에 실패했습니다.",
                    export_result.get("error", "Unknown error")
                )
            )

        # 세션 완료 처리
        session_manager.complete_session(session_id)

        # 기존 TemplateResponse 형식으로 반환
        exported_data = export_result["data"]
        template_data = exported_data["template"]
        variable_data = exported_data["variables"]

        response_data = {
            "id": 1,  # 임시 ID (실제로는 DB에서 생성)
            "userId": template_data["user_id"],
            "categoryId": template_data["category_id"],
            "title": template_data["title"],
            "content": template_data["content"],
            "imageUrl": template_data["image_url"],
            "type": detect_content_type(template_data["content"]),
            "buttons": _get_metadata_buttons(generation_result, template_data["content"]),
            "variables": [
                {
                    "id": i + 1,
                    "variableKey": var.get("variable_key"),
                    "placeholder": var.get("placeholder"),
                    "inputType": var.get("input_type", "TEXT")
                } for i, var in enumerate(variable_data)
            ],
            "industry": _get_metadata_industries(generation_result, template_data["content"], template_data["category_id"]),
            "purpose": _get_metadata_purpose(generation_result, template_data["content"], session.original_request)
        }

        # 세션 완료 요약 추가
        response_data["session_summary"] = {
            "session_id": session_id,
            "total_updates": session.update_count,
            "completion_time_minutes": round((session.last_updated - session.created_at).total_seconds() / 60, 1),
            "final_completion_percentage": session.completion_percentage,
            "template_source": session.template_source,
            "quality_score": final_preview.get("quality_analysis", {}).get("quality_score", 0)
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "INTERNAL_ERROR",
                "템플릿 완성 중 오류가 발생했습니다.",
                str(e)
            )
        )


# ===========================================
# 세션 관리 및 모니터링 API
# ===========================================

@app.get("/ai/sessions/stats")
async def get_session_stats():
    """세션 통계 조회 (관리용)"""
    session_manager = get_session_manager()
    stats = session_manager.get_stats()
    return {
        "success": True,
        "stats": stats.to_dict(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ai/sessions")
async def get_session_list(limit: int = 20, status: Optional[str] = None):
    """세션 목록 조회 (관리/디버깅용)"""
    session_manager = get_session_manager()

    # 상태 필터 처리
    status_filter = None
    if status:
        try:
            status_filter = SessionStatus(status.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "INVALID_STATUS",
                    f"유효하지 않은 상태값입니다: {status}",
                    "가능한 값: active, completed, expired, error"
                )
            )

    sessions = session_manager.get_session_list(limit=limit, status_filter=status_filter)

    return {
        "success": True,
        "sessions": sessions,
        "total_count": len(sessions),
        "limit": limit,
        "status_filter": status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ai/sessions/{session_id}")
async def get_session_info(session_id: str):
    """개별 세션 정보 조회"""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=create_error_response(
                "SESSION_NOT_FOUND",
                "세션을 찾을 수 없거나 만료되었습니다.",
                f"세션 ID: {session_id}"
            )
        )

    return {
        "success": True,
        "session": session.to_dict(),
        "progress_summary": session.get_progress_summary(),
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/ai/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제 (관리용)"""
    session_manager = get_session_manager()
    success = session_manager.delete_session(session_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=create_error_response(
                "SESSION_NOT_FOUND",
                "삭제할 세션을 찾을 수 없습니다.",
                f"세션 ID: {session_id}"
            )
        )

    return {
        "success": True,
        "message": f"세션 {session_id}이 성공적으로 삭제되었습니다.",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/ai/templates/stream")
async def stream_template_generation(request: TemplateCreationRequest):
    """
    AI를 사용하여 템플릿을 실시간 스트리밍으로 생성
    """
    async def generate_stream():
        try:
            # 요청 검증 (기존 create_template과 동일)
            if not check_rate_limit(request.user_id):
                yield f"data: {json.dumps(create_error_response('RATE_LIMIT_EXCEEDED', '너무 많은 요청을 보내셨습니다.', None, 30), ensure_ascii=False)}\n\n"
                return

            if not request.request_content or not request.request_content.strip():
                yield f"data: {json.dumps(create_error_response('INVALID_INPUT', 'requestContent is required'), ensure_ascii=False)}\n\n"
                return

            if not request.user_id or request.user_id <= 0:
                yield f"data: {json.dumps(create_error_response('INVALID_INPUT', 'userId is required'), ensure_ascii=False)}\n\n"
                return

            if len(request.request_content.strip()) > 500:
                yield f"data: {json.dumps(create_error_response('CONTENT_TOO_LARGE', '입력 텍스트가 너무 깁니다.'), ensure_ascii=False)}\n\n"
                return

            if not is_meaningful_text(request.request_content):
                yield f"data: {json.dumps(create_error_response('INVALID_INPUT', '의미있는 텍스트를 입력해주세요.'), ensure_ascii=False)}\n\n"
                return

            # 진행 상황 스트리밍
            yield f"data: {json.dumps({'status': 'processing', 'message': '요청을 처리하고 있습니다...', 'progress': 10}, ensure_ascii=False)}\n\n"

            processed_content = request.request_content.replace(".", ". ").replace("  ", " ").strip()

            yield f"data: {json.dumps({'status': 'processing', 'message': 'AI 템플릿 생성 중...', 'progress': 30}, ensure_ascii=False)}\n\n"

            # 비동기 템플릿 생성
            generation_result = await template_api.generate_template_async(
                user_input=processed_content,
                conversation_context=request.conversation_context
            )

            yield f"data: {json.dumps({'status': 'processing', 'message': '템플릿 변환 중...', 'progress': 70}, ensure_ascii=False)}\n\n"

            if not generation_result.get("success"):
                # 에러 처리 (기존 로직과 동일)
                error_code = generation_result.get("error_code", "unknown")
                error_message = generation_result.get("error", "Template generation failed")

                if error_code == "INCOMPLETE_INFORMATION":
                    error_response = create_error_response("INCOMPLETE_INFORMATION", "추가 정보가 필요합니다", generation_result.get("reask_data"))
                elif error_code == "PROFANITY_RETRY":
                    error_response = create_error_response("PROFANITY_RETRY", "비속어가 감지되었습니다", generation_result.get("retry_data"))
                else:
                    error_response = create_error_response("TEMPLATE_GENERATION_ERROR", error_message)

                yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
                return

            # JSON 변환
            json_export = template_api.export_to_json(
                result=generation_result,
                user_input=processed_content,
                user_id=request.user_id
            )

            if not json_export.get("success"):
                yield f"data: {json.dumps(create_error_response('JSON_EXPORT_ERROR', '템플릿 데이터 변환에 실패했습니다.'), ensure_ascii=False)}\n\n"
                return

            yield f"data: {json.dumps({'status': 'processing', 'message': '최종 결과 준비 중...', 'progress': 90}, ensure_ascii=False)}\n\n"

            # 최종 응답 준비 (기존 로직과 동일)
            exported_data = json_export["data"]
            template_data = exported_data["template"]
            variable_data = exported_data["variables"]

            response_data = {
                "id": 1,
                "userId": template_data["user_id"],
                "categoryId": template_data["category_id"],
                "title": template_data["title"],
                "content": template_data["content"],
                "imageUrl": template_data["image_url"],
                "type": detect_content_type(template_data["content"]),
                "buttons": _get_metadata_buttons(generation_result, template_data["content"]),
                "variables": [
                    {
                        "id": i + 1,
                        "variableKey": var.get("variable_key"),
                        "placeholder": var.get("placeholder"),
                        "inputType": var.get("input_type", "TEXT")
                    } for i, var in enumerate(variable_data)
                ],
                "industry": _get_metadata_industries(generation_result, template_data["content"], template_data["category_id"]),
                "purpose": _get_metadata_purpose(generation_result, template_data["content"], processed_content)
            }

            # 최종 성공 응답
            yield f"data: {json.dumps({'status': 'completed', 'message': '완료되었습니다!', 'progress': 100, 'result': response_data}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps(create_error_response('UNEXPECTED_ERROR', f'예상치 못한 오류: {str(e)}'), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

