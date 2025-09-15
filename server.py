
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import time
from datetime import datetime
import re
from collections import defaultdict

# api.py에서 기존 로직을 가져옵니다.
from api import get_template_api

# --- Pydantic 모델 정의 ---
# FastAPI가 JSON의 snake_case <-> camelCase 변환을 자동으로 처리해줍니다.

class TemplateCreationRequest(BaseModel):
    """Spring Boot에서 오는 요청을 위한 모델"""
    user_id: int = Field(..., alias='userId')
    request_content: str = Field(..., alias='requestContent')
    
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
    industries: List[Industry]
    purposes: List[Purpose]


# --- 유틸리티 함수 ---

# Rate limiting을 위한 메모리 저장소
request_counts = defaultdict(list)
RATE_LIMIT_PER_MINUTE = 10
RATE_LIMIT_WINDOW = 60  # seconds

def create_error_response(code: str, message: str, details: Optional[str] = None, retry_after: Optional[int] = None) -> Dict[str, Any]:
    """표준화된 에러 응답 생성"""
    error_response = {
        "error": {
            "code": code,
            "message": message
        },
        "timestamp": datetime.now().isoformat() + "Z"
    }
    if details:
        error_response["error"]["details"] = details
    if retry_after:
        error_response["retryAfter"] = retry_after
    return error_response

def is_meaningful_text(text: str) -> bool:
    """의미있는 텍스트인지 판별"""
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

def check_rate_limit(user_id: int) -> bool:
    """사용자별 요청 제한 확인"""
    current_time = time.time()
    user_requests = request_counts[user_id]
    
    # 1분 이전 요청들 제거
    request_counts[user_id] = [req_time for req_time in user_requests 
                              if current_time - req_time < RATE_LIMIT_WINDOW]
    
    # 현재 요청 수 확인
    if len(request_counts[user_id]) >= RATE_LIMIT_PER_MINUTE:
        return False
    
    # 현재 요청 추가
    request_counts[user_id].append(current_time)
    return True

# --- FastAPI 애플리케이션 설정 ---

app = FastAPI()
# api.py에 정의된 싱글톤 인스턴스를 가져옵니다.
template_api = get_template_api()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/ai/templates" and request.method == "POST":
        body = await request.body()
        print(f"🔍 Content-Type: {request.headers.get('content-type')}")
        print(f"🔍 Content-Length: {request.headers.get('content-length')}")
        print(f"🔍 Body Length: {len(body)} bytes")
        print(f"🔍 원시 요청 데이터: '{body.decode('utf-8')}'")
        
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
        print(f"📥 받은 요청: user_id={request.user_id}, content={request.request_content}")
        
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
            
        # 의미있는 텍스트인지 검증
        if not is_meaningful_text(request.request_content):
            raise HTTPException(
                status_code=400, 
                detail=create_error_response(
                    "INVALID_INPUT",
                    "입력 내용이 올바르지 않습니다. 의미있는 텍스트를 입력해주세요.",
                    "특수문자만 입력되었거나 공백만 포함된 요청입니다."
                )
            )
        
        # 텍스트 전처리 (긴 텍스트에 적절한 공백 추가)
        processed_content = request.request_content.replace(".", ". ").replace("  ", " ").strip()
        print(f"📝 전처리된 내용: {processed_content[:100]}...")
        # 1. api.py의 템플릿 생성 함수 호출 (전처리된 내용 사용)
        generation_result = template_api.generate_template(user_input=processed_content)

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
            
            # 에러 타입별 처리
            error_type = generation_result.get("error_type", "unknown")
            error_message = generation_result.get("error", "Template generation failed")
            
            if error_type == "profanity":
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

        # 3. 최종 응답 모델(TemplateResponse)에 맞춰 데이터 매핑
        # 참고: 현재 로직은 buttons, industries, purposes를 생성하지 않으므로, 빈 리스트로 반환합니다.
        #      이 부분은 필요 시 추가 로직 구현이 필요합니다.
        #      id는 DB 저장 후 실제 ID를 받아와야 하지만, 여기서는 예시로 1을 사용합니다.
        response_data = {
            "id": 1, # 임시 ID
            "userId": template_data["user_id"],
            "categoryId": template_data["category_id"],
            "title": template_data["title"],
            "content": template_data["content"],
            "imageUrl": template_data["image_url"],
            "type": "MESSAGE", # api.py에서 "MESSAGE"로 고정되어 있음
            "buttons": [], # TODO: 버튼 생성 로직 필요
            "variables": [
                {
                    "id": i + 1, # 임시 ID
                    "variableKey": var.get("variable_key"),
                    "placeholder": "#" + var.get("placeholder"), 
                    "inputType": var.get("input_type", "TEXT")  # 기본값 설정
                } for i, var in enumerate(variable_data)
            ],
            "industries": [], # TODO: 업종(Industry) 매핑 로직 필요
            "purposes": [] # TODO: 목적(Purpose) 매핑 로직 필요
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

