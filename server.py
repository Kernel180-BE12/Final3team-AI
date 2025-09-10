
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import json

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


# --- FastAPI 애플리케이션 설정 ---

app = FastAPI()
# api.py에 정의된 싱글톤 인스턴스를 가져옵니다.
template_api = get_template_api()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/ai/templates" and request.method == "POST":
        body = await request.body()
        print(f"🔍 원시 요청 데이터: {body.decode('utf-8')}")
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
        
        # 요청 필드 검증
        if not request.request_content or not request.request_content.strip():
            raise HTTPException(status_code=400, detail="requestContent is required and cannot be empty")
        
        if not request.user_id or request.user_id <= 0:
            raise HTTPException(status_code=400, detail="userId is required and must be greater than 0")
        
        # 텍스트 전처리 (긴 텍스트에 적절한 공백 추가)
        processed_content = request.request_content.replace(".", ". ").replace("  ", " ").strip()
        print(f"📝 전처리된 내용: {processed_content[:100]}...")
        # 1. api.py의 템플릿 생성 함수 호출 (전처리된 내용 사용)
        generation_result = template_api.generate_template(user_input=processed_content)

        if not generation_result.get("success"):
            # 템플릿 생성 실패 시 (예: 중복)
            if generation_result.get("is_duplicate"):
                 raise HTTPException(status_code=409, detail=generation_result)
            # 기타 실패
            raise HTTPException(status_code=500, detail=generation_result.get("error", "Template generation failed"))

        # 2. 생성된 결과를 DB 저장용 JSON 포맷으로 변환
        json_export = template_api.export_to_json(
            result=generation_result,
            user_input=processed_content,  # 전처리된 내용 사용
            user_id=request.user_id
        )

        if not json_export.get("success"):
            raise HTTPException(status_code=500, detail=json_export.get("error", "Failed to export template to JSON"))

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
                    "inputType": var.get("input_type")
                } for i, var in enumerate(variable_data)
            ],
            "industries": [], # TODO: 업종(Industry) 매핑 로직 필요
            "purposes": [] # TODO: 목적(Purpose) 매핑 로직 필요
        }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

