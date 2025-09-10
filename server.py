
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# api.py에서 기존 로직을 가져옵니다.
from api import get_template_api

# --- Pydantic 모델 정의 ---
# FastAPI가 JSON의 snake_case <-> camelCase 변환을 자동으로 처리해줍니다.

class TemplateCreationRequest(BaseModel):
    """Spring Boot에서 오는 요청을 위한 모델"""
    user_id: int = Field(..., alias='userId')
    request_content: str = Field(..., alias='requestContent')

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


@app.post("/ai/templates", response_model=TemplateResponse, status_code=200)
async def create_template(request: TemplateCreationRequest):
    """
    AI를 사용하여 템플릿을 생성하고 Spring Boot가 요구하는 형식으로 반환합니다.
    """
    try:
        # 1. api.py의 템플릿 생성 함수 호출
        generation_result = template_api.generate_template(user_input=request.request_content)

        if not generation_result.get("success"):
            # 템플릿 생성 실패 시 (예: 중복)
            if generation_result.get("is_duplicate"):
                 raise HTTPException(status_code=409, detail=generation_result)
            # 기타 실패
            raise HTTPException(status_code=500, detail=generation_result.get("error", "Template generation failed"))

        # 2. 생성된 결과를 DB 저장용 JSON 포맷으로 변환
        json_export = template_api.export_to_json(
            result=generation_result,
            user_input=request.request_content,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

