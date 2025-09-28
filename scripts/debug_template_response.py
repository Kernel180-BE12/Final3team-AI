#!/usr/bin/env python3
"""
템플릿 생성 응답 디버깅
실제 API 응답 내용을 상세히 분석
"""

import asyncio
import aiohttp
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"
TEMPLATE_ENDPOINT = f"{BASE_URL}/ai/templates"

async def debug_template_response():
    """템플릿 생성 응답 디버깅"""

    test_request = {
        "userId": 1,
        "requestContent": "내일 오후 2시에 강남 스타벅스에서 독서모임이 있습니다. 참가자들에게 알림을 보내주세요"
    }

    print(" 템플릿 생성 응답 디버깅")
    print("=" * 50)
    print(f"요청 데이터: {json.dumps(test_request, ensure_ascii=False, indent=2)}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(TEMPLATE_ENDPOINT, json=test_request) as response:
                response_text = await response.text()

                print(f"\n 응답 정보")
                print(f"Status Code: {response.status}")
                print(f"Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
                print(f"Content-Length: {len(response_text)}")

                print(f"\n Raw Response:")
                print("-" * 30)
                print(response_text)

                try:
                    response_data = json.loads(response_text)
                    print(f"\n Parsed JSON:")
                    print("-" * 30)
                    print(json.dumps(response_data, ensure_ascii=False, indent=2))

                    # 중요 필드들 체크
                    print(f"\n Key Fields Analysis:")
                    print(f"- 'template' field exists: {'template' in response_data}")
                    print(f"- 'variables' field exists: {'variables' in response_data}")
                    print(f"- 'metadata' field exists: {'metadata' in response_data}")
                    print(f"- 'success' field exists: {'success' in response_data}")

                    if 'template' in response_data:
                        template = response_data['template']
                        print(f"- Template type: {type(template)}")
                        print(f"- Template length: {len(str(template)) if template else 0}")
                        if template:
                            print(f"- Template preview: {str(template)[:200]}...")

                    if 'variables' in response_data:
                        variables = response_data['variables']
                        print(f"- Variables type: {type(variables)}")
                        print(f"- Variables count: {len(variables) if variables else 0}")
                        if variables:
                            print(f"- Variables: {variables}")

                except json.JSONDecodeError as e:
                    print(f"\nJSON Parse Error: {e}")

        except Exception as e:
            print(f"Request Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_template_response())