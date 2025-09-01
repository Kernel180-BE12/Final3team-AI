#!/usr/bin/env python3
"""
백엔드 연동 테스트 스크립트
실제 기업 DB 스키마에 맞춰 JSON 데이터 생성 및 전송 테스트
"""

import json
from api import get_template_api

def test_json_generation():
    """JSON 생성 테스트"""
    print("=== 기업 DB 스키마 맞춤 JSON 생성 테스트 ===\n")
    
    # API 인스턴스 생성
    api = get_template_api()
    
    # 테스트 케이스들
    test_cases = [
        "ABC 브랜드 셔츠 고객감사 10% 할인 쿠폰을 발급한다고 알림톡 보내줘",
        "XYZ 백화점에서 신규 브랜드 이월행사 40-60% 할인 안내",
        "DEF 마켓 생활용품 20% 할인쿠폰 광고성 알림톡",
        "GHI 멤버십 회원 대상 헬스케어 혜택 안내",
        "JKL 가구 매장 겨울철 A/S 점검 안내"
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n--- 테스트 케이스 {i} ---")
        print(f"입력: {user_input}")
        
        # 1. 템플릿 생성
        result = api.generate_template(user_input)
        
        if result.get("success"):
            print(" 템플릿 생성 성공")
            
            # 2. JSON 변환 (기업 스키마 - 자동 감지)
            json_data = api.export_to_json(result, user_input, user_id=101)
            
            if json_data.get("success"):
                print(" JSON 변환 성공")
                
                # JSON 구조 출력
                data = json_data["data"]
                template_info = data["template"]
                
                print(f"\n DB 저장용 JSON:")
                print(f"  - 제목: {template_info['title']}")
                print(f"  - 상태: {template_info['status']}")
                print(f"  - 타입: {template_info['type']}")
                print(f"  - 카테고리 ID: {template_info['category_id']}")
                print(f"  - Field: {template_info['field']}")
                print(f"  - 변수 개수: {len(data['variables'])}개")
                print(f"  - 엔티티 개수: {len(data['entities'])}개")
                print(f"  - 업종 ID: {data['industry_mapping']['industry_id'] if data['industry_mapping'] else 'N/A'}")
                print(f"  - 템플릿 길이: {len(template_info['content'])}자")
                
                # 전체 JSON 저장 (선택적)
                with open(f"test_output_case_{i}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  - JSON 파일 저장: test_output_case_{i}.json")
                
            else:
                print(f" JSON 변환 실패: {json_data.get('error')}")
        else:
            print(f" 템플릿 생성 실패: {result.get('error')}")
        
        print("-" * 50)

def test_backend_send_simulation():
    """백엔드 전송 시뮬레이션"""
    print("\n\n=== 백엔드 전송 시뮬레이션 ===\n")
    
    api = get_template_api()
    
    # 테스트용 백엔드 URL (실제로는 존재하지 않음)
    backend_url = "https://your-company-api.com"
    
    user_input = "쿠폰 발급 안내 알림톡 보내줘"
    print(f"입력: {user_input}")
    print(f"백엔드 URL: {backend_url}/api/v1/templates/create")
    
    # 전체 파이프라인 실행 (실제 전송은 실패할 것임)
    result = api.generate_and_send(user_input, backend_url)
    
    print(f"\n 전체 파이프라인 결과:")
    print(f"  - 템플릿 생성: {'' if result['template_generation']['success'] else ''}")
    print(f"  - JSON 변환: {'' if result['json_export']['success'] else ''}")  
    print(f"  - 백엔드 전송: {'' if result['backend_send']['success'] else ''}")
    print(f"  - 전체 성공: {'' if result['overall_success'] else ''}")
    
    if not result['backend_send']['success']:
        print(f"  - 전송 실패 사유: {result['backend_send']['error']}")
        print("  ℹ️  실제 백엔드 연동 시에는 올바른 URL을 사용하세요")

def show_integration_guide():
    """백엔드 개발자를 위한 연동 가이드"""
    print("\n\n=== 백엔드 개발자를 위한 연동 가이드 ===\n")
    
    print(" API 사용법:")
    print("""
    from api import get_template_api
    
    # API 인스턴스 생성
    api = get_template_api()
    
    # 방법 1: 단계별 처리 (자동 업종/카테고리 감지)
    user_input = "쿠폰 발급 안내"
    result = api.generate_template(user_input)
    json_data = api.export_to_json(result, user_input, user_id=101)  # 자동 감지
    send_result = api.send_to_backend(json_data["data"], "https://your-api.com")
    
    # 방법 2: 한번에 처리
    complete_result = api.generate_and_send("쿠폰 발급 안내", "https://your-api.com")
    
    # 방법 3: 수동 지정
    json_data_manual = api.export_to_json(result, user_input, user_id=101, category_id=9101, industry_id=1)
    """)
    
    print("\n📡 백엔드 API 엔드포인트:")
    print("  POST /api/v1/templates/create")
    print("  Content-Type: application/json")
    print("  Accept: application/json")
    
    print("\n JSON 스키마:")
    print("""
    {
      "template": {
        "user_id": 101,
        "category_id": 202, 
        "title": "쿠폰 발급 안내",
        "content": "생성된 템플릿 내용...",
        "status": "CREATE_REQUESTED",
        "type": "MESSAGE",
        "is_public": 0,
        "image_url": null,
        "created_at": "2025-08-31",
        "updated_at": "2025-08-31"
      },
      "entities": [...],
      "variables": [...],
      "metadata": {...}
    }
    """)

if __name__ == "__main__":
    try:
        test_json_generation()
        test_backend_send_simulation() 
        show_integration_guide()
        
    except Exception as e:
        print(f"\n 테스트 실행 중 오류 발생: {e}")
        print("환경 설정을 확인해주세요 (.env 파일, API 키 등)")