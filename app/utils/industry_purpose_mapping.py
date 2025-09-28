"""
Industry-Purpose-Category 매핑 테이블
실제 temp_fix.json 데이터를 기반으로 한 동적 매핑 시스템
"""

# 카테고리 코드 정의 (카카오 알림톡 템플릿 카테고리)
CATEGORY_CODES = {
    "RESERVATION": "003001",        # 예약/확인용
    "GENERAL_NOTICE": "004001",     # 일반 안내용 (기본값)
    "PRICE_PAYMENT": "007003",      # 가격/결제용
}

# 실제 데이터에서 확인된 Industries
INDUSTRIES = [
    "학원", "병원", "부동산", "온라인 강의", "피트니스",
    "동문회", "모임", "공연/행사"
]

# 실제 데이터에서 확인된 Purposes
PURPOSES = [
    "공지/안내", "예약", "결제 안내", "회원 관리", "콘텐츠 발송",
    "쿠폰", "행사 신청", "설문", "리포트", "뉴스레터"
]

# 업종별 기본 카테고리 매핑
INDUSTRY_DEFAULT_MAPPING = {
    # 예약이 주요한 업종들
    "병원": "RESERVATION",
    "피트니스": "RESERVATION",

    # 교육/학습 관련
    "학원": "GENERAL_NOTICE",
    "온라인 강의": "GENERAL_NOTICE",

    # 정보/서비스 제공 중심
    "부동산": "GENERAL_NOTICE",  # 주로 정보 제공, 자료 안내, 세미나 등

    # 커뮤니티/그룹
    "동문회": "GENERAL_NOTICE",
    "모임": "GENERAL_NOTICE",
    "공연/행사": "GENERAL_NOTICE",
}

# 목적별 카테고리 매핑 (우선순위 높음)
PURPOSE_CATEGORY_MAPPING = {
    # 예약 관련
    "예약": "RESERVATION",

    # 가격/결제 관련
    "결제 안내": "PRICE_PAYMENT",

    # 일반 안내/공지
    "공지/안내": "GENERAL_NOTICE",
    "회원 관리": "GENERAL_NOTICE",
    "콘텐츠 발송": "GENERAL_NOTICE",
    "쿠폰": "GENERAL_NOTICE",
    "행사 신청": "GENERAL_NOTICE",
    "설문": "GENERAL_NOTICE",
    "리포트": "GENERAL_NOTICE",
    "뉴스레터": "GENERAL_NOTICE",
}

# 특수 조합 매핑 (업종 + 목적)
SPECIAL_COMBINATIONS = {
    # 병원 + 예약 = 예약 카테고리
    ("병원", "예약"): "RESERVATION",

    # 피트니스 + 예약 = 예약 카테고리
    ("피트니스", "예약"): "RESERVATION",

    # 학원 + 결제 = 가격 카테고리
    ("학원", "결제 안내"): "PRICE_PAYMENT",
    ("온라인 강의", "결제 안내"): "PRICE_PAYMENT",
}

def get_category_info(industry_list: list, purpose_list: list) -> dict:
    """
    Industry/Purpose 조합에 따라 CategoryID와 Title을 동적으로 결정
    실제 temp_fix.json 데이터 기반

    Args:
        industry_list: [{"id": 1, "name": "학원"}, ...] or [{"name": "학원"}, ...]
        purpose_list: [{"id": 2, "name": "공지/안내"}, ...] or [{"name": "공지/안내"}, ...]

    Returns:
        {
            "categoryId": "003001",
            "title": "병원 예약 확인",
            "reasoning": "병원 + 예약 조합"
        }
    """
    # 기본값 설정
    default_result = {
        "categoryId": CATEGORY_CODES["GENERAL_NOTICE"],
        "title": "알림톡",
        "reasoning": "기본값 적용"
    }

    # 입력값 검증 및 추출
    if not industry_list or not purpose_list:
        return default_result

    # name 값 추출 (id가 있든 없든 상관없이)
    industry = None
    purpose = None

    if industry_list and isinstance(industry_list[0], dict):
        industry = industry_list[0].get("name", "기타")

    if purpose_list and isinstance(purpose_list[0], dict):
        purpose = purpose_list[0].get("name", "기타")

    if not industry or not purpose:
        return default_result

    # 1. 특수 조합 체크 (최우선)
    special_key = (industry, purpose)
    if special_key in SPECIAL_COMBINATIONS:
        category_type = SPECIAL_COMBINATIONS[special_key]
        category_id = CATEGORY_CODES[category_type]
        title = generate_title(industry, purpose, category_type)
        return {
            "categoryId": category_id,
            "title": title,
            "reasoning": f"특수 조합: {industry} + {purpose}"
        }

    # 2. 목적 우선 매핑 체크
    if purpose in PURPOSE_CATEGORY_MAPPING:
        category_type = PURPOSE_CATEGORY_MAPPING[purpose]
        category_id = CATEGORY_CODES[category_type]
        title = generate_title(industry, purpose, category_type)
        return {
            "categoryId": category_id,
            "title": title,
            "reasoning": f"목적 기반: {purpose}"
        }

    # 3. 업종 기본 매핑 체크
    if industry in INDUSTRY_DEFAULT_MAPPING:
        category_type = INDUSTRY_DEFAULT_MAPPING[industry]
        category_id = CATEGORY_CODES[category_type]
        title = generate_title(industry, purpose, category_type)
        return {
            "categoryId": category_id,
            "title": title,
            "reasoning": f"업종 기반: {industry}"
        }

    # 4. 기본값 반환
    return default_result

def generate_title(industry: str, purpose: str, category_type: str) -> str:
    """
    업종과 목적에 따라 의미있는 제목 생성
    """
    # 카테고리별 제목 템플릿
    title_templates = {
        "RESERVATION": {
            "병원": "병원 예약 확인",
            "부동산": "상담 예약 확인",
            "피트니스": "운동 예약 확인",
            "default": f"{industry} 예약"
        },
        "PRICE_PAYMENT": {
            "학원": "수강료 안내",
            "온라인 강의": "강의료 안내",
            "default": "가격 안내"
        },
        "GENERAL_NOTICE": {
            "학원": "학원 안내",
            "온라인 강의": "강의 안내",
            "동문회": "동문회 소식",
            "모임": "모임 안내",
            "공연/행사": "행사 안내",
            "병원": "병원 안내",
            "부동산": "부동산 안내",
            "피트니스": "운동 안내",
            "default": "알림톡"
        }
    }

    # 해당 카테고리의 템플릿에서 업종별 제목 찾기
    category_templates = title_templates.get(category_type, {})

    if industry in category_templates:
        return category_templates[industry]
    elif "default" in category_templates:
        return category_templates["default"]
    else:
        return "알림톡"

def get_all_mappings() -> dict:
    """
    모든 매핑 정보를 반환 (디버깅/관리용)
    """
    return {
        "category_codes": CATEGORY_CODES,
        "industries": INDUSTRIES,
        "purposes": PURPOSES,
        "industry_defaults": INDUSTRY_DEFAULT_MAPPING,
        "purpose_mappings": PURPOSE_CATEGORY_MAPPING,
        "special_combinations": SPECIAL_COMBINATIONS,
        "total_combinations": len(SPECIAL_COMBINATIONS)
    }

# 사용 예시 및 테스트
if __name__ == "__main__":
    # 실제 데이터 기반 테스트 케이스들
    test_cases = [
        {
            "name": "병원 예약",
            "industry": [{"name": "병원"}],
            "purpose": [{"name": "예약"}],
            "expected": "003001"
        },
        {
            "name": "학원 결제",
            "industry": [{"name": "학원"}],
            "purpose": [{"name": "결제 안내"}],
            "expected": "002001"
        },
        {
            "name": "동문회 공지",
            "industry": [{"name": "동문회"}],
            "purpose": [{"name": "공지/안내"}],
            "expected": "004001"
        },
        {
            "name": "부동산 예약",
            "industry": [{"name": "부동산"}],
            "purpose": [{"name": "예약"}],
            "expected": "003001"
        },
        {
            "name": "기타 기본값",
            "industry": [{"name": "기타"}],
            "purpose": [{"name": "기타"}],
            "expected": "004001"
        }
    ]

    print("=== 매핑 테이블 테스트 (실제 데이터 기반) ===")
    for i, test in enumerate(test_cases, 1):
        result = get_category_info(test["industry"], test["purpose"])
        success = "PASS" if result["categoryId"] == test["expected"] else "FAIL"

        print(f"\n{i}. {success} {test['name']} 테스트:")
        print(f"   입력: {test['industry'][0]['name']} + {test['purpose'][0]['name']}")
        print(f"   결과: {result['categoryId']} ({result['title']})")
        print(f"   이유: {result['reasoning']}")
        print(f"   예상: {test['expected']}")

    print(f"\n=== 매핑 통계 ===")
    mappings = get_all_mappings()
    print(f"지원 업종: {len(mappings['industries'])}개")
    print(f"지원 목적: {len(mappings['purposes'])}개")
    print(f"특수 조합: {mappings['total_combinations']}개")