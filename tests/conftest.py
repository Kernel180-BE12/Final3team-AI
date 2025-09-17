"""
pytest 설정 및 공통 픽스처
"""
import pytest
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_tools_results():
    """Agent2 Tools 결과 샘플 데이터"""
    return {
        "blacklist": {
            "compliance_check": "PASSED",
            "violations": [],
            "data_loaded": True
        },
        "whitelist": {
            "approval_status": "APPROVED",
            "approved_terms": ["안내드립니다", "문의주세요", "감사합니다"],
            "usage_score": 85
        },
        "guideline": {
            "compliance_level": "HIGH",
            "issues": [],
            "recommendations": ["변수명 명확화"]
        },
        "law": {
            "compliance_status": "COMPLIANT",
            "legal_issues": [],
            "risk_level": "LOW"
        }
    }

@pytest.fixture
def failed_tools_results():
    """실패하는 Tools 결과 샘플"""
    return {
        "blacklist": {
            "compliance_check": "FAILED",
            "violations": ["대출", "투자"],
            "data_loaded": True
        },
        "whitelist": {
            "approval_status": "NOT_APPROVED",
            "approved_terms": [],
            "usage_score": 20
        },
        "guideline": {
            "compliance_level": "LOW",
            "issues": ["부적절한 표현", "길이 초과"],
            "recommendations": ["표현 수정", "길이 단축"]
        },
        "law": {
            "compliance_status": "NON_COMPLIANT",
            "legal_issues": ["광고성 내용 미표기"],
            "risk_level": "HIGH"
        }
    }

@pytest.fixture
def sample_good_template():
    """좋은 템플릿 샘플"""
    return """안녕하세요, ${고객명}님!

${병원명}에서 ${검진종류} 건강검진 예약이 ${예약일시}에 예정되어 있습니다.

검진 전 주의사항:
- ${주의사항}

문의사항이 있으시면 ${연락처}로 연락주세요.

감사합니다."""

@pytest.fixture
def sample_bad_template():
    """나쁜 템플릿 샘플"""
    return """🎉 100% 무료 투자 기회! 🎉

대출 즉시 승인! 투자 수익 보장!

지금 바로 클릭하세요!

부업으로 월 1000만원 벌기!"""