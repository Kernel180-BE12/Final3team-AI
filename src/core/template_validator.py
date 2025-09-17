#!/usr/bin/env python3
"""
Agent2 4개 Tools 결과 기반 템플릿 검증 시스템
RAGAS 대신 자체 검증 로직으로 템플릿 품질 보장
"""
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ValidationStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"

@dataclass
class ValidationResult:
    """개별 검증 결과"""
    tool_name: str
    status: ValidationStatus
    score: float  # 0.0 ~ 1.0
    issues: List[str]
    details: Dict[str, Any]

@dataclass
class TemplateValidationReport:
    """전체 템플릿 검증 보고서"""
    success: bool
    overall_score: float
    validation_results: List[ValidationResult]
    failed_checks: List[str]
    warnings: List[str]
    recommendation: str
    should_regenerate: bool

class TemplateValidator:
    """
    Agent2의 4개 Tools 결과를 활용한 템플릿 검증 시스템
    """

    def __init__(self):
        """검증 시스템 초기화"""
        # 검증 가중치 (중요도별)
        self.weights = {
            "blacklist": 0.35,    # 가장 중요 (법적 리스크)
            "law": 0.30,          # 두 번째 중요 (법규 준수)
            "guideline": 0.25,    # 세 번째 (가이드라인)
            "whitelist": 0.10     # 상대적으로 낮음 (권장사항)
        }

        # 통과 기준점
        self.pass_threshold = 0.7
        self.regeneration_threshold = 0.5

    def validate_template(self,
                         template: str,
                         tools_results: Dict[str, Any],
                         user_input: str = "") -> TemplateValidationReport:
        """
        템플릿 전체 검증 실행

        Args:
            template: 생성된 템플릿
            tools_results: Agent2의 4개 Tools 실행 결과
            user_input: 원본 사용자 입력

        Returns:
            TemplateValidationReport
        """
        validation_results = []

        # 🔍 DEBUG: Tools 결과 로깅
        print(f"🔍 DEBUG - Tools Results: {tools_results}")

        # 1. BlackList 검증
        blacklist_result = self._validate_blacklist_compliance(
            template, tools_results.get("tools_results", {}).get("blacklist", {}), user_input
        )
        validation_results.append(blacklist_result)

        # 2. WhiteList 검증
        whitelist_result = self._validate_whitelist_usage(
            template, tools_results.get("tools_results", {}).get("whitelist", {}), user_input
        )
        validation_results.append(whitelist_result)

        # 3. Guideline 검증
        guideline_result = self._validate_guideline_compliance(
            template, tools_results.get("tools_results", {}).get("guideline", {}), user_input
        )
        validation_results.append(guideline_result)

        # 4. Law 검증
        law_result = self._validate_law_compliance(
            template, tools_results.get("tools_results", {}).get("law", {}), user_input
        )
        validation_results.append(law_result)

        # 5. 전체 평가
        return self._generate_final_report(template, validation_results, user_input)

    def _validate_blacklist_compliance(self,
                                     template: str,
                                     blacklist_result: Dict,
                                     user_input: str) -> ValidationResult:
        """BlackList Tool 결과 기반 검증"""
        issues = []
        score = 1.0

        # BlackList Tool에서 감지된 위반사항 확인
        compliance_check = blacklist_result.get("compliance_check", "UNKNOWN")
        violations = blacklist_result.get("risk_keywords", [])

        # 🔍 DEBUG: BlackList 결과 로깅
        print(f"🔍 DEBUG - BlackList Result: {blacklist_result}")
        print(f"🔍 DEBUG - compliance_status: {compliance_check}")

        if compliance_check == "FAILED":
            # 치명적 실패
            score = 0.0
            for violation in violations:
                issues.append(f"금지어 사용: {violation}")

        elif compliance_check == "REVIEW_REQUIRED":
            # 검토 필요 (경고 수준)
            score = 0.6
            for violation in violations:
                issues.append(f"주의 표현: {violation}")

        elif compliance_check == "PASSED":
            # 정상 통과
            score = 1.0

        elif compliance_check != "PASSED":
            # 알 수 없는 상태
            score = 0.3
            issues.append("BlackList 검증 상태 불명확")

        # 추가 검증: 템플릿에서 직접 금지어 패턴 재검사
        forbidden_patterns = [
            r'무료.*체험', r'100%.*보장', r'즉시.*승인',
            r'대출.*가능', r'투자.*수익', r'부업.*모집'
        ]

        for pattern in forbidden_patterns:
            if re.search(pattern, template, re.IGNORECASE):
                score = min(score, 0.2)
                issues.append(f"의심스러운 패턴 감지: {pattern}")

        status = ValidationStatus.PASSED if score >= 0.7 else (
            ValidationStatus.WARNING if score >= 0.4 else ValidationStatus.FAILED
        )

        return ValidationResult(
            tool_name="blacklist",
            status=status,
            score=score,
            issues=issues,
            details={
                "original_compliance": compliance_check,
                "violations_count": len(violations),
                "template_length": len(template)
            }
        )

    def _validate_whitelist_usage(self,
                                template: str,
                                whitelist_result: Dict,
                                user_input: str) -> ValidationResult:
        """WhiteList Tool 결과 기반 검증"""
        issues = []
        score = 0.8  # 기본점수 (선택사항이므로)

        approval_status = whitelist_result.get("approval_status", "UNKNOWN")
        approved_terms = whitelist_result.get("approved_terms", [])
        usage_score = whitelist_result.get("usage_score", 0)

        if approval_status == "APPROVED":
            # 승인된 표현 잘 사용
            score = min(1.0, 0.8 + (usage_score / 100 * 0.2))

        elif approval_status == "PARTIAL":
            # 부분적 승인
            score = 0.7
            issues.append("승인된 표현 사용이 부분적임")

        elif approval_status == "NOT_APPROVED":
            # 승인된 표현 미사용
            score = 0.6
            issues.append("승인된 표현을 충분히 활용하지 않음")

        # 추가 검증: 권장 표현 사용도
        recommended_patterns = [
            r'안내.*드립니다', r'확인.*부탁드립니다', r'문의.*주세요',
            r'감사합니다', r'이용해.*주세요'
        ]

        used_patterns = 0
        for pattern in recommended_patterns:
            if re.search(pattern, template, re.IGNORECASE):
                used_patterns += 1

        pattern_bonus = (used_patterns / len(recommended_patterns)) * 0.1
        score = min(1.0, score + pattern_bonus)

        status = ValidationStatus.PASSED if score >= 0.6 else ValidationStatus.WARNING

        return ValidationResult(
            tool_name="whitelist",
            status=status,
            score=score,
            issues=issues,
            details={
                "approval_status": approval_status,
                "approved_terms_count": len(approved_terms),
                "usage_score": usage_score,
                "recommended_patterns_used": used_patterns
            }
        )

    def _validate_guideline_compliance(self,
                                     template: str,
                                     guideline_result: Dict,
                                     user_input: str) -> ValidationResult:
        """Guideline Tool 결과 기반 검증"""
        issues = []
        score = 1.0

        compliance_level = guideline_result.get("compliance_level", "UNKNOWN")
        guideline_issues = guideline_result.get("issues", [])
        recommendations = guideline_result.get("recommendations", [])

        # Guideline Tool 결과에 따른 점수 조정
        if compliance_level == "HIGH":
            score = 1.0
        elif compliance_level == "MEDIUM":
            score = 0.7
            issues.extend([f"가이드라인 이슈: {issue}" for issue in guideline_issues])
        elif compliance_level == "LOW":
            score = 0.4
            issues.extend([f"가이드라인 위반: {issue}" for issue in guideline_issues])
        else:
            score = 0.5
            issues.append("가이드라인 준수 수준 불명확")

        # 추가 검증: 알림톡 기본 구조 확인
        structure_checks = [
            (r'\$\{.*\}', "변수 사용"),
            (r'.{1,90}', "적정 길이 (90자 이내)"),
            (r'[가-힣]', "한글 사용"),
        ]

        structure_score = 0
        for pattern, desc in structure_checks:
            if re.search(pattern, template):
                structure_score += 1
            else:
                issues.append(f"구조 검증 실패: {desc}")

        # 구조 점수 반영
        structure_bonus = (structure_score / len(structure_checks)) * 0.1
        score = min(1.0, score + structure_bonus)

        status = ValidationStatus.PASSED if score >= 0.7 else (
            ValidationStatus.WARNING if score >= 0.5 else ValidationStatus.FAILED
        )

        return ValidationResult(
            tool_name="guideline",
            status=status,
            score=score,
            issues=issues,
            details={
                "compliance_level": compliance_level,
                "issues_count": len(guideline_issues),
                "recommendations_count": len(recommendations),
                "structure_score": structure_score
            }
        )

    def _validate_law_compliance(self,
                               template: str,
                               law_result: Dict,
                               user_input: str) -> ValidationResult:
        """Law Tool 결과 기반 검증"""
        issues = []
        score = 1.0

        compliance_status = law_result.get("compliance_status", "UNKNOWN")
        legal_issues = law_result.get("legal_issues", [])
        risk_level = law_result.get("risk_level", "UNKNOWN")

        # Law Tool 결과에 따른 점수 조정
        if compliance_status == "COMPLIANT":
            if risk_level == "LOW":
                score = 1.0
            elif risk_level == "MEDIUM":
                score = 0.8
            else:
                score = 0.7

        elif compliance_status == "PARTIAL":
            score = 0.5
            issues.extend([f"법적 이슈: {issue}" for issue in legal_issues])

        elif compliance_status == "NON_COMPLIANT":
            score = 0.0
            issues.extend([f"법규 위반: {issue}" for issue in legal_issues])

        else:
            score = 0.3
            issues.append("법규 준수 상태 불명확")

        # 추가 검증: 정보통신법 기본 요구사항
        legal_patterns = [
            (r'광고', "광고 표시 확인"),
            (r'개인정보', "개인정보 처리 관련"),
            (r'수신거부', "수신거부 관련"),
            (r'무료', "무료 표현 검증"),
        ]

        for pattern, desc in legal_patterns:
            if re.search(pattern, template, re.IGNORECASE):
                # 해당 패턴이 있으면 더 엄격한 검증 필요
                if score > 0.8:
                    score = 0.8  # 추가 검토 필요
                issues.append(f"법적 검토 필요: {desc}")

        status = ValidationStatus.PASSED if score >= 0.8 else (
            ValidationStatus.WARNING if score >= 0.5 else ValidationStatus.FAILED
        )

        return ValidationResult(
            tool_name="law",
            status=status,
            score=score,
            issues=issues,
            details={
                "compliance_status": compliance_status,
                "risk_level": risk_level,
                "legal_issues_count": len(legal_issues)
            }
        )

    def _generate_final_report(self,
                             template: str,
                             validation_results: List[ValidationResult],
                             user_input: str) -> TemplateValidationReport:
        """최종 검증 보고서 생성"""

        # 가중평균으로 전체 점수 계산
        weighted_sum = 0
        total_weight = 0

        failed_checks = []
        warnings = []

        for result in validation_results:
            weight = self.weights.get(result.tool_name, 0.25)
            weighted_sum += result.score * weight
            total_weight += weight

            if result.status == ValidationStatus.FAILED:
                failed_checks.append(result.tool_name)

            if result.status == ValidationStatus.WARNING:
                warnings.append(result.tool_name)

            # 이슈들을 경고나 실패로 분류
            for issue in result.issues:
                if result.status == ValidationStatus.FAILED:
                    failed_checks.append(f"{result.tool_name}: {issue}")
                else:
                    warnings.append(f"{result.tool_name}: {issue}")

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0

        # 성공/실패 판정
        success = overall_score >= self.pass_threshold and len(failed_checks) == 0
        should_regenerate = not success  # 통과하지 않으면 무조건 재생성

        # 권장사항 생성
        recommendation = self._generate_recommendation(overall_score, validation_results, success)

        return TemplateValidationReport(
            success=success,
            overall_score=overall_score,
            validation_results=validation_results,
            failed_checks=failed_checks,
            warnings=warnings,
            recommendation=recommendation,
            should_regenerate=should_regenerate
        )

    def _generate_recommendation(self,
                               overall_score: float,
                               validation_results: List[ValidationResult],
                               success: bool) -> str:
        """점수와 결과에 따른 권장사항 생성"""

        if success and overall_score >= 0.9:
            return "✅ 우수: 템플릿이 모든 기준을 만족합니다. 바로 사용 가능합니다."

        elif success and overall_score >= 0.7:
            return "✅ 양호: 템플릿이 기본 요구사항을 만족합니다. 사용 가능합니다."

        elif overall_score >= 0.5:
            critical_issues = [r for r in validation_results if r.tool_name in ['blacklist', 'law'] and r.status == ValidationStatus.FAILED]
            if critical_issues:
                return "⚠️ 주의: 법적/규정 위험이 있습니다. 템플릿 재생성을 권장합니다."
            else:
                return "⚠️ 개선: 일부 가이드라인 미준수가 있습니다. 검토 후 사용하세요."

        else:
            return "❌ 불합격: 템플릿이 여러 기준을 만족하지 않습니다. 반드시 재생성하세요."

# 전역 인스턴스
_template_validator = None

def get_template_validator() -> TemplateValidator:
    """싱글톤 템플릿 검증기 반환"""
    global _template_validator
    if _template_validator is None:
        _template_validator = TemplateValidator()
    return _template_validator


if __name__ == "__main__":
    # 테스트 코드
    print("=== 템플릿 검증 시스템 테스트 ===")

    validator = TemplateValidator()

    # 샘플 템플릿과 Tools 결과
    sample_template = """
안녕하세요, ${고객명}님!

${병원명}에서 ${검진종류} 건강검진 예약이 ${예약일시}에 예정되어 있습니다.

검진 전 주의사항:
- ${주의사항}
- 검진 2시간 전 금식 필요

문의사항이 있으시면 ${연락처}로 연락주세요.

감사합니다.
    """.strip()

    # 샘플 Tools 결과 (실제 Agent2 결과 형태)
    sample_tools_results = {
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

    # 검증 실행
    report = validator.validate_template(
        template=sample_template,
        tools_results=sample_tools_results,
        user_input="건강검진 안내 메시지 만들어줘"
    )

    print(f"검증 결과: {'성공' if report.success else '실패'}")
    print(f"전체 점수: {report.overall_score:.2f}")
    print(f"재생성 필요: {'예' if report.should_regenerate else '아니오'}")
    print(f"권장사항: {report.recommendation}")

    if report.failed_checks:
        print(f"실패 항목: {', '.join(report.failed_checks)}")
    if report.warnings:
        print(f"경고 항목: {', '.join(report.warnings)}")