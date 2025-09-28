#!/usr/bin/env python3
"""
카카오 알림톡 변수 형식 검증 도구
반려 사례 기반으로 잘못된 변수 형식을 사전 탐지
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class VariableValidationResult:
    """변수 검증 결과"""
    is_valid: bool
    violations: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH
    fixed_content: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class KakaoVariableValidator:
    """카카오 알림톡 변수 형식 검증 전담 클래스"""

    def __init__(self):
        """변수 검증기 초기화"""
        self._setup_validation_patterns()

    def _setup_validation_patterns(self):
        """검증 패턴 설정 (반려 사례 기반)"""

        # 1. 잘못된 변수 형식 패턴들
        self.invalid_variable_patterns = [
            (r'\[([^\]]+)\]', '대괄호 변수'),  # [전화번호], [URL] 등
            (r'\b\d{3}-\d{4}-\d{4}\b', '고정 전화번호'),  # 010-1234-5678
            (r'\b\d{2}-\d{3,4}-\d{4}\b', '고정 전화번호'),  # 02-123-4567
            (r'\b1588-\d{4}\b', '고정 고객센터 번호'),  # 1588-1234
            (r'\b000-0000-0000\b', '더미 전화번호'),  # 000-0000-0000
            (r'\b123-456-7890\b', '더미 전화번호'),  # 123-456-7890
        ]

        # 2. 고정값 패턴들 (변수로 바뀌어야 함)
        self.fixed_value_patterns = [
            (r'\b2025년 \d{1,2}월 \d{1,2}일\b', '구체적 날짜'),
            (r'\b2025년 0?\d월 0?\d일\b', '구체적 날짜'),
            (r'\b\d+명\b', '구체적 인원수'),
            (r'\b비건 \d+명\b', '구체적 식단 인원'),
            (r'\b할랄 \d+명\b', '구체적 식단 인원'),
            (r'\b휠체어 접근 필요.*?\d+명\b', '구체적 접근성 인원'),
            (r'\b4명 중 2명\b', '구체적 변경 인원'),
            (r'\b오전 \d{1,2}시\b', '구체적 시간'),
            (r'\b오후 \d{1,2}시\b', '구체적 시간'),
        ]

        # 3. 올바른 변수 형식
        self.valid_variable_pattern = r'#\{[^}]+\}'

        # 4. 알림톡 금지 문구 패턴들
        self.prohibited_patterns = [
            (r'\(광고\)', '(광고) 표기'),
            (r'\[광고\]', '[광고] 표기'),
            (r'광고\s*문구', '광고 관련 문구'),
            (r'무료수신거부', '무료수신거부 문구'),
            (r'수신거부.*?번호', '수신거부 번호'),
        ]

        # 5. 변수로 변환해야 할 고정 표현들
        self.fixed_expressions = [
            ('용산 물류센터', '#{물류센터명}'),
            ('강남점', '#{지점명}'),
            ('롯데백화점', '#{백화점명}'),
            ('KB국민카드', '#{카드사명}'),
            ('레노마 셔츠&타이', '#{브랜드명}'),
        ]

    def validate_template_content(self, content: str) -> VariableValidationResult:
        """
        템플릿 내용의 변수 형식 검증

        Args:
            content: 검증할 템플릿 내용

        Returns:
            검증 결과
        """
        violations = []
        risk_level = "LOW"
        fixed_content = content

        # 1. 알림톡 금지 문구 검사 (최우선)
        for pattern, description in self.prohibited_patterns:
            matches = re.findall(pattern, content)
            if matches:
                violations.append(f"알림톡 금지 문구 사용: {description} - '{', '.join(matches)}' (알림톡에서는 사용 불가)")
                risk_level = "HIGH"

        # 2. 잘못된 변수 형식 검사
        for pattern, description in self.invalid_variable_patterns:
            matches = re.findall(pattern, content)
            if matches:
                violations.append(f"{description} 형식 오류: {matches}")
                risk_level = "HIGH"

                # 자동 수정 시도
                if description == '대괄호 변수':
                    for match in matches:
                        old_format = f'[{match}]'
                        new_format = f'#{{{match}}}'
                        fixed_content = fixed_content.replace(old_format, new_format)

        # 3. 고정값 패턴 검사
        for pattern, description in self.fixed_value_patterns:
            matches = re.findall(pattern, content)
            if matches:
                violations.append(f"{description} 고정값 사용: {matches} -> #{{{description}}} 형식 필요")
                if risk_level != "HIGH":
                    risk_level = "MEDIUM"

        # 4. 고정 표현 검사
        for fixed_expr, variable_format in self.fixed_expressions:
            if fixed_expr in content:
                violations.append(f"고정 표현 사용: '{fixed_expr}' -> '{variable_format}' 사용 필요")
                fixed_content = fixed_content.replace(fixed_expr, variable_format)
                if risk_level == "LOW":
                    risk_level = "MEDIUM"

        # 5. 변수 형식 전체 검사
        all_variables = re.findall(r'#\{[^}]+\}|\[[^\]]+\]|\b\d{2,3}-\d{3,4}-\d{4}\b', content)
        valid_variables = re.findall(self.valid_variable_pattern, content)

        if len(all_variables) > len(valid_variables):
            violations.append(f"올바르지 않은 변수 형식 발견. #{{변수명}} 형식만 사용 가능")
            risk_level = "HIGH"

        is_valid = len(violations) == 0

        return VariableValidationResult(
            is_valid=is_valid,
            violations=violations,
            risk_level=risk_level,
            fixed_content=fixed_content if fixed_content != content else None,
            details={
                "invalid_variables_count": len(all_variables) - len(valid_variables),
                "total_variables_count": len(all_variables),
                "auto_fixed": fixed_content != content
            }
        )

    def get_variable_suggestions(self, content: str) -> List[Dict[str, str]]:
        """
        변수 형식 개선 제안 생성

        Args:
            content: 템플릿 내용

        Returns:
            개선 제안 리스트
        """
        suggestions = []

        # 1. 대괄호 변수 개선 제안
        bracket_vars = re.findall(r'\[([^\]]+)\]', content)
        for var in bracket_vars:
            suggestions.append({
                "type": "format_fix",
                "original": f"[{var}]",
                "suggested": f"#{{{var}}}",
                "reason": "대괄호 형식은 허용되지 않습니다. #{} 형식을 사용하세요."
            })

        # 2. 고정값 변수화 제안
        for pattern, description in self.fixed_value_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                suggestions.append({
                    "type": "variable_conversion",
                    "original": match,
                    "suggested": f"#{{{description}}}",
                    "reason": f"구체적인 {description}는 변수로 처리해야 합니다."
                })

        return suggestions

    def is_content_compliant(self, content: str) -> bool:
        """
        간단한 준수 여부 확인

        Args:
            content: 템플릿 내용

        Returns:
            준수 여부
        """
        result = self.validate_template_content(content)
        return result.is_valid

    def get_violation_summary(self, result: VariableValidationResult) -> str:
        """
        위반 사항 요약 메시지 생성

        Args:
            result: 검증 결과

        Returns:
            요약 메시지
        """
        if result.is_valid:
            return "모든 변수 형식이 올바릅니다."

        summary = f"변수 형식 오류 {len(result.violations)}건 발견:\n"
        for violation in result.violations:
            summary += f"- {violation}\n"

        if result.fixed_content:
            summary += "\n자동 수정된 내용을 확인해주세요."

        return summary


# 싱글톤 인스턴스
_kakao_validator_instance: Optional[KakaoVariableValidator] = None


def get_kakao_variable_validator() -> KakaoVariableValidator:
    """전역 카카오 변수 검증기 인스턴스 반환"""
    global _kakao_validator_instance
    if _kakao_validator_instance is None:
        _kakao_validator_instance = KakaoVariableValidator()
    return _kakao_validator_instance


def validate_kakao_variables(content: str) -> VariableValidationResult:
    """
    편의 함수: 카카오 변수 형식 검증

    Args:
        content: 템플릿 내용

    Returns:
        검증 결과
    """
    validator = get_kakao_variable_validator()
    return validator.validate_template_content(content)


if __name__ == "__main__":
    # 테스트
    print("=== 카카오 변수 검증 도구 테스트 ===")

    validator = KakaoVariableValidator()

    # 반려 사례 기반 테스트
    test_cases = [
        "연락처: [전화번호]로 문의하세요",  # 대괄호 오류
        "문의: 000-0000-0000으로 연락하세요",  # 더미 전화번호
        "2025년 09월 24일에 방문하세요",  # 구체적 날짜
        "총 4명 중 2명이 변경됩니다",  # 구체적 인원수
        "#{고객명}님, #{예약일시}에 방문하세요",  # 올바른 형식
    ]

    for i, test_content in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_content}")
        result = validator.validate_template_content(test_content)
        print(f"결과: {'통과' if result.is_valid else '실패'} (위험도: {result.risk_level})")
        if result.violations:
            for violation in result.violations:
                print(f"  - {violation}")
        if result.fixed_content:
            print(f"  수정안: {result.fixed_content}")