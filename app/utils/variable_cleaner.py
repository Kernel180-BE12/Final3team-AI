#!/usr/bin/env python3
"""
변수명 정리 유틸리티
Agent가 생성한 변수명에서 불필요한 문자들을 제거
"""

import re
from typing import List, Dict, Any


def clean_variable_name(variable_name: str) -> str:
    """
    변수명에서 불필요한 문자들을 제거하고 정리

    Args:
        variable_name: 원본 변수명

    Returns:
        정리된 변수명
    """
    if not variable_name or not isinstance(variable_name, str):
        return "변수"

    # 1. 앞뒤 공백 제거
    cleaned = variable_name.strip()

    # 2. 대괄호 제거 (문제의 원인)
    cleaned = re.sub(r'^\[+', '', cleaned)  # 시작 부분의 [ 제거
    cleaned = re.sub(r'\]+$', '', cleaned)  # 끝 부분의 ] 제거

    # 3. 중괄호 제거
    cleaned = re.sub(r'^\{+', '', cleaned)  # 시작 부분의 { 제거
    cleaned = re.sub(r'\}+$', '', cleaned)  # 끝 부분의 } 제거

    # 4. # 제거
    cleaned = cleaned.lstrip('#')

    # 5. 특수문자 정리 (단, 한글과 영문은 유지)
    cleaned = re.sub(r'[^\w가-힣]', '', cleaned)

    # 6. 앞뒤 공백 다시 제거
    cleaned = cleaned.strip()

    # 7. 빈 문자열인 경우 기본값 반환
    if not cleaned:
        return "변수"

    return cleaned


def clean_placeholder(variable_name: str) -> str:
    """
    정리된 변수명으로 올바른 placeholder 생성

    Args:
        variable_name: 변수명

    Returns:
        올바른 placeholder 형식 (#{변수명})
    """
    cleaned_name = clean_variable_name(variable_name)
    return f"#{{{cleaned_name}}}"


def clean_variables_list(variables_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    변수 리스트 전체를 정리

    Args:
        variables_list: 원본 변수 리스트

    Returns:
        정리된 변수 리스트
    """
    cleaned_variables = []

    for i, var in enumerate(variables_list):
        if isinstance(var, dict):
            # 원본 variable_key 추출
            original_key = var.get('variable_key', var.get('variableKey', str(var)))

            # 변수명 정리
            cleaned_key = clean_variable_name(original_key)

            # 정리된 변수 정보 생성
            cleaned_var = {
                "id": i + 1,
                "variable_key": cleaned_key,
                "placeholder": clean_placeholder(cleaned_key),
                "input_type": var.get('input_type', var.get('inputType', 'TEXT'))
            }

            cleaned_variables.append(cleaned_var)

        elif isinstance(var, str):
            # 문자열 형태의 변수
            cleaned_key = clean_variable_name(var)

            cleaned_var = {
                "id": i + 1,
                "variable_key": cleaned_key,
                "placeholder": clean_placeholder(cleaned_key),
                "input_type": "TEXT"
            }

            cleaned_variables.append(cleaned_var)

    return cleaned_variables


def clean_template_content(content: str) -> str:
    """
    템플릿 내용에서 잘못된 변수 패턴을 정리

    Args:
        content: 템플릿 내용

    Returns:
        정리된 템플릿 내용
    """
    if not content:
        return content

    # 1. [변수명] 패턴을 #{변수명}으로 변환
    content = re.sub(r'\[([^\]]+)\]', lambda m: f"#{{{clean_variable_name(m.group(1))}}}", content)

    # 2. 잘못된 #{[변수명]} 패턴 정리
    content = re.sub(r'#\{\[([^\]]+)\]\}', lambda m: f"#{{{clean_variable_name(m.group(1))}}}", content)

    # 3. 잘못된 #{변수명]} 패턴 정리
    content = re.sub(r'#\{([^\}]+)\]\}', lambda m: f"#{{{clean_variable_name(m.group(1))}}}", content)

    # 4. 잘못된 #{[변수명} 패턴 정리
    content = re.sub(r'#\{\[([^\}]+)\}', lambda m: f"#{{{clean_variable_name(m.group(1))}}}", content)

    return content


# 테스트 함수
def test_variable_cleaner():
    """변수 정리 테스트"""
    test_cases = [
        "[카페이름",
        "카페연락처]",
        "[고객명]",
        "#{변수}",
        "  공백변수  ",
        "",
        None
    ]

    print("=== Variable Cleaner Test ===")
    for test_case in test_cases:
        cleaned = clean_variable_name(test_case) if test_case else "None"
        placeholder = clean_placeholder(test_case) if test_case else "None"
        print(f"원본: '{test_case}' → 정리: '{cleaned}' → Placeholder: '{placeholder}'")


if __name__ == "__main__":
    test_variable_cleaner()