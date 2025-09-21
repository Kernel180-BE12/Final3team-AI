#!/usr/bin/env python3
"""
변수 매핑 도구 (Variable Mapper)

Agent1의 5W1H 변수를 Agent2의 템플릿 변수에 매핑하고,
부족한 변수는 사용자에게 질문하여 수집하는 도구
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Union
from typing_extensions import TypedDict
from app.tools.profanity_checker import get_profanity_checker

# 타입 정의
class TemplateVariable(TypedDict):
    variable_key: str
    placeholder: str
    input_type: str
    required: bool

class MappingResult(TypedDict):
    mapped_variables: Dict[str, str]
    unmapped_variables: List[TemplateVariable]
    mapping_details: List[Dict[str, str]]
    mapping_coverage: float


class VariableMapper:
    """Agent1 변수와 템플릿 변수 간의 매핑 및 수집 관리"""

    def __init__(self):
        """변수 매퍼 초기화"""
        self.profanity_checker = get_profanity_checker()

        # 변수 매핑 규칙 정의 (템플릿 변수 키워드 → Agent1 변수)
        self.mapping_rules = {
            # 사람/대상 관련
            "고객명": ["누가 (To/Recipient)"],
            "수신자": ["누가 (To/Recipient)"],
            "고객": ["누가 (To/Recipient)"],
            "회원명": ["누가 (To/Recipient)"],
            "참가자": ["누가 (To/Recipient)"],
            "이름": ["누가 (To/Recipient)"],

            # 내용/제목 관련
            "제목": ["무엇을 (What/Subject)"],
            "내용": ["무엇을 (What/Subject)"],
            "상품명": ["무엇을 (What/Subject)"],
            "서비스": ["무엇을 (What/Subject)"],
            "이벤트": ["무엇을 (What/Subject)"],
            "모임명": ["무엇을 (What/Subject)"],
            "주문내용": ["무엇을 (What/Subject)"],
            "예약내용": ["무엇을 (What/Subject)"],

            # 시간 관련
            "일시": ["언제 (When/Time)"],
            "날짜": ["언제 (When/Time)"],
            "시간": ["언제 (When/Time)"],
            "예약시간": ["언제 (When/Time)"],
            "시작시간": ["언제 (When/Time)"],
            "마감일": ["언제 (When/Time)"],
            "기간": ["언제 (When/Time)"],

            # 장소 관련
            "장소": ["어디서 (Where/Place)"],
            "위치": ["어디서 (Where/Place)"],
            "주소": ["어디서 (Where/Place)"],
            "매장": ["어디서 (Where/Place)"],
            "카페": ["어디서 (Where/Place)"],
            "지점": ["어디서 (Where/Place)"],
            "건물": ["어디서 (Where/Place)"],

            # 방법/연락처 관련
            "연락처": ["어떻게 (How/Method)"],
            "전화번호": ["어떻게 (How/Method)"],
            "이메일": ["어떻게 (How/Method)"],
            "홈페이지": ["어떻게 (How/Method)"],
            "방법": ["어떻게 (How/Method)"],

            # 이유/목적 관련
            "목적": ["왜 (Why/Reason)"],
            "이유": ["왜 (Why/Reason)"],
            "사유": ["왜 (Why/Reason)"]
        }

    def map_variables(self, agent1_variables: Dict[str, str], template_variables: List[TemplateVariable], method: str = "hybrid") -> MappingResult:
        """
        Agent1 변수를 템플릿 변수에 매핑 (키워드 기반 + LLM 기반 하이브리드)

        Args:
            agent1_variables: Agent1에서 추출된 5W1H 변수
            template_variables: 템플릿에서 추출된 변수 리스트
            method: 매핑 방법 ("keyword", "llm", "hybrid")

        Returns:
            매핑 결과 딕셔너리
        """
        print(f"변수 매핑 시작: Agent1={len(agent1_variables)}개, Template={len(template_variables)}개, Method={method}")

        mapped_variables = {}
        unmapped_template_vars = []
        mapping_details = []

        if method == "llm":
            # LLM만 사용
            mapped_variables = self.llm_map_variables(agent1_variables, template_variables)

            # 매핑되지 않은 변수들 찾기 (빈 문자열이 아닌 것만 성공으로 판단)
            for template_var in template_variables:
                var_key = template_var["variable_key"]
                mapped_value = mapped_variables.get(var_key, "")

                if mapped_value and mapped_value.strip():
                    mapping_details.append({
                        "template_var": var_key,
                        "agent1_source": "LLM 매핑",
                        "value": mapped_value,
                        "mapping_type": "llm"
                    })
                    print(f"  LLM 매핑: {var_key} <- {mapped_value}")
                else:
                    unmapped_template_vars.append(template_var)
                    print(f"  LLM 매핑 실패: {var_key}")

        elif method == "keyword":
            # 기존 키워드 매핑만 사용
            mapped_variables, unmapped_template_vars, mapping_details = self._keyword_mapping_only(agent1_variables, template_variables)

        else:  # hybrid (기본값)
            # 1단계: 키워드 기반 매핑 먼저 시도
            mapped_variables, unmapped_template_vars, mapping_details = self._keyword_mapping_only(agent1_variables, template_variables)

            # 2단계: 매핑되지 않은 변수들 LLM으로 재시도
            if unmapped_template_vars:
                print(f"  키워드 매핑 실패 {len(unmapped_template_vars)}개 변수 -> LLM 재시도")
                llm_result = self.llm_map_variables(agent1_variables, unmapped_template_vars)

                # LLM 결과 통합
                still_unmapped = []
                for template_var in unmapped_template_vars:
                    var_key = template_var["variable_key"]
                    llm_value = llm_result.get(var_key, "")

                    if llm_value and llm_value.strip():
                        mapped_variables[var_key] = llm_value
                        mapping_details.append({
                            "template_var": var_key,
                            "agent1_source": "LLM 매핑 (fallback)",
                            "value": llm_value,
                            "mapping_type": "llm_fallback"
                        })
                        print(f"  LLM fallback 성공: {var_key} <- {llm_value}")
                    else:
                        still_unmapped.append(template_var)
                        print(f"  최종 매핑 실패: {var_key}")

                unmapped_template_vars = still_unmapped

        mapping_coverage = len(mapped_variables) / len(template_variables) if template_variables else 0
        print(f"매핑 완료: {len(mapped_variables)}/{len(template_variables)} ({mapping_coverage:.1%})")

        return {
            "mapped_variables": mapped_variables,
            "unmapped_variables": unmapped_template_vars,
            "mapping_details": mapping_details,
            "mapping_coverage": mapping_coverage
        }

    def _keyword_mapping_only(self, agent1_variables: Dict[str, str], template_variables: List[TemplateVariable]) -> Tuple[Dict[str, str], List[TemplateVariable], List[Dict[str, str]]]:
        """키워드 기반 매핑만 수행 (기존 로직)"""
        mapped_variables = {}
        unmapped_template_vars = []
        mapping_details = []

        for template_var in template_variables:
            var_key = template_var["variable_key"]
            mapped_value = self._find_direct_mapping(var_key, agent1_variables)

            if mapped_value:
                mapped_variables[var_key] = mapped_value
                mapping_details.append({
                    "template_var": var_key,
                    "agent1_source": self._get_mapping_source(var_key, agent1_variables),
                    "value": mapped_value,
                    "mapping_type": "direct"
                })
                print(f"  직접 매핑: {var_key} <- {self._get_mapping_source(var_key, agent1_variables)}")
            else:
                # 유사성 기반 매핑 시도
                similar_value = self._find_similarity_mapping(var_key, agent1_variables)
                if similar_value:
                    mapped_variables[var_key] = similar_value
                    mapping_details.append({
                        "template_var": var_key,
                        "agent1_source": self._get_similarity_source(var_key, agent1_variables),
                        "value": similar_value,
                        "mapping_type": "similarity"
                    })
                    print(f"  유사성 매핑: {var_key} <- {self._get_similarity_source(var_key, agent1_variables)}")
                else:
                    unmapped_template_vars.append(template_var)
                    print(f"  키워드 매핑 실패: {var_key}")

        return mapped_variables, unmapped_template_vars, mapping_details

    def _find_direct_mapping(self, template_var_key: str, agent1_variables: Dict[str, str]) -> Optional[str]:
        """직접 매핑 시도"""
        # 키워드 기반 매핑
        for keyword, agent1_keys in self.mapping_rules.items():
            if keyword in template_var_key.lower():
                for agent1_key in agent1_keys:
                    value = agent1_variables.get(agent1_key, "없음")
                    if value != "없음" and value.strip():
                        return value

        return None

    def _find_similarity_mapping(self, template_var_key: str, agent1_variables: Dict[str, str]) -> Optional[str]:
        """유사성 기반 매핑 시도"""
        template_key_lower = template_var_key.lower()

        for agent1_key, value in agent1_variables.items():
            if value == "없음" or not value.strip():
                continue

            # 유사성 체크
            if any(word in template_key_lower for word in ["시간", "일시", "날짜"]) and "언제" in agent1_key:
                return value
            elif any(word in template_key_lower for word in ["장소", "위치", "주소"]) and "어디서" in agent1_key:
                return value
            elif any(word in template_key_lower for word in ["내용", "제목", "상품"]) and "무엇을" in agent1_key:
                return value
            elif any(word in template_key_lower for word in ["고객", "이름", "회원"]) and "누가" in agent1_key:
                return value

        return None

    def _get_mapping_source(self, template_var_key: str, agent1_variables: Dict[str, str]) -> Optional[str]:
        """매핑 소스 Agent1 키 반환"""
        for keyword, agent1_keys in self.mapping_rules.items():
            if keyword in template_var_key.lower():
                for agent1_key in agent1_keys:
                    if agent1_variables.get(agent1_key, "없음") != "없음":
                        return agent1_key
        return None

    def _get_similarity_source(self, template_var_key: str, agent1_variables: Dict[str, str]) -> Optional[str]:
        """유사성 매핑 소스 반환"""
        template_key_lower = template_var_key.lower()

        for agent1_key, value in agent1_variables.items():
            if value == "없음" or not value.strip():
                continue

            if any(word in template_key_lower for word in ["시간", "일시", "날짜"]) and "언제" in agent1_key:
                return agent1_key
            elif any(word in template_key_lower for word in ["장소", "위치", "주소"]) and "어디서" in agent1_key:
                return agent1_key
            elif any(word in template_key_lower for word in ["내용", "제목", "상품"]) and "무엇을" in agent1_key:
                return agent1_key
            elif any(word in template_key_lower for word in ["고객", "이름", "회원"]) and "누가" in agent1_key:
                return agent1_key

        return None

    def llm_map_variables(self, agent1_variables: Dict[str, str], template_variables: List[TemplateVariable]) -> Dict[str, str]:
        """LLM 기반 변수 매핑 (API/Server에서 가져온 로직)"""
        import json

        if not agent1_variables or not template_variables:
            return {}

        # LLM 호출을 위한 동적 import (순환 import 방지)
        try:
            from app.utils.llm_provider_manager import invoke_llm_with_fallback
        except ImportError:
            print("LLM 매핑을 위한 llm_provider_manager를 찾을 수 없습니다.")
            return {}

        template_var_keys = [var["variable_key"] for var in template_variables if var.get("variable_key")]

        if not template_var_keys:
            return {}

        prompt = f"""다음 Agent1에서 추출한 변수들과 템플릿 변수들을 매핑해주세요.

Agent1 추출 변수:
{json.dumps(agent1_variables, ensure_ascii=False, indent=2)}

템플릿 변수 목록:
{template_var_keys}

지시사항:
1. Agent1 변수의 값을 템플릿 변수에 적절히 매핑해주세요
2. 의미적으로 일치하는 것들을 연결해주세요 (예: "누가" → "수신자명", "무엇을" → "상품명" 등)
3. 매핑되지 않는 템플릿 변수는 빈 문자열("")로 설정하세요
4. JSON 형식으로만 응답해주세요

예시:
{{
  "수신자명": "김영수 고객님",
  "상품명": "아이폰 15",
  "픽업시간": "내일 오후 3시",
  "장소": "강남점"
}}

응답 (JSON만):"""

        try:
            response, _, _ = invoke_llm_with_fallback(prompt=prompt)

            # JSON 응답 파싱
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                mapped_values = json.loads(json_str)

                # 템플릿 변수에 있는 키만 필터링하고 문자열로 변환
                filtered_mapping = {}
                for var_key in template_var_keys:
                    if var_key in mapped_values:
                        value = str(mapped_values[var_key]).strip()
                        filtered_mapping[var_key] = value if value else ""
                    else:
                        filtered_mapping[var_key] = ""

                print(f"LLM 매핑 완료: {len(filtered_mapping)}개 변수")
                return filtered_mapping
            else:
                print(f"LLM 매핑 JSON 파싱 실패: {response}")
                return {var_key: "" for var_key in template_var_keys}

        except Exception as e:
            print(f"LLM 변수 매핑 실패: {e}")
            return {var_key: "" for var_key in template_var_keys}

    def create_collection_request(self, unmapped_variables: List[TemplateVariable]) -> Dict[str, Any]:
        """
        사용자에게 누락된 변수 수집 요청 생성

        Args:
            unmapped_variables: 매핑되지 않은 변수 리스트

        Returns:
            수집 요청 정보
        """
        if not unmapped_variables:
            return {
                "needs_collection": False,
                "message": "모든 변수가 성공적으로 매핑되었습니다.",
                "missing_variables": []
            }

        # 사용자 친화적인 질문 생성
        questions = []
        for var in unmapped_variables:
            var_key = var["variable_key"]
            question = self._generate_question_for_variable(var_key)
            questions.append({
                "variable_key": var_key,
                "question": question,
                "input_type": var.get("input_type", "TEXT"),
                "required": var.get("required", True),
                "validation_rules": self._get_validation_rules(var_key)
            })

        return {
            "needs_collection": True,
            "message": f"{len(unmapped_variables)}개의 추가 정보가 필요합니다.",
            "missing_variables": questions,
            "validation_tools": {
                "profanity": True,
                "blacklist": True,
                "whitelist": True,
                "info_comm": True,
                "template_validator": True
            }
        }

    def _generate_question_for_variable(self, var_key: str) -> str:
        """변수에 맞는 질문 생성"""
        var_key_lower = var_key.lower()

        # 시간 관련
        if any(word in var_key_lower for word in ["시간", "일시", "날짜"]):
            return f"{var_key}을(를) 언제로 설정하시겠습니까? (예: 2024년 1월 15일 14:00)"

        # 장소 관련
        elif any(word in var_key_lower for word in ["장소", "위치", "주소"]):
            return f"{var_key}을(를) 어디로 설정하시겠습니까? (예: 서울시 강남구 테헤란로 123)"

        # 연락처 관련
        elif any(word in var_key_lower for word in ["연락처", "전화", "번호"]):
            return f"{var_key}을(를) 입력해주세요. (예: 010-1234-5678)"

        # 이름 관련
        elif any(word in var_key_lower for word in ["이름", "명", "고객"]):
            return f"{var_key}을(를) 입력해주세요."

        # 기타
        else:
            return f"{var_key}에 대한 정보를 입력해주세요."

    def _get_validation_rules(self, var_key: str) -> List[str]:
        """변수별 검증 규칙 반환"""
        var_key_lower = var_key.lower()
        rules = ["profanity_check"]  # 기본: 비속어 검사

        # 연락처는 형식 검사 추가
        if any(word in var_key_lower for word in ["연락처", "전화", "번호"]):
            rules.append("phone_format")

        # 이메일은 형식 검사 추가
        if "이메일" in var_key_lower or "email" in var_key_lower:
            rules.append("email_format")

        # 모든 변수는 문서 검증 (blacklist, whitelist, info_comm)
        rules.extend(["blacklist_check", "whitelist_check", "info_comm_check", "template_validator_check"])

        return rules

    def validate_user_input(self, var_key: str, user_input: str, validation_rules: List[str] = None) -> Dict[str, Any]:
        """
        사용자 입력 검증

        Args:
            var_key: 변수 키
            user_input: 사용자 입력값
            validation_rules: 적용할 검증 규칙

        Returns:
            검증 결과
        """
        if not validation_rules:
            validation_rules = self._get_validation_rules(var_key)

        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "validated_value": user_input
        }

        # 1. 비속어 검사
        if "profanity_check" in validation_rules:
            profanity_result = self.profanity_checker.check_text(user_input)
            if not profanity_result["is_clean"]:
                results["is_valid"] = False
                results["errors"].append(f"부적절한 언어가 포함되어 있습니다: {', '.join(profanity_result['detected_words'])}")

        # 2. 형식 검사
        if "phone_format" in validation_rules:
            if not self._validate_phone_format(user_input):
                results["warnings"].append("전화번호 형식을 확인해주세요. (예: 010-1234-5678)")

        if "email_format" in validation_rules:
            if not self._validate_email_format(user_input):
                results["warnings"].append("이메일 형식을 확인해주세요.")

        # 3. 빈 값 검사
        if not user_input.strip():
            results["is_valid"] = False
            results["errors"].append("값을 입력해주세요.")

        return results

    def _validate_phone_format(self, phone: str) -> bool:
        """전화번호 형식 검증"""
        patterns = [
            r'^\d{3}-\d{4}-\d{4}$',  # 010-1234-5678
            r'^\d{3}\d{4}\d{4}$',    # 01012345678
            r'^\d{2}-\d{3,4}-\d{4}$' # 02-123-4567
        ]
        return any(re.match(pattern, phone.strip()) for pattern in patterns)

    def _validate_email_format(self, email: str) -> bool:
        """이메일 형식 검증"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email.strip()) is not None


# 편의 함수들
def map_agent1_to_template(agent1_variables: Dict[str, str], template_variables: List[TemplateVariable]) -> MappingResult:
    """
    편의 함수: Agent1 변수를 템플릿 변수에 매핑

    Args:
        agent1_variables: Agent1 추출 변수
        template_variables: 템플릿 변수 리스트

    Returns:
        매핑 결과
    """
    mapper = VariableMapper()
    return mapper.map_variables(agent1_variables, template_variables)


def create_collection_request(unmapped_variables: List[TemplateVariable]) -> Dict[str, Any]:
    """
    편의 함수: 변수 수집 요청 생성

    Args:
        unmapped_variables: 매핑되지 않은 변수들

    Returns:
        수집 요청 정보
    """
    mapper = VariableMapper()
    return mapper.create_collection_request(unmapped_variables)


# 전역 인스턴스 (싱글톤)
_variable_mapper_instance: Optional[VariableMapper] = None


def get_variable_mapper() -> VariableMapper:
    """전역 변수 매퍼 인스턴스 반환"""
    global _variable_mapper_instance
    if _variable_mapper_instance is None:
        _variable_mapper_instance = VariableMapper()
    return _variable_mapper_instance


if __name__ == "__main__":
    # 테스트
    print("=== 변수 매핑 도구 테스트 ===")

    # 테스트 데이터
    agent1_vars = {
        '누가 (To/Recipient)': '김철수님',
        '무엇을 (What/Subject)': '독서모임',
        '언제 (When/Time)': '2024년 1월 15일 14:00',
        '어디서 (Where/Place)': '강남 카페',
        '어떻게 (How/Method)': '없음',
        '왜 (Why/Reason)': '독서 토론을 위해'
    }

    template_vars = [
        {"variable_key": "고객명", "placeholder": "#{고객명}", "input_type": "TEXT", "required": True},
        {"variable_key": "모임명", "placeholder": "#{모임명}", "input_type": "TEXT", "required": True},
        {"variable_key": "일시", "placeholder": "#{일시}", "input_type": "TEXT", "required": True},
        {"variable_key": "장소", "placeholder": "#{장소}", "input_type": "TEXT", "required": True},
        {"variable_key": "연락처", "placeholder": "#{연락처}", "input_type": "TEXT", "required": True}
    ]

    # 매핑 테스트
    result = map_agent1_to_template(agent1_vars, template_vars)
    print(f"매핑 결과: {result['mapped_variables']}")
    print(f"매핑 커버리지: {result['mapping_coverage']:.2%}")
    print(f"누락된 변수 수: {len(result['unmapped_variables'])}")

    # 수집 요청 테스트
    if result['unmapped_variables']:
        collection_request = create_collection_request(result['unmapped_variables'])
        print(f"수집 요청: {collection_request}")