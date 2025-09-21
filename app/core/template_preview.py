"""
부분 완성 템플릿 미리보기 생성기
확정된 변수는 실제 값으로, 누락 변수는 플레이스홀더로 표시
"""
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .session_models import SessionData, VariableInfo


class TemplatePreviewGenerator:
    """
    템플릿 미리보기 생성기
    - 확정 변수 → 실제 값 치환
    - 누락 변수 → 플레이스홀더 표시
    - 완성도 계산 및 품질 점수 제공
    """

    def __init__(self):
        """미리보기 생성기 초기화"""
        # 플레이스홀더 스타일 설정
        self.placeholder_styles = {
            "missing": "#{{{var_name}}}",           # 누락 변수
            "partial": "[{var_name}]",              # 부분 입력
            "invalid": "⚠️ {var_name} ⚠️",         # 유효하지 않은 값
            "preview": "📝 {var_name}을(를) 입력해주세요"  # 상세 안내
        }

    def generate_preview(self, session: SessionData, preview_style: str = "missing") -> Dict[str, Any]:
        """
        세션 데이터를 기반으로 미리보기 생성

        Args:
            session: 세션 데이터
            preview_style: 미리보기 스타일 ("missing", "partial", "invalid", "preview")

        Returns:
            미리보기 결과 딕셔너리
        """
        if not session.template_content:
            return {
                "success": False,
                "error": "템플릿이 설정되지 않았습니다",
                "preview_template": "",
                "completion_percentage": 0.0
            }

        # 템플릿에서 변수 치환
        preview_template = self._replace_variables(
            template=session.template_content,
            user_variables=session.user_variables,
            template_variables=session.template_variables,
            style=preview_style
        )

        # 완성도 및 품질 분석
        analysis = self._analyze_completion(session)

        # 다음 추천 변수
        next_variables = self._suggest_next_variables(session)

        return {
            "success": True,
            "preview_template": preview_template,
            "completion_percentage": session.completion_percentage,
            "total_variables": len(session.template_variables),
            "completed_variables": len([v for v in session.user_variables.values() if v and v.strip()]),
            "missing_variables": session.missing_variables,
            "next_suggested_variables": next_variables,
            "quality_analysis": analysis,
            "preview_metadata": {
                "preview_length": len(preview_template),
                "estimated_final_length": self._estimate_final_length(session),
                "style_used": preview_style,
                "last_updated": session.last_updated.isoformat(),
                "update_count": session.update_count
            }
        }

    def _replace_variables(self, template: str, user_variables: Dict[str, str],
                          template_variables: Dict[str, VariableInfo], style: str) -> str:
        """
        템플릿의 변수를 치환

        Args:
            template: 원본 템플릿
            user_variables: 사용자 입력 변수
            template_variables: 템플릿 변수 정보
            style: 치환 스타일

        Returns:
            치환된 템플릿
        """
        result_template = template

        # 모든 템플릿 변수에 대해 처리
        for var_key, var_info in template_variables.items():
            placeholder = var_info.placeholder

            # 사용자가 입력한 값이 있는지 확인
            user_value = user_variables.get(var_key, "").strip()

            if user_value:
                # 값 검증
                if self._validate_variable_value(user_value, var_info):
                    # 유효한 값으로 치환
                    result_template = result_template.replace(placeholder, user_value)
                else:
                    # 유효하지 않은 값 표시
                    invalid_placeholder = self.placeholder_styles["invalid"].format(var_name=var_key)
                    result_template = result_template.replace(placeholder, invalid_placeholder)
            else:
                # 누락된 변수 처리
                missing_placeholder = self.placeholder_styles[style].format(var_name=var_key)
                result_template = result_template.replace(placeholder, missing_placeholder)

        return result_template

    def _validate_variable_value(self, value: str, var_info: VariableInfo) -> bool:
        """
        변수 값 유효성 검사

        Args:
            value: 검사할 값
            var_info: 변수 정보

        Returns:
            유효성 여부
        """
        if not value or not value.strip():
            return False

        # 타입별 검증
        if var_info.variable_type == "DATE":
            return self._validate_date(value)
        elif var_info.variable_type == "TIME":
            return self._validate_time(value)
        elif var_info.variable_type == "NUMBER":
            return self._validate_number(value)
        elif var_info.variable_type == "PHONE":
            return self._validate_phone(value)
        elif var_info.variable_type == "EMAIL":
            return self._validate_email(value)
        elif var_info.variable_type == "TEXT":
            return len(value) >= 1 and len(value) <= 100  # 기본 텍스트 검증

        # 패턴 검증 (있는 경우)
        if var_info.validation_pattern:
            try:
                return bool(re.match(var_info.validation_pattern, value))
            except re.error:
                pass

        return True  # 기본적으로 유효하다고 가정

    def _validate_date(self, value: str) -> bool:
        """날짜 형식 검증"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',      # 2024-01-15
            r'\d{4}\.\d{2}\.\d{2}',    # 2024.01.15
            r'\d{2}/\d{2}/\d{4}',      # 01/15/2024
            r'\d{1,2}월\s*\d{1,2}일',   # 1월 15일
            r'(월|화|수|목|금|토|일)요일', # 월요일
            r'(오늘|내일|모레|이번|다음)'   # 상대적 날짜
        ]
        return any(re.search(pattern, value) for pattern in date_patterns)

    def _validate_time(self, value: str) -> bool:
        """시간 형식 검증"""
        time_patterns = [
            r'\d{1,2}:\d{2}',          # 14:30
            r'\d{1,2}시\s*\d{0,2}분?',  # 2시 30분
            r'(오전|오후)\s*\d{1,2}시',  # 오후 2시
            r'\d{1,2}시반'             # 2시반
        ]
        return any(re.search(pattern, value) for pattern in time_patterns)

    def _validate_number(self, value: str) -> bool:
        """숫자 형식 검증"""
        try:
            float(value.replace(',', '').replace('원', '').replace('개', ''))
            return True
        except ValueError:
            return False

    def _validate_phone(self, value: str) -> bool:
        """전화번호 형식 검증"""
        phone_pattern = r'(\d{2,3}-?\d{3,4}-?\d{4})|(\d{10,11})'
        return bool(re.search(phone_pattern, value))

    def _validate_email(self, value: str) -> bool:
        """이메일 형식 검증"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, value))

    def _analyze_completion(self, session: SessionData) -> Dict[str, Any]:
        """
        완성도 분석

        Args:
            session: 세션 데이터

        Returns:
            분석 결과
        """
        total_vars = len(session.template_variables)
        completed_vars = len([v for v in session.user_variables.values() if v and v.strip()])
        missing_vars = len(session.missing_variables)

        # 품질 점수 계산
        quality_score = self._calculate_quality_score(session)

        # 완성 예상 시간
        estimated_completion_time = self._estimate_completion_time(session)

        return {
            "completion_status": self._get_completion_status(session.completion_percentage),
            "quality_score": quality_score,
            "variable_analysis": {
                "total": total_vars,
                "completed": completed_vars,
                "missing": missing_vars,
                "completion_rate": session.completion_percentage
            },
            "validation_status": {
                "valid_variables": self._count_valid_variables(session),
                "invalid_variables": self._count_invalid_variables(session),
                "validation_errors": session.validation_errors
            },
            "estimated_completion_time_minutes": estimated_completion_time,
            "readiness_for_completion": session.completion_percentage >= 100.0
        }

    def _calculate_quality_score(self, session: SessionData) -> float:
        """
        품질 점수 계산 (0-100)

        Args:
            session: 세션 데이터

        Returns:
            품질 점수
        """
        if not session.template_variables:
            return 0.0

        # 기본 완성도 점수 (70%)
        completion_score = session.completion_percentage * 0.7

        # 변수 유효성 점수 (20%)
        valid_count = self._count_valid_variables(session)
        total_user_vars = len([v for v in session.user_variables.values() if v and v.strip()])
        validity_score = (valid_count / total_user_vars * 20) if total_user_vars > 0 else 0

        # 템플릿 길이 적정성 점수 (10%)
        length_score = self._calculate_length_score(session) * 10

        quality_score = completion_score + validity_score + length_score
        return min(100.0, max(0.0, quality_score))

    def _count_valid_variables(self, session: SessionData) -> int:
        """유효한 변수 개수 계산"""
        valid_count = 0
        for var_key, var_value in session.user_variables.items():
            if var_value and var_value.strip():
                var_info = session.template_variables.get(var_key)
                if var_info and self._validate_variable_value(var_value, var_info):
                    valid_count += 1
        return valid_count

    def _count_invalid_variables(self, session: SessionData) -> int:
        """유효하지 않은 변수 개수 계산"""
        total_user_vars = len([v for v in session.user_variables.values() if v and v.strip()])
        valid_count = self._count_valid_variables(session)
        return total_user_vars - valid_count

    def _calculate_length_score(self, session: SessionData) -> float:
        """템플릿 길이 적정성 점수 (0-1)"""
        if not session.template_content:
            return 0.0

        estimated_length = self._estimate_final_length(session)

        # 알림톡 최적 길이: 90자 이하 (1.0점), 200자 이하 (0.8점), 그 이상 (0.5점)
        if estimated_length <= 90:
            return 1.0
        elif estimated_length <= 200:
            return 0.8
        elif estimated_length <= 300:
            return 0.6
        else:
            return 0.4

    def _estimate_final_length(self, session: SessionData) -> int:
        """최종 템플릿 길이 예상"""
        if not session.template_content:
            return 0

        # 현재 템플릿에서 플레이스홀더를 평균 길이로 치환하여 예상
        estimated_template = session.template_content

        for var_key, var_info in session.template_variables.items():
            placeholder = var_info.placeholder
            user_value = session.user_variables.get(var_key, "").strip()

            if user_value:
                # 실제 값으로 치환
                estimated_template = estimated_template.replace(placeholder, user_value)
            else:
                # 예상 평균 길이로 치환
                avg_length = self._get_average_variable_length(var_info.variable_type)
                placeholder_text = "X" * avg_length
                estimated_template = estimated_template.replace(placeholder, placeholder_text)

        return len(estimated_template)

    def _get_average_variable_length(self, variable_type: str) -> int:
        """변수 타입별 평균 길이 반환"""
        type_lengths = {
            "TEXT": 8,      # 일반 텍스트
            "NAME": 6,      # 이름
            "DATE": 12,     # 날짜
            "TIME": 8,      # 시간
            "PHONE": 13,    # 전화번호
            "EMAIL": 20,    # 이메일
            "NUMBER": 5,    # 숫자
            "ADDRESS": 25,  # 주소
            "COMPANY": 10   # 회사명
        }
        return type_lengths.get(variable_type, 8)

    def _suggest_next_variables(self, session: SessionData) -> List[Dict[str, Any]]:
        """
        다음에 입력할 변수 추천

        Args:
            session: 세션 데이터

        Returns:
            추천 변수 목록
        """
        suggestions = []

        # 우선순위: 필수 변수 → 중요 변수 → 나머지
        missing_vars = session.missing_variables

        for var_key in missing_vars:
            var_info = session.template_variables.get(var_key)
            if not var_info:
                continue

            # 우선순위 계산
            priority = self._calculate_variable_priority(var_key, var_info)

            suggestion = {
                "variable_key": var_key,
                "placeholder": var_info.placeholder,
                "variable_type": var_info.variable_type,
                "required": var_info.required,
                "description": var_info.description,
                "example": var_info.example or self._generate_example(var_info),
                "priority": priority,
                "input_hint": self._generate_input_hint(var_info)
            }

            suggestions.append(suggestion)

        # 우선순위로 정렬하여 상위 3개만 반환
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        return suggestions[:3]

    def _calculate_variable_priority(self, var_key: str, var_info: VariableInfo) -> int:
        """변수 우선순위 계산 (높을수록 우선)"""
        priority = 0

        # 필수 변수는 높은 우선순위
        if var_info.required:
            priority += 10

        # 변수 타입별 우선순위
        type_priorities = {
            "NAME": 9,      # 이름 (가장 중요)
            "DATE": 8,      # 날짜
            "TIME": 8,      # 시간
            "PHONE": 7,     # 연락처
            "ADDRESS": 6,   # 장소
            "TEXT": 5       # 기타 텍스트
        }
        priority += type_priorities.get(var_info.variable_type, 3)

        # 변수명 기반 우선순위 (휴리스틱)
        high_priority_keywords = ["이름", "성명", "날짜", "시간", "연락처", "전화"]
        if any(keyword in var_key for keyword in high_priority_keywords):
            priority += 5

        return priority

    def _generate_example(self, var_info: VariableInfo) -> str:
        """변수 타입별 예시 생성"""
        examples = {
            "TEXT": "홍길동",
            "NAME": "김철수",
            "DATE": "2024년 1월 15일",
            "TIME": "오후 2시",
            "PHONE": "010-1234-5678",
            "EMAIL": "example@email.com",
            "NUMBER": "10",
            "ADDRESS": "강남역 스타벅스",
            "COMPANY": "ABC회사"
        }
        return examples.get(var_info.variable_type, "예시값")

    def _generate_input_hint(self, var_info: VariableInfo) -> str:
        """입력 힌트 생성"""
        hints = {
            "TEXT": f"{var_info.key}을(를) 입력해주세요",
            "NAME": "이름을 입력해주세요 (예: 홍길동)",
            "DATE": "날짜를 입력해주세요 (예: 1월 15일, 2024-01-15)",
            "TIME": "시간을 입력해주세요 (예: 오후 2시, 14:30)",
            "PHONE": "연락처를 입력해주세요 (예: 010-1234-5678)",
            "EMAIL": "이메일을 입력해주세요 (예: name@email.com)",
            "NUMBER": "숫자를 입력해주세요",
            "ADDRESS": "주소나 장소를 입력해주세요",
            "COMPANY": "회사명을 입력해주세요"
        }
        return hints.get(var_info.variable_type, f"{var_info.key}을(를) 입력해주세요")

    def _get_completion_status(self, completion_percentage: float) -> str:
        """완성도에 따른 상태 문자열"""
        if completion_percentage >= 100:
            return "완료"
        elif completion_percentage >= 80:
            return "거의완료"
        elif completion_percentage >= 50:
            return "진행중"
        elif completion_percentage >= 20:
            return "시작됨"
        else:
            return "초기상태"

    def _estimate_completion_time(self, session: SessionData) -> float:
        """완성 예상 시간 (분)"""
        missing_count = len(session.missing_variables)

        # 변수당 평균 입력 시간 (분)
        avg_time_per_variable = 0.5

        # 변수 타입별 가중치
        total_time = 0
        for var_key in session.missing_variables:
            var_info = session.template_variables.get(var_key)
            if var_info:
                type_multiplier = {
                    "TEXT": 1.0,
                    "NAME": 0.5,
                    "DATE": 1.2,
                    "TIME": 1.0,
                    "PHONE": 0.8,
                    "EMAIL": 1.5,
                    "ADDRESS": 2.0
                }.get(var_info.variable_type, 1.0)

                total_time += avg_time_per_variable * type_multiplier

        return max(0.5, total_time)  # 최소 30초


# 전역 인스턴스
_preview_generator: Optional[TemplatePreviewGenerator] = None


def get_preview_generator() -> TemplatePreviewGenerator:
    """싱글톤 미리보기 생성기 반환"""
    global _preview_generator
    if _preview_generator is None:
        _preview_generator = TemplatePreviewGenerator()
    return _preview_generator