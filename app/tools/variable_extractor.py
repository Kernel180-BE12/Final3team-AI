"""
Agent1용 변수 추출기
사용자 입력에서 5W1H (Who, What, When, Where, Why, How) 변수를 추출
"""

from typing import Dict, Optional, List
import google.generativeai as genai


class VariableExtractor:
    """Agent1용 변수 추출 클래스"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        초기화
        
        Args:
            api_key: Gemini API 키
            model_name: 사용할 모델명
        """
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def extract_variables(self, query: str) -> Dict[str, str]:
        """
        사용자 입력에서 5W1H 변수를 추출 (동기 버전 - 하위 호환성)

        Args:
            query: 사용자 입력 텍스트

        Returns:
            추출된 변수들의 딕셔너리
        """
        prompt = self._create_extraction_prompt(query)

        try:
            response = self.model.generate_content(prompt)
            variables = self._parse_variables(response.text)
            return variables
        except Exception as e:
            print(f"변수 추출 중 오류 발생: {e}")
            return self._get_empty_variables()

    async def extract_variables_async(self, query: str) -> Dict[str, str]:
        """
        사용자 입력에서 5W1H 변수를 추출 (비동기 버전 - 안전모드)

        Args:
            query: 사용자 입력 텍스트

        Returns:
            추출된 변수들의 딕셔너리 (항상 유효한 값 보장)
        """
        # Import 문을 함수 시작 부분으로 이동하여 import 오류 방지
        import sys
        import os
        from pathlib import Path

        # 프로젝트 루트 경로 찾기
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # src/tools/ -> src/ -> project_root/
        utils_path = project_root / "src" / "utils"

        if str(utils_path) not in sys.path:
            sys.path.insert(0, str(utils_path))

        from app.utils.llm_provider_manager import ainvoke_llm_with_fallback

        prompt = self._create_extraction_prompt(query)

        try:
            response_text, provider, model = await ainvoke_llm_with_fallback(prompt)
            variables = self._parse_variables(response_text)
            print(f"변수 추출 완료 (비동기) - Provider: {provider}, Model: {model}")

            # 🚨 안전모드: 핵심 변수가 누락되면 강제 채우기
            variables = self._apply_safety_mode(variables, query)
            return variables
        except Exception as e:
            print(f"변수 추출 중 오류 발생 (비동기): {e}")
            # 오류 시에도 안전모드 적용
            fallback_variables = self._get_empty_variables()
            return self._apply_safety_mode(fallback_variables, query)
    
    def _create_extraction_prompt(self, query: str) -> str:
        """변수 추출용 프롬프트 생성"""
        return f"""
사용자의 알림톡 요청을 분석하여 다음 5W1H 변수들을 추출해주세요.
**중요: "없음"은 절대 사용하지 마세요. 항상 합리적으로 추론하여 값을 제공하세요.**

변수 정의 및 **적극적 추론** 가이드:
- 누가 (To/Recipient): 메시지 수신자
  → 항상 추론: "고객님", "회원님", "참가자분들", "이용자님" 등
- 무엇을 (What/Subject): 알림의 주요 내용이나 이벤트
  → 요청의 핵심 키워드에서 추출 (예: "독서모임", "예약 확인", "상품 안내")
- 어떻게 (How/Method): 안내 방식이나 행동 요청
  → 기본값: "안내", "알림", "공지" 중 하나 선택
- 언제 (When/Time): 날짜, 시간 정보
  → 명시 안된 경우: "적절한 시간에", "예정된 시간에", "곧"
- 어디서 (Where/Place): 장소, 위치 정보
  → 명시 안된 경우: "지정된 장소에서", "온라인으로", "해당 장소에서"
- 왜 (Why/Reason): 목적이나 이유
  → 기본값: "서비스 제공을 위해", "고객 편의를 위해", "일정 안내를 위해"

**적극적 추론 예시:**
"카페 오픈 이벤트 안내"
→ 누가: "고객님" (추론)
→ 무엇을: "카페 오픈 이벤트"
→ 어떻게: "안내" (추론)
→ 언제: "오픈 예정일에" (추론)
→ 어디서: "새로 오픈하는 카페에서" (추론)
→ 왜: "새로운 서비스 제공을 위해" (추론)

---
요청: "{query}"
---

추출된 변수 (모든 값을 합리적으로 추론하여 채워주세요):
- 누가 (To/Recipient):
- 무엇을 (What/Subject):
- 어떻게 (How/Method):
- 언제 (When/Time):
- 어디서 (Where/Place):
- 왜 (Why/Reason):
        """
    
    def _parse_variables(self, response: str) -> Dict[str, str]:
        """LLM 응답을 파싱하여 변수 딕셔너리 생성"""
        variables = self._get_empty_variables()
        
        lines = response.strip().split('\n')
        for line in lines:
            if ':' in line:
                key_part, value_part = line.split(':', 1)
                key_part = key_part.strip().replace('-', '').strip()
                value_part = value_part.strip()
                
                # 키 매핑
                if '누가' in key_part or 'To' in key_part or 'Recipient' in key_part:
                    variables['누가 (To/Recipient)'] = value_part
                elif '무엇을' in key_part or 'What' in key_part or 'Subject' in key_part:
                    variables['무엇을 (What/Subject)'] = value_part
                elif '어떻게' in key_part or 'How' in key_part or 'Method' in key_part:
                    variables['어떻게 (How/Method)'] = value_part
                elif '언제' in key_part or 'When' in key_part or 'Time' in key_part:
                    variables['언제 (When/Time)'] = value_part
                elif '어디서' in key_part or 'Where' in key_part or 'Place' in key_part:
                    variables['어디서 (Where/Place)'] = value_part
                elif '왜' in key_part or 'Why' in key_part or 'Reason' in key_part:
                    variables['왜 (Why/Reason)'] = value_part
        
        return variables
    
    def _get_empty_variables(self) -> Dict[str, str]:
        """기본값 변수 딕셔너리 반환 (에러 시 사용)"""
        return {
            '누가 (To/Recipient)': '고객님',
            '무엇을 (What/Subject)': '알림',
            '어떻게 (How/Method)': '안내',
            '언제 (When/Time)': '적절한 시간에',
            '어디서 (Where/Place)': '해당 장소에서',
            '왜 (Why/Reason)': '서비스 제공을 위해'
        }
    
    def validate_variables(self, variables: Dict[str, str]) -> Dict[str, bool]:
        """
        추출된 변수들의 유효성 검증 (관대한 검증)

        Args:
            variables: 추출된 변수 딕셔너리

        Returns:
            각 변수의 유효성 상태
        """
        validation_result = {}
        invalid_keywords = ['없음', 'none', 'null', '모름', '알 수 없음']

        for key, value in variables.items():
            # 더 관대한 검증: 어떤 값이든 있으면 유효로 처리
            value_lower = value.lower().strip() if value else ""
            is_valid = (
                len(value_lower) > 0 and
                not any(invalid in value_lower for invalid in invalid_keywords)
            )
            validation_result[key] = is_valid

        return validation_result
    
    def get_missing_variables(self, variables: Dict[str, str]) -> list:
        """
        누락된 변수 목록 반환 (관대한 기준)

        Args:
            variables: 추출된 변수 딕셔너리

        Returns:
            누락된 변수 키 리스트
        """
        missing = []
        invalid_keywords = ['없음', 'none', 'null', '모름', '알 수 없음']

        for key, value in variables.items():
            value_lower = value.lower().strip() if value else ""
            is_missing = (
                len(value_lower) == 0 or
                any(invalid in value_lower for invalid in invalid_keywords)
            )
            if is_missing:
                missing.append(key)

        return missing

    def _apply_safety_mode(self, variables: Dict[str, str], query: str) -> Dict[str, str]:
        """
        안전모드: 변수가 누락되거나 "없음"이면 쿼리에서 강제 추출

        Args:
            variables: 현재 추출된 변수
            query: 원본 사용자 입력

        Returns:
            안전모드가 적용된 변수 딕셔너리
        """
        invalid_keywords = ['없음', 'none', 'null', '모름', '알 수 없음']

        # "무엇을" 변수가 누락되거나 무효하면 쿼리에서 직접 추출
        what_value = variables.get('무엇을 (What/Subject)', '').strip()
        if not what_value or any(invalid in what_value.lower() for invalid in invalid_keywords):
            # 쿼리에서 핵심 키워드 추출
            extracted_what = self._extract_subject_from_query(query)
            variables['무엇을 (What/Subject)'] = extracted_what
            print(f"안전모드: 무엇을 강제 추출 - '{extracted_what}'")

        # 다른 변수들도 기본값 적용
        for key, value in variables.items():
            if not value or any(invalid in value.lower() for invalid in invalid_keywords):
                if key == '누가 (To/Recipient)':
                    variables[key] = '고객님'
                elif key == '어떻게 (How/Method)':
                    variables[key] = '안내'
                elif key == '언제 (When/Time)':
                    variables[key] = '적절한 시간에'
                elif key == '어디서 (Where/Place)':
                    variables[key] = '해당 장소에서'
                elif key == '왜 (Why/Reason)':
                    variables[key] = '서비스 제공을 위해'

        return variables

    def _extract_subject_from_query(self, query: str) -> str:
        """
        쿼리에서 핵심 주제를 직접 추출 (키워드 기반)

        Args:
            query: 사용자 입력

        Returns:
            추출된 주제
        """
        query_lower = query.lower()

        # 핵심 키워드 패턴 매칭 (확장)
        patterns = {
            '독서': '독서모임',
            '할인': '할인 이벤트',
            '세일': '할인 이벤트',
            '이벤트': '이벤트',
            '예약': '예약 확인',
            '주문': '주문 확인',
            '배송': '배송 안내',
            '시스템': '시스템 점검',
            '점검': '점검 안내',
            '부트캠프': '부트캠프 안내',
            '설명회': '설명회',
            '강의': '강의 안내',
            '멤버십': '멤버십',
            '쿠폰': '쿠폰',
            '당첨': '당첨 안내',
            '진료': '진료 예약',
            '카페': '카페 이벤트',
            '출시': '신상품 출시',
            '승급': '등급 승급',
            '리뷰': '리뷰 이벤트',
            '운영시간': '운영시간 변경',
            '조사': '만족도 조사',
            '업데이트': '서비스 업데이트',
            # 추가 키워드
            '축하': '축하 메시지',
            '병원': '병원 서비스',
            '오픈': '오픈 이벤트',
            '온라인': '온라인 서비스',
            '갱신': '갱신 안내',
            '발급': '발급 완료',
            '공지': '공지사항',
            '참여': '참여 안내',
            '신상품': '신상품 출시',
            '정기': '정기 서비스',
            '일정': '일정 안내',
            '변경': '변경 안내',
            '작성': '작성 안내',
            '회원': '회원 서비스',
            '안내': '안내',
            '알림': '알림',
            '확인': '확인',
            '완료': '완료 안내'
        }

        for keyword, subject in patterns.items():
            if keyword in query_lower:
                return subject

        # 패턴이 없으면 원본 쿼리의 첫 번째 단어 사용
        words = query.split()
        if words:
            return f"{words[0]} 안내"

        return "알림"

    def determine_required_variables_by_context(self, user_input: str) -> List[str]:
        """상황별로 필요한 변수 동적 결정 (더 관대한 정책)"""
        user_input_lower = user_input.lower()

        # 모든 경우에 '무엇을'만 필수로 하여 성공률 극대화
        # 다른 변수들은 추론으로 채워지므로 필수에서 제외

        # 모임/회의/이벤트 - 내용만 필수
        if any(keyword in user_input_lower for keyword in ['모임', '회의', '미팅', '만남', '행사', '이벤트']):
            return ['무엇을 (What/Subject)']

        # 예약/방문 - 내용만 필수
        elif any(keyword in user_input_lower for keyword in ['예약', '방문', '진료', '상담', '검진']):
            return ['무엇을 (What/Subject)']

        # 쿠폰/할인/이벤트 - 내용만 필수
        elif any(keyword in user_input_lower for keyword in ['쿠폰', '할인', '특가', '세일', '프로모션']):
            return ['무엇을 (What/Subject)']

        # 배송/주문 - 내용만 필수
        elif any(keyword in user_input_lower for keyword in ['배송', '주문', '결제', '구매', '발송']):
            return ['무엇을 (What/Subject)']

        # 공지/안내 - 내용만 필수
        elif any(keyword in user_input_lower for keyword in ['공지', '안내', '알림', '공지사항', '안내사항']):
            return ['무엇을 (What/Subject)']

        # 기본값 - 내용만 필수
        else:
            return ['무엇을 (What/Subject)']

    def check_mandatory_variables(self, variables: Dict[str, str], user_input: str = "") -> Dict[str, any]:
        """
        상황별 필수 변수 확인 (동적)

        Args:
            variables: 추출된 변수 딕셔너리
            user_input: 사용자 입력 (상황 판단용)

        Returns:
            필수 변수 확인 결과
        """
        # 상황별 필수 변수 결정
        mandatory_vars = self.determine_required_variables_by_context(user_input)
        missing_mandatory = []

        for var in mandatory_vars:
            if variables.get(var, '없음') == '없음' or not variables.get(var, '').strip():
                missing_mandatory.append(var)

        return {
            'is_complete': len(missing_mandatory) == 0,
            'missing_mandatory': missing_mandatory,
            'completeness_score': (len(mandatory_vars) - len(missing_mandatory)) / len(mandatory_vars) if mandatory_vars else 1.0,
            'required_variables': mandatory_vars  # 어떤 변수가 필수였는지 정보 추가
        }


def extract_variables(query: str, api_key: str) -> Dict[str, str]:
    """
    편의 함수: 변수 추출 실행
    
    Args:
        query: 사용자 입력 텍스트
        api_key: Gemini API 키
        
    Returns:
        추출된 변수 딕셔너리
    """
    extractor = VariableExtractor(api_key)
    return extractor.extract_variables(query)


if __name__ == "__main__":
    # 테스트 (API 키가 필요함)
    import os
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY 환경변수를 설정해주세요.")
        exit(1)
    
    extractor = VariableExtractor(api_key)
    
    test_queries = [
        "내일 오후 2시에 강남역에서 김철수씨에게 프로젝트 회의 안내를 보내주세요",
        "다음 주 월요일에 모든 직원들에게 시스템 점검 공지를 전달해주세요",
        "고객님께 주문하신 상품이 준비되었다는 알림을 보내주세요"
    ]
    
    print("=== 변수 추출 테스트 ===")
    for query in test_queries:
        print(f"\n입력: {query}")
        variables = extractor.extract_variables(query)
        validation = extractor.validate_variables(variables)
        mandatory_check = extractor.check_mandatory_variables(variables)
        
        print("추출된 변수:")
        for key, value in variables.items():
            status = "✓" if validation[key] else "✗"
            print(f"  {status} {key}: {value}")
        
        print(f"필수 변수 완성도: {mandatory_check['completeness_score']:.1%}")
        if mandatory_check['missing_mandatory']:
            print(f"누락된 필수 변수: {', '.join(mandatory_check['missing_mandatory'])}")
        
        print("-" * 50)