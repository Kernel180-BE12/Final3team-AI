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
        사용자 입력에서 5W1H 변수를 추출 (비동기 버전)

        Args:
            query: 사용자 입력 텍스트

        Returns:
            추출된 변수들의 딕셔너리
        """
        from ..utils.llm_provider_manager import ainvoke_llm_with_fallback

        prompt = self._create_extraction_prompt(query)

        try:
            response_text, provider, model = await ainvoke_llm_with_fallback(prompt)
            variables = self._parse_variables(response_text)
            print(f"변수 추출 완료 (비동기) - Provider: {provider}, Model: {model}")
            return variables
        except Exception as e:
            print(f"변수 추출 중 오류 발생 (비동기): {e}")
            return self._get_empty_variables()
    
    def _create_extraction_prompt(self, query: str) -> str:
        """변수 추출용 프롬프트 생성"""
        return f"""
사용자의 알림톡 요청을 분석하여 다음 5W1H 변수들을 추출해주세요.
명시되지 않은 정보는 문맥을 통해 합리적으로 추론하거나, 정말 알 수 없는 경우만 "없음"으로 답변하세요.

변수 정의 및 추론 가이드:
- 누가 (To/Recipient): 메시지 수신자
  → 명시 안된 경우: "고객님", "회원님", "참가자분들" 등으로 추론
- 무엇을 (What/Subject): 알림의 주요 내용이나 이벤트
  → 핵심 키워드에서 추출 (예: "독서모임", "결제 완료", "예약 확인")
- 어떻게 (How/Method): 안내 방식이나 행동 요청
  → "알림톡"이 기본, 추가로 "확인", "참석", "안내" 등
- 언제 (When/Time): 날짜, 시간 정보
  → "내일", "오늘", 구체적 시간 모두 포함
- 어디서 (Where/Place): 장소, 위치 정보
  → 구체적 주소나 일반적 장소명
- 왜 (Why/Reason): 목적이나 이유
  → 명시 안된 경우 "서비스 안내" 등으로 추론

추론 예시:
"내일 12시에 강남 카페에서 독서모임"
→ 누가: "독서모임 참가자분들" (추론)
→ 무엇을: "독서모임"
→ 언제: "내일 12시"
→ 어디서: "강남 카페"

---
요청: "{query}"
---

추출된 변수:
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
        """빈 변수 딕셔너리 반환"""
        return {
            '누가 (To/Recipient)': '없음',
            '무엇을 (What/Subject)': '없음',
            '어떻게 (How/Method)': '없음',
            '언제 (When/Time)': '없음',
            '어디서 (Where/Place)': '없음',
            '왜 (Why/Reason)': '없음'
        }
    
    def validate_variables(self, variables: Dict[str, str]) -> Dict[str, bool]:
        """
        추출된 변수들의 유효성 검증
        
        Args:
            variables: 추출된 변수 딕셔너리
            
        Returns:
            각 변수의 유효성 상태
        """
        validation_result = {}
        for key, value in variables.items():
            validation_result[key] = value != "없음" and len(value.strip()) > 0
        
        return validation_result
    
    def get_missing_variables(self, variables: Dict[str, str]) -> list:
        """
        누락된 변수 목록 반환
        
        Args:
            variables: 추출된 변수 딕셔너리
            
        Returns:
            누락된 변수 키 리스트
        """
        missing = []
        for key, value in variables.items():
            if value == "없음" or len(value.strip()) == 0:
                missing.append(key)
        
        return missing

    def determine_required_variables_by_context(self, user_input: str) -> List[str]:
        """상황별로 필요한 변수 동적 결정"""
        user_input_lower = user_input.lower()

        # 모임/회의/이벤트 - 시간과 장소가 중요
        if any(keyword in user_input_lower for keyword in ['모임', '회의', '미팅', '만남', '행사', '이벤트']):
            return ['무엇을 (What/Subject)', '언제 (When/Time)', '어디서 (Where/Place)']

        # 예약/방문 - 시간과 장소 필수
        elif any(keyword in user_input_lower for keyword in ['예약', '방문', '진료', '상담', '검진']):
            return ['무엇을 (What/Subject)', '언제 (When/Time)', '어디서 (Where/Place)']

        # 쿠폰/할인/이벤트 - 기간이 중요
        elif any(keyword in user_input_lower for keyword in ['쿠폰', '할인', '특가', '세일', '프로모션']):
            return ['무엇을 (What/Subject)', '언제 (When/Time)']

        # 배송/주문 - 대상과 내용이 중요
        elif any(keyword in user_input_lower for keyword in ['배송', '주문', '결제', '구매', '발송']):
            return ['누가 (To/Recipient)', '무엇을 (What/Subject)']

        # 공지/안내 - 내용만 있으면 충분
        elif any(keyword in user_input_lower for keyword in ['공지', '안내', '알림', '공지사항', '안내사항']):
            return ['무엇을 (What/Subject)']

        # 기본값 - 최소한의 필수 정보
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