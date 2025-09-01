"""
Agent1용 변수 추출기
사용자 입력에서 6W (Who, What, How, When, Where, Why) 변수를 추출
"""

from typing import Dict, Optional
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
        사용자 입력에서 6W 변수를 추출
        
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
    
    def _create_extraction_prompt(self, query: str) -> str:
        """변수 추출용 프롬프트 생성"""
        return f"""
        사용자의 요청을 분석하여 다음 변수들을 추출해줘. 각 변수에 대한 정보가 없으면 "없음"이라고 명확히 답변해줘.

        - 누가 (To/Recipient): 수신자, 대상자, 참여자
        - 무엇을 (What/Subject): 주제, 내용, 서비스명, 상품명
        - 어떻게 (How/Method): 방법, 절차, 수단
        - 언제 (When/Time): 날짜, 시간, 기간
        - 어디서 (Where/Place): 장소, 위치, 주소
        - 왜 (Why/Reason): 목적, 이유, 사유

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
    
    def check_mandatory_variables(self, variables: Dict[str, str]) -> Dict[str, any]:
        """
        필수 변수 확인 (누가, 무엇을, 어떻게)
        
        Args:
            variables: 추출된 변수 딕셔너리
            
        Returns:
            필수 변수 확인 결과
        """
        mandatory_vars = ['누가 (To/Recipient)', '무엇을 (What/Subject)', '어떻게 (How/Method)']
        missing_mandatory = []
        
        for var in mandatory_vars:
            if variables.get(var, '없음') == '없음':
                missing_mandatory.append(var)
        
        return {
            'is_complete': len(missing_mandatory) == 0,
            'missing_mandatory': missing_mandatory,
            'completeness_score': (len(mandatory_vars) - len(missing_mandatory)) / len(mandatory_vars)
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