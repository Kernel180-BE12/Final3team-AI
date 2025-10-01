"""
Agent1용 의도 분류기
사용자 입력과 추출된 변수를 바탕으로 의도를 분류
"""

from typing import Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config.llm_providers import get_llm_manager, LLMProvider


class IntentClassifier:
    """Agent1용 의도 분류 클래스"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        초기화 - LLM 관리자를 통해 최적 LLM 선택

        Args:
            api_key: API 키
            model_name: 폴백용 모델명
        """
        self.api_key = api_key
        self.model_name = model_name

        # LLM 관리자를 통해 LLM 선택
        llm_manager = get_llm_manager()
        primary_config = llm_manager.get_primary_config()

        try:
            if primary_config and primary_config.provider == LLMProvider.OPENAI:
                self.model = ChatOpenAI(
                    model=primary_config.model_name,
                    api_key=primary_config.api_key,
                    temperature=primary_config.temperature,
                    max_tokens=primary_config.max_tokens
                )
                self.provider = "openai"
            elif primary_config and primary_config.provider == LLMProvider.GEMINI:
                self.model = ChatGoogleGenerativeAI(
                    model=primary_config.model_name,
                    google_api_key=primary_config.api_key,
                    temperature=primary_config.temperature
                )
                self.provider = "gemini"
            else:
                # 폴백
                self.model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
                self.provider = "gemini"
        except Exception as e:
            print(f"IntentClassifier LLM 초기화 실패, 폴백 사용: {e}")
            self.model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
            self.provider = "gemini"
        
        # 새로운 카테고리 체계에 맞는 의도 카테고리 정의
        self.intent_categories = {
            # 회원 (001)
            "회원가입": ["회원가입", "가입", "신규", "등록", "회원"],
            "인증/비밀번호/로그인": ["로그인", "비밀번호", "인증", "본인확인", "패스워드"],
            "회원정보/회원혜택": ["회원정보", "회원혜택", "등급", "멤버십", "프로필"],
            
            # 구매 (002)
            "구매완료": ["구매완료", "주문완료", "결제완료", "구매", "주문"],
            "상품가입": ["상품가입", "서비스가입", "신청"],
            "진행상태": ["진행상태", "처리중", "진행현황", "상태"],
            "구매취소": ["구매취소", "주문취소", "결제취소", "취소"],
            "구매예약/입고알림": ["입고알림", "재입고", "예약", "대기"],
            
            # 예약 (003)
            "예약완료/예약내역": ["예약완료", "예약확인", "예약", "booking"],
            "예약상태": ["예약상태", "예약변경", "예약현황"],
            "예약취소": ["예약취소", "예약철회"],
            "예약알림/리마인드": ["예약알림", "리마인드", "알림", "reminder"],
            
            # 서비스이용 (004)
            "이용안내/공지": ["이용안내", "공지", "안내", "공지사항", "notice"],
            "신청접수": ["신청접수", "접수완료", "신청"],
            "처리완료": ["처리완료", "완료알림", "완료"],
            "이용도구": ["이용도구", "도구", "툴"],
            "방문서비스": ["방문서비스", "방문", "출장"],
            "피드백요청": ["피드백요청", "설문", "만족도"],
            "구매감사/이용확인": ["구매감사", "이용확인", "감사"],
            "리마인드": ["리마인드", "알림", "reminder"],
            
            # 리포팅 (005)
            "피드백": ["피드백", "후기", "리뷰", "만족도"],
            "요금청구": ["요금청구", "청구", "billing"],
            "계약/견적": ["계약", "견적", "contract"],
            "안전/피해예방": ["안전", "피해예방", "보안"],
            "뉴스레터": ["뉴스레터", "newsletter", "소식"],
            "거래알림": ["거래알림", "거래", "transaction"],
            
            # 배송 (006)
            "배송상태": ["배송상태", "배송조회", "delivery"],
            "배송예정": ["배송예정", "발송예정", "출고"],
            "배송완료": ["배송완료", "배달완료", "도착"],
            "배송실패": ["배송실패", "배송지연", "재배송"],
            
            # 법적고지 (007)
            "수신동의": ["수신동의", "동의", "consent"],
            "개인정보": ["개인정보", "privacy", "프라이버시"],
            "약관변경": ["약관변경", "정책변경", "terms"],
            "휴면관련": ["휴면", "dormant", "비활성"],
            
            # 업무알림 (008)
            "주문/예약": ["업무알림", "내부알림", "시스템"],
            "내부업무알림": ["내부업무", "업무", "work"],
            
            # 쿠폰/포인트 (009)
            "쿠폰발급": ["쿠폰발급", "쿠폰지급", "쿠폰"],
            "쿠폰사용": ["쿠폰사용", "쿠폰적용"],
            "포인트적립": ["포인트적립", "적립금", "포인트"],
            "포인트사용": ["포인트사용", "포인트소멸"],
            "쿠폰/포인트안내": ["쿠폰안내", "포인트안내", "혜택안내"],
            
            # 기타
            "기타": ["기타", "일반", "etc"]
        }
    
    def classify_intent(self, query: str, variables: Dict[str, str] = None) -> Dict[str, any]:
        """
        사용자 입력에서 의도를 분류 (동기 버전 - 하위 호환성)

        Args:
            query: 사용자 입력 텍스트
            variables: 추출된 변수 (선택사항)

        Returns:
            분류 결과 딕셔너리
        """
        # 1차: 키워드 기반 분류
        keyword_intent = self._classify_by_keywords(query)

        # 2차: LLM 기반 분류
        llm_intent = self._classify_by_llm(query, variables)

        # 3차: 신뢰도 계산 및 최종 결정
        final_intent = self._determine_final_intent(keyword_intent, llm_intent, query)

        return {
            'intent': final_intent['category'],
            'confidence': final_intent['confidence'],
            'keyword_result': keyword_intent,
            'llm_result': llm_intent,
            'reasoning': final_intent['reasoning']
        }

    async def classify_intent_async(self, query: str, variables: Dict[str, str] = None) -> Dict[str, any]:
        """
        사용자 입력에서 의도를 분류 (비동기 버전)

        Args:
            query: 사용자 입력 텍스트
            variables: 추출된 변수 (선택사항)

        Returns:
            분류 결과 딕셔너리
        """
        # 1차: 키워드 기반 분류 (동기)
        keyword_intent = self._classify_by_keywords(query)

        # 2차: LLM 기반 분류 (비동기)
        llm_intent = await self._classify_by_llm_async(query, variables)

        # 3차: 신뢰도 계산 및 최종 결정
        final_intent = self._determine_final_intent(keyword_intent, llm_intent, query)

        return {
            'intent': final_intent['category'],
            'confidence': final_intent['confidence'],
            'keyword_result': keyword_intent,
            'llm_result': llm_intent,
            'reasoning': final_intent['reasoning']
        }
    
    def _classify_by_keywords(self, query: str) -> Dict[str, any]:
        """키워드 기반 의도 분류"""
        query_lower = query.lower()
        scores = {}
        
        for category, keywords in self.intent_categories.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                scores[category] = {
                    'score': score,
                    'matched_keywords': matched_keywords
                }
        
        if not scores:
            return {'category': '기타', 'confidence': 0.3, 'matched_keywords': []}
        
        # 가장 높은 점수의 카테고리 선택
        best_category = max(scores.keys(), key=lambda x: scores[x]['score'])
        max_score = scores[best_category]['score']
        
        # 신뢰도 계산 (매치된 키워드 수 기반)
        confidence = min(max_score * 0.2, 1.0)  # 키워드 1개당 0.2점
        
        return {
            'category': best_category,
            'confidence': confidence,
            'matched_keywords': scores[best_category]['matched_keywords']
        }
    
    def _classify_by_llm(self, query: str, variables: Dict[str, str] = None) -> Dict[str, any]:
        """LLM 기반 의도 분류 (동기 버전 - 하위 호환성)"""
        prompt = self._create_classification_prompt(query, variables)

        try:
            response = self.model.invoke(prompt)
            result = self._parse_llm_response(response.content)
            return result
        except Exception as e:
            print(f"LLM 의도 분류 중 오류 발생: {e}")
            return {'category': '기타', 'confidence': 0.3, 'reasoning': 'LLM 오류로 인한 기본값'}

    async def _classify_by_llm_async(self, query: str, variables: Dict[str, str] = None) -> Dict[str, any]:
        """LLM 기반 의도 분류 (비동기 버전)"""
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

        prompt = self._create_classification_prompt(query, variables)

        try:
            response_text, provider, model = await ainvoke_llm_with_fallback(prompt)
            result = self._parse_llm_response(response_text)
            print(f"의도 분류 완료 (비동기) - Provider: {provider}, Model: {model}")
            return result
        except Exception as e:
            print(f"LLM 의도 분류 중 오류 발생 (비동기): {e}")
            return {'category': '기타', 'confidence': 0.3, 'reasoning': 'LLM 오류로 인한 기본값'}
    
    def _create_classification_prompt(self, query: str, variables: Dict[str, str] = None) -> str:
        """의도 분류용 프롬프트 생성"""
        categories = list(self.intent_categories.keys())
        categories_str = ", ".join(categories)
        
        variables_str = ""
        if variables:
            variables_str = f"\n[추출된 변수]\n"
            for key, value in variables.items():
                if value != "없음":
                    variables_str += f"- {key}: {value}\n"
        
        return f"""
            다음은 카카오 알림톡 템플릿 생성을 위한 사용자 요청입니다.
            사용자의 요청을 분석하여 가장 적절한 의도 카테고리를 선택해주세요.

            [가능한 의도 카테고리]
            {categories_str}

            [사용자 요청]
            {query}
            {variables_str}

            [분석 지침]
            1. 사용자의 핵심 의도를 파악하세요
            2. 위의 카테고리 중 가장 적절한 것을 선택하세요
            3. 확실하지 않다면 "기타"를 선택하세요
            4. 신뢰도를 0.0~1.0 사이로 평가하세요

            [응답 형식]
            의도: [선택한 카테고리]
            신뢰도: [0.0~1.0]
            이유: [선택 근거를 한 줄로]
            """
    
    def _parse_llm_response(self, response: str) -> Dict[str, any]:
        """LLM 응답 파싱"""
        lines = response.strip().split('\n')
        result = {
            'category': '기타',
            'confidence': 0.5,
            'reasoning': 'LLM 응답 파싱 실패'
        }
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if '의도' in key:
                    result['category'] = value
                elif '신뢰도' in key:
                    try:
                        result['confidence'] = float(value)
                    except ValueError:
                        result['confidence'] = 0.5
                elif '이유' in key:
                    result['reasoning'] = value
        
        return result
    
    def _determine_final_intent(self, keyword_result: Dict, llm_result: Dict, query: str) -> Dict[str, any]:
        """키워드와 LLM 결과를 종합하여 최종 의도 결정"""
        
        # 두 결과가 일치하는 경우
        if keyword_result['category'] == llm_result['category']:
            confidence = min((keyword_result['confidence'] + llm_result['confidence']) / 2 + 0.2, 1.0)
            return {
                'category': keyword_result['category'],
                'confidence': confidence,
                'reasoning': f"키워드와 LLM 분석 결과 일치: {llm_result.get('reasoning', '')}"
            }
        
        # 신뢰도 기반 선택
        if keyword_result['confidence'] > llm_result['confidence']:
            return {
                'category': keyword_result['category'],
                'confidence': keyword_result['confidence'],
                'reasoning': f"키워드 매칭 우선 (매치: {', '.join(keyword_result['matched_keywords'])})"
            }
        else:
            return {
                'category': llm_result['category'],
                'confidence': llm_result['confidence'],
                'reasoning': f"LLM 분석 우선: {llm_result.get('reasoning', '')}"
            }
    
    def get_intent_suggestions(self, query: str) -> List[Dict[str, any]]:
        """
        의도 분류 후보들을 신뢰도 순으로 반환
        
        Args:
            query: 사용자 입력 텍스트
            
        Returns:
            의도 후보 리스트 (신뢰도 내림차순)
        """
        keyword_results = {}
        query_lower = query.lower()
        
        # 모든 카테고리에 대해 점수 계산
        for category, keywords in self.intent_categories.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                confidence = min(score * 0.15, 0.9)
                keyword_results[category] = {
                    'category': category,
                    'confidence': confidence,
                    'matched_keywords': matched_keywords
                }
        
        # 신뢰도 순으로 정렬
        suggestions = sorted(keyword_results.values(), key=lambda x: x['confidence'], reverse=True)
        
        # 상위 5개만 반환
        return suggestions[:5]


def classify_intent(query: str, api_key: str, variables: Dict[str, str] = None) -> Dict[str, any]:
    """
    편의 함수: 의도 분류 실행
    
    Args:
        query: 사용자 입력 텍스트
        api_key: Gemini API 키
        variables: 추출된 변수 (선택사항)
        
    Returns:
        분류 결과 딕셔너리
    """
    classifier = IntentClassifier(api_key)
    return classifier.classify_intent(query, variables)


if __name__ == "__main__":
    # 테스트 (API 키가 필요함)
    import os
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY 환경변수를 설정해주세요.")
        exit(1)
    
    classifier = IntentClassifier(api_key)
    
    test_queries = [
        "내일 시스템 점검이 있다고 고객들에게 알려주세요",
        "주문하신 상품이 배송 준비 중이라고 안내해주세요",
        "다음 주 화요일에 중요한 회의가 있습니다",
        "할인 이벤트를 고객들에게 알려주세요",
        "포인트가 내일까지만 유효하다고 안내해주세요"
    ]
    
    print("=== 의도 분류 테스트 ===")
    for query in test_queries:
        print(f"\n입력: {query}")
        result = classifier.classify_intent(query)
        suggestions = classifier.get_intent_suggestions(query)
        
        print(f"분류된 의도: {result['intent']}")
        print(f"신뢰도: {result['confidence']:.2f}")
        print(f"근거: {result['reasoning']}")
        
        if suggestions:
            print("의도 후보들:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"  {i}. {suggestion['category']} (신뢰도: {suggestion['confidence']:.2f})")
        
        print("-" * 50)