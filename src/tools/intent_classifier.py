"""
Agent1용 의도 분류기
사용자 입력과 추출된 변수를 바탕으로 의도를 분류
"""

from typing import Dict, List, Optional
import google.generativeai as genai


class IntentClassifier:
    """Agent1용 의도 분류 클래스"""
    
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
        
        # 알림톡 의도 카테고리 정의
        self.intent_categories = {
            # 서비스 관련
            "예약 확인": ["예약", "booking", "reservation", "확인"],
            "주문 확인": ["주문", "order", "구매", "결제", "구입"],
            "배송 안내": ["배송", "delivery", "택배", "발송", "운송"],
            "서비스 안내": ["서비스", "이용", "사용", "가입"],
            
            # 이벤트 관련
            "이벤트 안내": ["이벤트", "event", "행사", "축제", "프로모션"],
            "할인 안내": ["할인", "discount", "세일", "특가", "혜택"],
            "쿠폰 안내": ["쿠폰", "coupon", "적립", "포인트"],
            "경품 안내": ["경품", "선물", "gift", "증정"],
            
            # 시스템 관련
            "시스템 점검": ["점검", "maintenance", "업데이트", "시스템", "서버"],
            "정책 변경": ["정책", "policy", "약관", "규정", "변경"],
            "보안 안내": ["보안", "security", "인증", "비밀번호"],
            
            # 일정 관련
            "회의 안내": ["회의", "meeting", "미팅", "컨퍼런스"],
            "교육 안내": ["교육", "training", "세미나", "워크샵"],
            "방문 안내": ["방문", "visit", "내방", "출장"],
            
            # 고객 관리
            "만료 안내": ["만료", "expire", "소멸", "종료"],
            "갱신 안내": ["갱신", "renewal", "연장", "업데이트"],
            "인증 안내": ["인증", "verification", "확인", "승인"],
            
            # 기타
            "일반 안내": ["안내", "알림", "공지", "정보", "guide"]
        }
    
    def classify_intent(self, query: str, variables: Dict[str, str] = None) -> Dict[str, any]:
        """
        사용자 입력에서 의도를 분류
        
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
            return {'category': '일반 안내', 'confidence': 0.3, 'matched_keywords': []}
        
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
        """LLM 기반 의도 분류"""
        prompt = self._create_classification_prompt(query, variables)
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_llm_response(response.text)
            return result
        except Exception as e:
            print(f"LLM 의도 분류 중 오류 발생: {e}")
            return {'category': '일반 안내', 'confidence': 0.3, 'reasoning': 'LLM 오류로 인한 기본값'}
    
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
        3. 확실하지 않다면 "일반 안내"를 선택하세요
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
            'category': '일반 안내',
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