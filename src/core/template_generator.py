import re
import os
import faiss
import numpy as np
import pickle
from datetime import date, timedelta
from typing import Dict, List, Tuple
from .base_processor import BaseTemplateProcessor


# prototype.py에서 가져온 클래스들
class VariableExtractor:
    def __init__(self, llm):
        self.llm = llm
    
    def extract(self, query):
        """변수 추출 (prototype.py 스타일)"""
        prompt = f'''
        사용자의 요청을 분석하여 다음 변수들을 추출해줘. 각 변수에 대한 정보가 없으면 "없음"이라고 명확히 답변해줘.

        - 누가 (To/Recipient):
        - 무엇을 (What/Subject):
        - 어떻게 (How/Method):
        - 언제 (When/Time):
        - 어디서 (Where/Place):
        - 왜 (Why/Reason):

        ---
        요청: "{query}"
        ---

        추출된 변수:
        '''
        
        response = self.llm.generate_with_gemini(prompt)
        return self._parse_variables(response)
    
    def _parse_variables(self, response: str) -> dict:
        variables = {}
        lines = response.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace('-', '').strip()
                variables[key] = value.strip()
        return variables

class IntentRecognizer:
    def __init__(self, llm):
        self.llm = llm
    
    def recognize(self, query, variables):
        """의도 파악 (prototype.py 스타일)"""
        prompt = f'''
        사용자의 요청과 추출된 변수를 바탕으로, 사용자의 핵심 의도를 한두 단어의 일반적인 카테고리로 분류해줘.
        (예: 이벤트 안내, 시스템 점검, 주문 확인, 포인트 소멸 안내, 컨퍼런스 안내)

        ---
        요청: "{query}"
        변수: {variables}
        ---

        핵심 의도:
        '''
        
        intent = self.llm.generate_with_gemini(prompt).strip()
        return intent

class TemplateGenerator(BaseTemplateProcessor):
    """prototype.py 스타일로 완전 개선된 템플릿 생성 클래스"""

    def __init__(self, api_key: str, gemini_model: str = "gemini-2.0-flash-exp"):
        super().__init__(api_key, gemini_model)
        self.variable_extractor = VariableExtractor(self)
        self.intent_recognizer = IntentRecognizer(self)
        
        # 벡터 DB 관련 초기화
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.policy_index_file = os.path.join(base_path, "faiss_index_policy.bin")
        self.policy_chunks_file = os.path.join(base_path, "document_chunks_policy.pkl")
        self.example_index_file = os.path.join(base_path, "faiss_index_examples.bin")
        self.example_chunks_file = os.path.join(base_path, "document_chunks_examples.pkl")
        self.validation_index_file = os.path.join(base_path, "faiss_index_validation.bin")
        self.validation_chunks_file = os.path.join(base_path, "document_chunks_validation.pkl")
        
        self.policy_index = None
        self.policy_chunks = None
        self.example_index = None
        self.example_chunks = None
        self.validation_index = None
        self.validation_chunks = None
    
    def preprocess_query(self, query: str) -> str:
        """
        '내일', '글피', 'N일 뒤'와 같은 시간 표현을 실제 날짜와 요일로 변환합니다.
        """
        today = date.today()

        # '내일'과 '글피' 처리
        if '내일' in query:
            tomorrow = today + timedelta(days=1)
            day_of_week = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"][tomorrow.weekday()]
            query = query.replace('내일', tomorrow.strftime('%Y년 %m월 %d일') + f'({day_of_week})')
        if '글피' in query:
            day_after_tomorrow = today + timedelta(days=2)
            day_of_week = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"][day_after_tomorrow.weekday()]
            query = query.replace('글피', day_after_tomorrow.strftime('%Y년 %m월 %d일') + f'({day_of_week})')

        # 'N일 뒤' 패턴 처리
        match = re.search(r'(\d+)\s*일\s*뒤', query)
        if match:
            days_to_add = int(match.group(1))
            future_date = today + timedelta(days=days_to_add)
            day_of_week = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"][future_date.weekday()]

            # 'N일 뒤'를 'YYYY년 MM월 DD일 (요일)' 형식으로 대체
            query = re.sub(r'(\d+)\s*일\s*뒤', future_date.strftime('%Y년 %m월 %d일') + f'({day_of_week})', query)

        return query

    def embed_texts(self, texts):
        """텍스트 임베딩 생성 (prototype.py에서 가져옴)"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            result = genai.embed_content(
                model="models/embedding-001",
                content=texts,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"임베딩 중 오류 발생: {e}")
            return [[0.0] * 768 for _ in range(len(texts))]

    def load_vector_databases(self):
        """벡터 데이터베이스 로드"""
        print("벡터 데이터베이스 로드 중...")
        
        # 정책 DB
        if os.path.exists(self.policy_index_file) and os.path.exists(self.policy_chunks_file):
            print("정책 인덱스 로드 중...")
            self.policy_index = faiss.read_index(self.policy_index_file)
            with open(self.policy_chunks_file, 'rb') as f:
                self.policy_chunks = pickle.load(f)
        
        # 예시 DB
        if os.path.exists(self.example_index_file) and os.path.exists(self.example_chunks_file):
            print("예시 인덱스 로드 중...")
            self.example_index = faiss.read_index(self.example_index_file)
            with open(self.example_chunks_file, 'rb') as f:
                self.example_chunks = pickle.load(f)
        
        # 검증 DB
        if os.path.exists(self.validation_index_file) and os.path.exists(self.validation_chunks_file):
            print("검증 인덱스 로드 중...")
            self.validation_index = faiss.read_index(self.validation_index_file)
            with open(self.validation_chunks_file, 'rb') as f:
                self.validation_chunks = pickle.load(f)

    def validate_against_policies(self, query, intent, top_k=3, threshold=0.8):
        """정책 위반 검증 (prototype.py 스타일)"""
        if not self.validation_index or not self.validation_chunks:
            print("검증 인덱스가 로드되지 않았습니다.")
            return False, None, None, None
            
        print("\n--- 정책 위반 검증 시작 ---")
        query_embedding = np.array(self.embed_texts([query])).astype('float32')
        distances, indices = self.validation_index.search(query_embedding, top_k)
        
        if distances.size == 0 or distances[0][0] > threshold:
            print("정책 위반 사항 없음")
            return False, None, None, None

        similarity_score = distances[0][0]
        retrieved_chunks = [self.validation_chunks[i] for i in indices[0]]
        
        context_for_llm = ""
        for chunk in retrieved_chunks:
            context_for_llm += f"출처: {chunk['source']}\n내용: {chunk['content']}\n---\n"
        
        print(f"유사도 검색 결과:\n{context_for_llm}")

        prompt = f"""
        당신은 카카오 알림톡 정책 심사 전문가입니다. 당신의 임무는 사용자의 요청이 주어진 [검증 규칙]을 위반하는지 판단하는 것입니다.
        **당신의 개인적인 지식이나 추론을 사용하지 마세요. 오직 주어진 [검증 규칙] 텍스트에만 근거하여 판단해야 합니다.**

        [사용자 원본 요청]
        {query}

        [검증 규칙]
        {context_for_llm}

        [판단]
        1.  [사용자 원본 요청]의 내용과 직접적으로 관련된 규칙을 [검증 규칙]에서 찾으세요.
        2.  만약, 요청 내용이 찾아낸 규칙을 명백히 위반한다면, "위반"이라고 답변하고, **어떤 출처의 어떤 규칙**을 위반했는지 **해당 규칙의 원본 텍스트를 그대로 인용하여** 설명해주세요.
        3.  위반 사항을 찾지 못했다면, "정상"이라고만 답변해주세요.
        """
        
        response = self.generate_with_gemini(prompt)
        print(f"\nLLM 판단: {response}")

        if "위반" in response.split('\n')[-1]:
            print(" 정책 위반 발견")
            return True, response, similarity_score, context_for_llm
        
        print("정책 위반 사항 없음")
        return False, None, None, None

    def generate_template_with_pipeline(
        self,
        user_input: str,
        guidelines: List[str] = None,
    ) -> Tuple[str, str]:
        """prototype.py 스타일의 완전한 파이프라인으로 템플릿 생성"""
        
        print("\n=== prototype.py 스타일 템플릿 생성 파이프라인 시작 ===")
        
        # 0. 벡터 DB 로드
        if not self.policy_index:
            self.load_vector_databases()
        
        # 1. 날짜 전처리
        processed_input = self.preprocess_query(user_input)
        print(f" 날짜 전처리: '{user_input}' → '{processed_input}'")

        # 2. 변수 추출
        print("\n--- 1. 변수 추출 시작 ---")
        variables = self.variable_extractor.extract(processed_input)
        print(f"추출된 변수: {variables}")
        
        # 3. 필수 변수 확인
        print("\n--- 2. 필수 변수 확인 ---")
        mandatory_vars = ["누가 (To/Recipient)", "무엇을 (What/Subject)", "어떻게 (How/Method)"]
        missing_vars = [var for var in mandatory_vars if variables.get(var, "없음") == "없음"]
        
        if missing_vars:
            print(f" 입력 정보 부족: {', '.join(missing_vars)}")
            return self._generate_fallback_template(processed_input, {"extracted_info": {}}), ""

        # 4. 의도 파악
        print("\n--- 3. 의도 파악 시작 ---")
        intent = self.intent_recognizer.recognize(processed_input, variables)
        print(f"파악된 의도: '{intent}'")

        # 5. 정책 검증
        print("\n--- 4. 정책 검증 시작 ---")
        is_violation, violation_reason, similarity, violation_context = self.validate_against_policies(
            processed_input, intent
        )
        if is_violation:
            print(f" 정책 위반 발견: {violation_reason}")
            return f"정책 위반으로 템플릿 생성을 중단했습니다.\n\n위반 사유:\n{violation_reason}", ""

        # 6. 벡터 검색 기반 템플릿 생성
        print("\n--- 5. 템플릿 생성 시작 ---")
        template = self._generate_template_with_vector_search(
            processed_input, variables, intent, guidelines
        )
        
        # 7. 엔티티 기반 미리보기 생성
        filled_template = self._fill_template_with_variables(template, variables)

        return template, filled_template

    def _generate_template_with_vector_search(
        self, 
        user_input: str, 
        variables: Dict, 
        intent: str, 
        guidelines: List[str] = None,
        top_k: int = 3
    ) -> str:
        """벡터 검색 기반 템플릿 생성 (prototype.py 스타일)"""
        
        # 의도 기반 벡터 검색으로 관련 정책/예시 찾기
        intent_embedding = np.array(self.embed_texts([intent])).astype('float32')
        
        # 정책 검색
        policy_context = ""
        if self.policy_index and self.policy_chunks:
            pol_distances, pol_indices = self.policy_index.search(intent_embedding, top_k)
            if pol_distances.size > 0:
                retrieved_policies = [self.policy_chunks[i] for i in pol_indices[0]]
                policy_context = "\n".join([chunk['content'] for chunk in retrieved_policies])
                print(f"검색된 정책 규칙:\n{policy_context}")

        # 예시 검색  
        example_context = ""
        if self.example_index and self.example_chunks:
            ex_distances, ex_indices = self.example_index.search(intent_embedding, top_k)
            if ex_distances.size > 0:
                retrieved_examples = [self.example_chunks[i] for i in ex_indices[0]]
                example_context = "\n".join([chunk['content'] for chunk in retrieved_examples])
                print(f"검색된 스타일 예시:\n{example_context}")

        # prototype.py 스타일의 시스템/휴먼 메시지 분리
        system_prompt = """
        당신은 카카오 알림톡 템플릿 생성 전문가입니다.
        다음 정보를 바탕으로 카카오 알림톡 템플릿 본문을 작성해주세요.
        - 설명, 피드백, 주석 등은 절대 포함하지 말고, 오직 템플릿 내용만 작성해주세요.
        - 변수가 필요한 부분은 `#{...}` 형식으로 표시해주세요.
        """
        
        human_prompt = f"""
        [사용자 요청]
        {user_input}

        [추출된 변수]
        {variables}

        [참고 정책]
        {policy_context}

        [참고 예시]
        {example_context}

        [알림톡 템플릿]
        """
        
        # 시스템 + 휴먼 프롬프트 결합
        full_prompt = system_prompt + "\n\n" + human_prompt
        
        try:
            response = self.generate_with_gemini(full_prompt)
            cleaned_response = response.replace("```", "").strip()
            return cleaned_response
        except Exception as e:
            print(f"벡터 검색 템플릿 생성 오류: {e}")
            return self._generate_fallback_template(user_input, {"extracted_info": {}})

    def _fill_template_with_variables(self, template: str, variables: Dict) -> str:
        """변수로 템플릿 채우기 (prototype.py 스타일)"""
        filled = template
        
        # 각 변수를 템플릿에 매핑
        variable_mapping = {
            "누가 (To/Recipient)": ["#{수신자명}", "#{고객명}", "#{회원명}"],
            "무엇을 (What/Subject)": ["#{행사명}", "#{이벤트명}", "#{서비스명}"],
            "어떻게 (How/Method)": ["#{방법}", "#{절차}", "#{안내사항}"],
            "언제 (When/Time)": ["#{일시}", "#{날짜}", "#{시간}"],
            "어디서 (Where/Place)": ["#{장소}", "#{위치}", "#{주소}"],
            "왜 (Why/Reason)": ["#{사유}", "#{목적}", "#{이유}"]
        }
        
        for var_key, value in variables.items():
            if value and value != "없음":
                if var_key in variable_mapping:
                    for template_var in variable_mapping[var_key]:
                        filled = filled.replace(template_var, value)
        
        return filled

    # 기존 메서드들도 그대로 유지
    def generate_template(
        self,
        user_input: str,
        entities: Dict,
        similar_templates: List[Tuple[str, float]],
        guidelines: List[str] = None,
    ) -> Tuple[str, str]:
        """기존 호환성을 위한 메서드 (새로운 파이프라인 사용)"""
        return self.generate_template_with_pipeline(user_input, guidelines)

    def _generate_guideline_based_template(
        self,
        user_input: str,
        entities: Dict,
        similar_templates: List[Tuple[str, float]],
        guidelines: List[str],
    ) -> str:
        """가이드라인 기반 템플릿 생성"""

        template_examples = self._format_template_examples(similar_templates)
        guidelines_text = "\n".join(guidelines[:3]) if guidelines else ""

        prompt = self._create_template_generation_prompt(
            user_input,
            entities,
            template_examples,
            guidelines_text,
            use_guidelines=True,
        )

        try:
            response = self.generate_with_gemini(prompt)
            print(f" Gemini 원본 응답 길이: {len(response)}")
            print(f" Gemini 원본 응답 (처음 200자): {response[:200]}...")
            
            cleaned_response = response.replace("```", "").strip()
            print(f" 정리된 응답 길이: {len(cleaned_response)}")
            
            return cleaned_response
        except Exception as e:
            print(f"가이드라인 기반 템플릿 생성 오류: {e}")
            return self._generate_fallback_template(user_input, entities)

    def _generate_basic_template(
        self,
        user_input: str,
        entities: Dict,
        similar_templates: List[Tuple[str, float]],
    ) -> str:
        """기본 템플릿 생성"""

        template_examples = self._format_template_examples(similar_templates)

        prompt = self._create_template_generation_prompt(
            user_input, entities, template_examples, "", use_guidelines=False
        )

        try:
            response = self.generate_with_gemini(prompt)
            print(f" 기본 템플릿 - Gemini 원본 응답 길이: {len(response)}")
            print(f" 기본 템플릿 - Gemini 원본 응답 (처음 200자): {response[:200]}...")
            
            cleaned_response = response.replace("```", "").strip()
            print(f" 기본 템플릿 - 정리된 응답 길이: {len(cleaned_response)}")
            
            return cleaned_response
        except Exception as e:
            print(f"기본 템플릿 생성 오류: {e}")
            return self._generate_fallback_template(user_input, entities)

    def _create_template_generation_prompt(
        self,
        user_input: str,
        entities: Dict,
        template_examples: str,
        guidelines: str,
        use_guidelines: bool = False,
    ) -> str:
        """prototype.py 스타일의 개선된 템플릿 생성 프롬프트"""

        extracted_info = entities.get("extracted_info", {})
        intent = entities.get("message_intent", "일반안내")
        context = entities.get("context", user_input)
        message_type = entities.get("message_type", "정보성")
        urgency = entities.get("urgency_level", "보통")

        # prototype.py 스타일의 시스템 프롬프트 적용
        system_prompt = """
        당신은 카카오 알림톡 템플릿 생성 전문가입니다.
        다음 정보를 바탕으로 카카오 알림톡 템플릿 본문을 작성해주세요.
        - 설명, 피드백, 주석 등은 절대 포함하지 말고, 오직 템플릿 내용만 작성해주세요.
        - 변수가 필요한 부분은 `#{...}` 형식으로 표시해주세요.
        """

        base_prompt = f"""
{system_prompt}

[사용자 요청]
{user_input}

[추출된 변수]
- 날짜: {', '.join(extracted_info.get('dates', [])) if extracted_info.get('dates') else '없음'}
- 이름: {', '.join(extracted_info.get('names', [])) if extracted_info.get('names') else '없음'}
- 장소: {', '.join(extracted_info.get('locations', [])) if extracted_info.get('locations') else '없음'}
- 이벤트: {', '.join(extracted_info.get('events', [])) if extracted_info.get('events') else '없음'}
- 기타정보: {', '.join(extracted_info.get('others', [])) if extracted_info.get('others') else '없음'}
- 메시지 의도: {intent}
- 메시지 유형: {message_type}
- 긴급도: {urgency}
"""

        if use_guidelines and guidelines:
            base_prompt += f"""
[참고 정책]
{guidelines}
"""

        if template_examples:
            base_prompt += f"""
[참고 예시]
{template_examples}
"""

        base_prompt += """
[알림톡 템플릿]
위 요청과 변수들을 바탕으로 카카오 알림톡 템플릿을 작성해주세요.

**필수 준수사항:**
1. 정보통신망법 준수 (정보성 메시지 기준)
2. 추출된 구체적 정보들을 #{변수명} 형태로 포함
3. 수신자에게 필요한 모든 정보 포함  
4. 명확하고 정중한 안내 톤
5. 메시지 끝에 발송 사유 및 법적 근거 명시

**템플릿 구조:**
- 인사말 및 발신자 소개
- 주요 안내 내용 (상세히)
- 구체적인 정보 (일시, 장소, 방법 등)
- 추가 안내사항 또는 주의사항
- 문의처 또는 연락방법
- 발송 사유 및 법적 근거

오직 템플릿 내용만 생성해주세요:
"""

        return base_prompt

    def _format_template_examples(
        self, similar_templates: List[Tuple[str, float]]
    ) -> str:
        """템플릿 예시 포맷팅"""
        if not similar_templates:
            return ""

        examples = []
        for i, (template, score) in enumerate(similar_templates[:2], 1):
            examples.append(f"{i}. {template}\n")

        return "\n".join(examples)

    def _fill_template_with_entities(self, template: str, entities: Dict) -> str:
        """템플릿에 추출된 엔티티 정보 자동 입력"""
        filled = template
        extracted_info = entities.get("extracted_info", {})

        # 날짜 정보 매핑
        if extracted_info.get("dates"):
            date_patterns = [
                (
                    r"#\{(일시|날짜|시간|적용일|방문일정|예약일정|행사일시)\}",
                    extracted_info["dates"][0],
                )
            ]
            for pattern, value in date_patterns:
                filled = re.sub(pattern, value, filled, flags=re.IGNORECASE)

        # 이름 정보 매핑
        if extracted_info.get("names"):
            name_patterns = [
                (
                    r"#\{(수신자명|수신자|고객명|보호자명|회원명)\}",
                    extracted_info["names"][0],
                )
            ]
            for pattern, value in name_patterns:
                filled = re.sub(pattern, value, filled, flags=re.IGNORECASE)

        # 장소 정보 매핑
        if extracted_info.get("locations"):
            location_patterns = [
                (
                    r"#\{(장소|매장명|주소|위치|행사장소)\}",
                    extracted_info["locations"][0],
                )
            ]
            for pattern, value in location_patterns:
                filled = re.sub(pattern, value, filled, flags=re.IGNORECASE)

        # 이벤트 정보 매핑
        if extracted_info.get("events"):
            event_patterns = [
                (
                    r"#\{(행사명|이벤트명|활동명|프로그램명)\}",
                    extracted_info["events"][0],
                )
            ]
            for pattern, value in event_patterns:
                filled = re.sub(pattern, value, filled, flags=re.IGNORECASE)

        return filled

    def _generate_fallback_template(self, user_input: str, entities: Dict) -> str:
        """오류 시 기본 템플릿 생성"""
        extracted_info = entities.get("extracted_info", {})
        intent = entities.get("message_intent", "일반안내")

        name_var = (
            extracted_info.get("names", ["#{수신자명}"])[0]
            if extracted_info.get("names")
            else "#{수신자명}"
        )
        date_var = (
            extracted_info.get("dates", ["#{일시}"])[0]
            if extracted_info.get("dates")
            else "#{일시}"
        )
        location_var = (
            extracted_info.get("locations", ["#{장소}"])[0]
            if extracted_info.get("locations")
            else "#{장소}"
        )
        event_var = (
            extracted_info.get("events", ["#{행사명}"])[0]
            if extracted_info.get("events")
            else "#{행사명}"
        )

        return f"""안녕하세요, {name_var}님.
{intent}에 대해 상세히 안내드립니다.

 주요 내용: {event_var}
 일시: {date_var}
 장소: {location_var}

[상세 안내사항]
- 참석하실 분들께서는 미리 준비해주시기 바랍니다.
- 자세한 내용은 별도 공지사항을 확인해주세요.
- 변경사항이 있을 경우 개별 안내드리겠습니다.

[문의사항]
궁금한 사항이 있으시면 언제든 연락 부탁드립니다.
- 연락처: #{{연락처}}
- 운영시간: 평일 오전 9시~오후 6시

※ 본 메시지는 관련 서비스를 신청하신 분들께만 발송되는 정보성 안내 메시지입니다."""

    def optimize_template(self, template: str, entities: Dict) -> str:
        """템플릿 최적화"""
        # 길이 체크
        if len(template) < 200:
            return self._expand_template(template, entities)

        # 변수 체크
        variables = self.extract_variables(template)
        if len(variables) < 3:
            return self._add_more_variables(template, entities)

        return template

    def _expand_template(self, template: str, entities: Dict) -> str:
        """템플릿 확장"""
        # 추가 정보 섹션 삽입
        additional_info = """
[추가 안내사항]
- 정확한 정보 확인을 위해 사전 연락 부탁드립니다.
- 변경사항 발생 시 즉시 안내드리겠습니다.
- 기타 문의사항은 고객센터를 이용해주세요."""

        # 법적 고지 전에 추가 정보 삽입
        if "※" in template:
            parts = template.split("※")
            return parts[0] + additional_info + "\n\n※" + "※".join(parts[1:])
        else:
            return template + additional_info

    def _add_more_variables(self, template: str, entities: Dict) -> str:
        """더 많은 변수 추가"""
        # 기본 변수들 추가
        additional_vars = ["#{문의전화}", "#{운영시간}", "#{담당자명}"]

        contact_section = f"""
[문의 및 연락처]
- 담당자: {additional_vars[2]}
- 연락처: {additional_vars[0]} 
- 운영시간: {additional_vars[1]}"""

        if "※" in template:
            parts = template.split("※")
            return parts[0] + contact_section + "\n\n※" + "※".join(parts[1:])
        else:
            return template + contact_section
