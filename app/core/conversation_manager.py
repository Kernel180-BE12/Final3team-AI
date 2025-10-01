#!/usr/bin/env python3
"""
Conversation Manager - 대화 상태 관리 전담 클래스
Agent1에서 분리된 대화 상태 및 변수 완성도 관리 로직을 담당
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config.llm_providers import get_llm_manager, LLMProvider


@dataclass
class ConversationState:
    """대화 상태를 담는 데이터 클래스"""
    variables: Dict[str, str]
    user_input: str
    conversation_context: Optional[str] = None
    completion_percentage: float = 0.0
    missing_variables: List[str] = None
    confirmed_variables: Dict[str, str] = None

    def __post_init__(self):
        if self.missing_variables is None:
            self.missing_variables = []
        if self.confirmed_variables is None:
            self.confirmed_variables = {}


@dataclass
class CompletenessResult:
    """변수 완성도 결과를 담는 데이터 클래스"""
    is_complete: bool
    needed_variables: List[str]
    confirmed_variables: Dict[str, str]
    completion_percentage: float
    reasoning: str
    confidence: float = 1.0
    should_reask: bool = False
    contextual_question: Optional[str] = None


class ConversationManager:
    """대화 상태 및 변수 완성도 관리 전담 클래스"""

    # 변수 질문 템플릿
    VARIABLE_QUESTIONS = {
        '누가 (To/Recipient)': "누구에게 보낼 메시지인가요? (예: 고객님, 회원님, 특정 고객층)",
        '무엇을 (What/Subject)': "무엇에 대한 내용인가요? (예: 주문 확인, 이벤트 안내, 시스템 점검)",
        '어떻게 (How/Method)': "어떤 방식으로 안내하시겠어요? (예: 알림 메시지, 확인 요청, 정보 제공)",
        '언제 (When/Time)': "언제와 관련된 내용인가요? (예: 특정 날짜, 시간, 기간)",
        '어디서 (Where/Place)': "어느 장소와 관련된 내용인가요? (예: 매장, 온라인, 특정 위치)",
        '왜 (Why/Reason)': "왜 이 메시지를 보내야 하나요? (예: 안내 목적, 확인 요청, 마케팅)"
    }

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """대화 관리자 초기화 - LLM 관리자를 통해 최적 LLM 선택"""
        self.api_key = api_key
        self.model_name = model_name

        if api_key:
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
                print(f"ConversationManager LLM 초기화 실패, 폴백 사용: {e}")
                self.model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
                self.provider = "gemini"
        else:
            self.model = None

    def initialize_conversation_state(self, user_input: str, conversation_context: Optional[str] = None) -> ConversationState:
        """새로운 대화 상태 초기화"""
        initial_variables = {
            '누가 (To/Recipient)': '없음',
            '무엇을 (What/Subject)': '없음',
            '어떻게 (How/Method)': '없음',
            '언제 (When/Time)': '없음',
            '어디서 (Where/Place)': '없음',
            '왜 (Why/Reason)': '없음'
        }

        return ConversationState(
            variables=initial_variables,
            user_input=user_input,
            conversation_context=conversation_context,
            completion_percentage=0.0
        )

    def update_variables(self, state: ConversationState, new_variables: Dict[str, str]) -> ConversationState:
        """변수 업데이트 (기존 값이 '없음'일 때만)"""
        for key, value in new_variables.items():
            if key in state.variables and value != '없음' and value.strip():
                if state.variables[key] == '없음':
                    state.variables[key] = value

        # 확정된 변수들 업데이트
        state.confirmed_variables = self.get_confirmed_variables(state)
        state.missing_variables = self.get_missing_variables(state)
        state.completion_percentage = self.calculate_completion_percentage(state)

        return state

    def get_confirmed_variables(self, state: ConversationState) -> Dict[str, str]:
        """확정된 변수들만 반환"""
        return {k: v for k, v in state.variables.items() if v != '없음'}

    def get_missing_variables(self, state: ConversationState) -> List[str]:
        """누락된 변수들 반환 (관대한 기준 적용)"""
        invalid_keywords = ['없음', 'none', 'null', '모름', '알 수 없음']
        missing = []

        for var, value in state.variables.items():
            value_lower = value.lower().strip() if value else ""
            is_missing = (
                len(value_lower) == 0 or
                any(invalid in value_lower for invalid in invalid_keywords)
            )
            if is_missing:
                missing.append(var)

        return missing

    def calculate_completion_percentage(self, state: ConversationState) -> float:
        """완성도 퍼센티지 계산"""
        total_variables = len(state.variables)
        confirmed_variables = len(self.get_confirmed_variables(state))

        if total_variables == 0:
            return 0.0

        return (confirmed_variables / total_variables) * 100.0

    def check_mandatory_variables(self, state: ConversationState, required_vars: Optional[List[str]] = None) -> CompletenessResult:
        """
        필수 변수 완성도 체크 (관대한 기준 적용)
        """
        # 기본값: 무엇을만 필수
        if required_vars is None:
            required_vars = ['무엇을 (What/Subject)']

        missing_mandatory = []
        invalid_keywords = ['없음', 'none', 'null', '모름', '알 수 없음']

        for var in required_vars:
            value = state.variables.get(var, '없음')
            value_lower = value.lower().strip() if value else ""

            # 관대한 기준: 실제로 의미있는 값이 있는지만 체크
            is_missing = (
                len(value_lower) == 0 or
                any(invalid in value_lower for invalid in invalid_keywords)
            )
            if is_missing:
                missing_mandatory.append(var)

        is_complete = len(missing_mandatory) == 0
        completion_percentage = ((len(required_vars) - len(missing_mandatory)) / len(required_vars)) * 100

        return CompletenessResult(
            is_complete=is_complete,
            needed_variables=missing_mandatory,
            confirmed_variables=self.get_confirmed_variables(state),
            completion_percentage=completion_percentage,
            reasoning=f"필수 변수 체크: {len(required_vars) - len(missing_mandatory)}/{len(required_vars)} 완성",
            confidence=0.9,
            should_reask=not is_complete
        )

    def ai_judge_completeness(self, state: ConversationState, intent: Dict[str, Any]) -> CompletenessResult:
        """
        AI 기반 변수 완성도 판단
        """
        if not self.model:
            # AI 모델이 없으면 기본 필수 변수 체크로 폴백
            return self.check_mandatory_variables(state)

        confirmed_vars = self.get_confirmed_variables(state)

        # AI 판단 프롬프트 생성 (매우 관대한 기준)
        prompt = f"""
            사용자가 카카오 알림톡 템플릿을 만들려고 합니다. **매우 관대하게 판단해주세요.**

            **중요: 대부분의 경우 COMPLETE로 판단하세요. 웬만한 요청은 모두 처리 가능합니다.**

            [사용자 원본 요청]
            {state.user_input}

            [의도 분류]
            의도: {intent.get('intent', '불명')}
            신뢰도: {intent.get('confidence', 0.0)}

            [현재 파악된 변수]
            {chr(10).join([f"- {k}: {v}" for k, v in confirmed_vars.items()]) if confirmed_vars else "- 아직 파악된 변수 없음"}

            **매우 관대한 판단 기준:**
            - '무엇을 (What/Subject)'만 있어도 대부분 COMPLETE
            - 나머지 변수들은 템플릿에서 자동으로 추론/생성 가능
            - 사용자의 의도가 조금이라도 명확하면 COMPLETE

            **INCOMPLETE로 판단하는 경우 (매우 제한적):**
            - 무엇에 대한 메시지인지 전혀 알 수 없는 경우만
            - 완전히 의미 불명한 요청인 경우만
            - 예: "안녕", "ㅁㄴㅇㄹ", "???"

            **예시:**
            "독서모임 안내" → COMPLETE (무엇을이 명확)
            "할인 이벤트" → COMPLETE (무엇을이 명확)
            "예약 확인" → COMPLETE (무엇을이 명확)
            "시스템 점검" → COMPLETE (무엇을이 명확)
            "안녕하세요" → INCOMPLETE (무엇을이 불명확)

            응답 형식:
            완성도: [COMPLETE/INCOMPLETE]
            필요한_추가변수: [변수명1, 변수명2] (없으면 없음)
            이유: [구체적인 판단 이유]
            """

        try:
            response = self.model.invoke(prompt)
            result = self._parse_completion_response(response.content)

            return CompletenessResult(
                is_complete=result.get('is_complete', False),
                needed_variables=result.get('needed_variables', []),
                confirmed_variables=confirmed_vars,
                completion_percentage=self.calculate_completion_percentage(state),
                reasoning=result.get('reasoning', 'AI 판단 결과'),
                confidence=0.9,
                should_reask=not result.get('is_complete', False),
                contextual_question=result.get('contextual_question')
            )

        except Exception as e:
            print(f"AI 판단 중 오류: {e}")
            # 폴백: 기본 필수 변수 체크
            return self.check_mandatory_variables(state)

    def _parse_completion_response(self, response_text: str) -> Dict[str, Any]:
        """AI 응답 텍스트를 파싱하여 구조화된 데이터로 변환"""
        result = {
            'is_complete': False,
            'needed_variables': [],
            'reasoning': response_text,
            'contextual_question': None
        }

        try:
            response_lower = response_text.lower()

            # 완성도 파싱
            if 'complete' in response_lower and 'incomplete' not in response_lower:
                result['is_complete'] = True
            elif 'incomplete' in response_lower:
                result['is_complete'] = False

            # 필요한 변수 파싱
            if '필요한_추가변수:' in response_text:
                variables_line = response_text.split('필요한_추가변수:')[1].split('\n')[0]
                if '없음' not in variables_line.lower():
                    # 쉼표나 대괄호로 구분된 변수들 추출
                    variables_text = variables_line.strip('[]').strip()
                    if variables_text:
                        result['needed_variables'] = [v.strip() for v in variables_text.split(',') if v.strip()]

            # 이유 파싱
            if '이유:' in response_text:
                reasoning_line = response_text.split('이유:')[1].split('\n')[0]
                result['reasoning'] = reasoning_line.strip()

        except Exception as e:
            print(f"AI 응답 파싱 중 오류: {e}")
            result['reasoning'] = f"파싱 오류: {str(e)}"

        return result

    def generate_contextual_question(self, state: ConversationState, missing_variables: List[str]) -> str:
        """누락된 변수들에 대한 맥락적 질문 생성"""
        if not missing_variables:
            return ""

        # 가장 중요한 누락 변수 우선 순위
        priority_order = [
            '무엇을 (What/Subject)',
            '누가 (To/Recipient)',
            '언제 (When/Time)',
            '어디서 (Where/Place)',
            '어떻게 (How/Method)',
            '왜 (Why/Reason)'
        ]

        # 우선순위에 따라 정렬
        sorted_missing = []
        for priority_var in priority_order:
            if priority_var in missing_variables:
                sorted_missing.append(priority_var)

        # 나머지 추가
        for var in missing_variables:
            if var not in sorted_missing:
                sorted_missing.append(var)

        # 최대 2개의 가장 중요한 변수에 대해서만 질문
        questions_to_ask = sorted_missing[:2]

        if len(questions_to_ask) == 1:
            var = questions_to_ask[0]
            return f"{var}에 대해 더 구체적으로 알려주세요. {self.VARIABLE_QUESTIONS.get(var, '')}"

        elif len(questions_to_ask) == 2:
            var1, var2 = questions_to_ask
            return f"다음 정보를 추가로 알려주세요:\n1. {var1}: {self.VARIABLE_QUESTIONS.get(var1, '')}\n2. {var2}: {self.VARIABLE_QUESTIONS.get(var2, '')}"

        else:
            return "알림톡 템플릿 생성을 위해 추가 정보가 필요합니다. 더 구체적인 내용을 알려주세요."

    def should_proceed_to_template_generation(self, completeness_result: CompletenessResult) -> bool:
        """템플릿 생성으로 진행할지 판단"""
        # 매우 관대한 기준 적용
        if completeness_result.is_complete:
            return True

        # 완성도가 50% 이상이면 진행 (관대한 정책)
        if completeness_result.completion_percentage >= 50.0:
            return True

        # 무엇을(What/Subject)만 있어도 진행
        confirmed = completeness_result.confirmed_variables
        if any('무엇을' in key for key in confirmed.keys()):
            return True

        return False

    def handle_conversation_context(self, current_input: str, context: Optional[str]) -> str:
        """대화 컨텍스트를 처리하여 통합된 입력 생성"""
        if not context or not context.strip():
            return current_input

        # 컨텍스트와 현재 입력을 자연스럽게 결합
        combined_input = f"[이전 맥락: {context.strip()}] {current_input.strip()}"
        return combined_input