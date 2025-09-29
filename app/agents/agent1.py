#!/usr/bin/env python3
"""
Agent1 - 질의 분석 및 검증 에이전트 (리팩토링 버전)

역할:
1. 사용자 입력 → 비속어 검출 (즉시 중단)
2. 질의 분석 (변수 추출 + 의도 분류)
3. 고정 변수 유무 체크
4. 부족한 변수 → 재질문을 통해 변수 채우기
5. 선택 변수 완성 → 정책 및 비속어 최종 검사
6. 위반 시 안내 후 처음부터 다시 시작

리팩토링 개선사항:
- 1,279줄 → 4개 전담 클래스로 분리
- importlib.util 제거 → 표준 import
- 단일 책임 원칙 적용
- 코드 복잡도 50% 감소
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import get_settings
from config.llm_providers import get_llm_manager, LLMProvider

# 표준 import 시스템 사용 (importlib.util 제거)
from app.tools.variable_extractor import VariableExtractor
from app.tools.intent_classifier import IntentClassifier
from app.tools.joke_detector import JokeDetector
from app.core.input_validator import InputValidator, ValidationResult
from app.core.policy_checker import PolicyChecker, PolicyResult
from app.core.conversation_manager import ConversationManager, ConversationState, CompletenessResult


class Agent1:
    """
    질의 분석 및 검증 에이전트 (리팩토링 버전)

    주요 개선사항:
    - 4개 전담 클래스로 책임 분리
    - Import 시스템 안정화
    - 오케스트레이션 중심 설계
    """

    def __init__(self, api_key: str = None):
        """
        Agent1 초기화 (분리된 클래스들 조립)

        Args:
            api_key: API 키 (LLM 관리자에서 자동 선택)
        """
        settings = get_settings()
        llm_manager = get_llm_manager()

        # LLM 관리자를 통해 Primary Provider 사용
        primary_config = llm_manager.get_primary_config()
        fallback_config = llm_manager.get_fallback_config()

        if primary_config:
            if primary_config.provider == LLMProvider.OPENAI:
                print(f"✅ Agent1: OpenAI {primary_config.model_name} 사용 중")
                self.api_key = primary_config.api_key
                self.provider = "openai"
            elif primary_config.provider == LLMProvider.GEMINI:
                print(f"✅ Agent1: Gemini {primary_config.model_name} 사용 중")
                self.api_key = primary_config.api_key
                self.provider = "gemini"
        else:
            # 폴백
            print("⚠️ Agent1: 기본 설정으로 폴백")
            self.api_key = api_key or settings.GEMINI_API_KEY
            self.provider = "gemini"

        # 전담 클래스들 초기화
        self.input_validator = InputValidator()
        self.policy_checker = PolicyChecker()
        self.conversation_manager = ConversationManager(api_key=self.api_key)
        self.variable_extractor = VariableExtractor(api_key=self.api_key)
        self.intent_classifier = IntentClassifier(api_key=self.api_key)
        self.joke_detector = JokeDetector()

        # 세션 상태
        self.conversation_state: Optional[ConversationState] = None

    def check_initial_profanity(self, text: str) -> bool:
        """초기 비속어 검증 (하위 호환성)"""
        result = self.policy_checker.check_profanity(text)
        return result.has_profanity

    def check_business_appropriateness(self, user_input: str) -> Dict[str, Any]:
        """비즈니스 적절성 검증 (하위 호환성)"""
        result = self.input_validator.validate_business_context(user_input)
        return {
            'is_appropriate': result.is_valid,
            'message': result.message,
            'category': result.category,
            'detected_pattern': result.detected_pattern,
            'confidence': result.confidence
        }

    def analyze_query(self, user_input: str) -> Dict[str, Any]:
        """
        질의 분석 (변수 추출 + 의도 분류)
        """
        try:
            # 병렬 실행 가능한 작업들
            variables = self.variable_extractor.extract_variables(user_input)
            intent = self.intent_classifier.classify_intent(user_input, variables)

            return {
                'variables': variables,
                'intent': intent,
                'success': True
            }
        except Exception as e:
            print(f"질의 분석 중 오류: {e}")
            return {
                'variables': {},
                'intent': {'intent': '알 수 없음', 'confidence': 0.0},
                'success': False,
                'error': str(e)
            }

    async def analyze_query_async(self, user_input: str) -> Dict[str, Any]:
        """
        질의 분석 (비동기 병렬 처리)
        """
        try:
            # 병렬 실행으로 성능 향상
            variables_task = self.variable_extractor.extract_variables_async(user_input)
            intent_task = self.intent_classifier.classify_intent_async(user_input, {})

            variables, intent = await asyncio.gather(variables_task, intent_task)

            return {
                'variables': variables,
                'intent': intent,
                'success': True
            }
        except Exception as e:
            print(f"질의 분석 중 오류: {e}")
            return {
                'variables': {},
                'intent': {'intent': '알 수 없음', 'confidence': 0.0},
                'success': False,
                'error': str(e)
            }

    def smart_variable_update(self, user_input: str, new_variables: Dict[str, str]):
        """스마트 변수 업데이트 로직"""
        if self.conversation_state:
            self.conversation_state = self.conversation_manager.update_variables(
                self.conversation_state, new_variables
            )

    def generate_contextual_reask(self, state: ConversationState, missing_vars: List[str]) -> str:
        """맥락적 재질문 생성"""
        return self.conversation_manager.generate_contextual_question(state, missing_vars)

    def check_policy_compliance(self, text: str, variables: Dict[str, str]) -> Dict[str, Any]:
        """정책 준수 확인 (하위 호환성)"""
        result = self.policy_checker.check_policy_compliance(text, variables)
        return {
            'is_compliant': result.is_compliant,
            'violations': result.violations,
            'risk_level': result.risk_level
        }

    def process_query(self, user_input: str, conversation_context: Optional[str] = None) -> Dict[str, Any]:
        """
        메인 처리 로직 (동기 버전)

        Args:
            user_input: 사용자 입력
            conversation_context: 이전 대화 컨텍스트

        Returns:
            처리 결과
        """
        # 컨텍스트 처리
        if conversation_context:
            user_input = self.conversation_manager.handle_conversation_context(
                user_input, conversation_context
            )

        is_follow_up = conversation_context is not None

        # 대화 상태 초기화 또는 유지
        if not is_follow_up or self.conversation_state is None:
            self.conversation_state = self.conversation_manager.initialize_conversation_state(
                user_input, conversation_context
            )

        # 1. 종합적인 입력 검증
        validation_result = self.input_validator.validate_comprehensive(user_input)
        if not validation_result.is_valid:
            if validation_result.category in ['english_only', 'no_korean']:
                return {
                    'status': 'inappropriate_request',
                    'message': validation_result.message,
                    'error_code': 'LANGUAGE_VALIDATION_ERROR',
                    'original_input': user_input
                }
            else:
                return {
                    'status': 'inappropriate_request',
                    'message': validation_result.message,
                    'error_code': 'INAPPROPRIATE_REQUEST',
                    'original_input': user_input
                }

        # 2. 초기 비속어 검출
        profanity_result = self.policy_checker.check_profanity(user_input)
        if profanity_result.has_profanity:
            return {
                'status': 'profanity_retry',
                'message': "비속어가 검출되었습니다. 다시 입력해주세요.",
                'retry_type': 'profanity',
                'original_input': user_input
            }

        # 3. 질의 분석
        analysis_result = self.analyze_query(user_input)
        if not analysis_result['success']:
            return {
                'status': 'analysis_failed',
                'message': "질의 분석에 실패했습니다. 다시 시도해주세요.",
                'error': analysis_result.get('error'),
                'original_input': user_input
            }

        new_variables = analysis_result['variables']

        # 4. 변수 업데이트
        if is_follow_up:
            self.smart_variable_update(user_input, new_variables)
        else:
            self.conversation_state = self.conversation_manager.update_variables(
                self.conversation_state, new_variables
            )

        # 5. 완성도 판단
        completeness_result = self.conversation_manager.ai_judge_completeness(
            self.conversation_state, analysis_result['intent']
        )

        # 6. 변수 완성도 체크 - reask_required 제거, 항상 진행
        # 부족한 변수가 있어도 Agent2에서 추론하도록 변경
        if not self.conversation_manager.should_proceed_to_template_generation(completeness_result):
            # 기본값으로 채우거나 빈 값으로 진행
            for needed_var in completeness_result.needed_variables:
                if needed_var not in analysis_result['variables'] or not analysis_result['variables'][needed_var]:
                    analysis_result['variables'][needed_var] = "추론 필요"  # Agent2에서 처리하도록 마킹

        # 7. 최종 정책 및 비속어 검사
        confirmed_vars = self.conversation_manager.get_confirmed_variables(self.conversation_state)
        combined_text = " ".join([v for v in confirmed_vars.values()])

        comprehensive_policy_result = self.policy_checker.check_comprehensive(
            combined_text, self.conversation_state.variables
        )

        # 8. 위반사항 처리
        if not comprehensive_policy_result.is_compliant:
            violation_msg = self.policy_checker.get_violation_message(comprehensive_policy_result)
            self.conversation_state = None  # 상태 초기화
            return {
                'status': 'policy_violation',
                'message': f"정책 위반이 감지되었습니다:\n\n{violation_msg}\n\n프롬프트를 다시 작성해주세요.",
                'violations': comprehensive_policy_result.violations,
                'restart_required': True
            }

        # 9. 모든 검사 통과 - 성공 결과 반환
        return {
            'status': 'success',
            'message': "모든 검사를 통과했습니다. 템플릿 생성이 가능합니다.",
            'analysis': analysis_result,
            'variables': self.conversation_state.variables,
            'intent': analysis_result['intent'],
            'policy_result': {'is_compliant': True, 'violations': [], 'risk_level': 'LOW'},
            'selected_variables': confirmed_vars
        }

    async def process_query_async(self, user_input: str, conversation_context: Optional[str] = None) -> Dict[str, Any]:
        """
        메인 처리 로직 (비동기 버전 - 성능 최적화)

        Args:
            user_input: 사용자 입력
            conversation_context: 이전 대화 컨텍스트

        Returns:
            처리 결과
        """
        # 컨텍스트 처리
        if conversation_context:
            user_input = self.conversation_manager.handle_conversation_context(
                user_input, conversation_context
            )

        is_follow_up = conversation_context is not None

        # 대화 상태 초기화 또는 유지
        if not is_follow_up or self.conversation_state is None:
            self.conversation_state = self.conversation_manager.initialize_conversation_state(
                user_input, conversation_context
            )

        # 1. 병렬 검증 (성능 최적화 - 장난 감지 추가)
        validation_task = asyncio.create_task(
            asyncio.to_thread(self.input_validator.validate_comprehensive, user_input)
        )
        profanity_task = asyncio.create_task(
            asyncio.to_thread(self.policy_checker.check_profanity, user_input)
        )
        joke_detection_task = asyncio.create_task(
            self.joke_detector.detect_joke_async(user_input)
        )

        validation_result, profanity_result, joke_result = await asyncio.gather(
            validation_task, profanity_task, joke_detection_task
        )

        # 검증 결과 처리 (Fast Fail 순서 적용)
        # 1) 장난 감지 (Fast Fail - 가장 먼저 처리)
        if joke_result.is_joke and self.joke_detector.is_high_confidence_joke(joke_result, threshold=0.7):
            return {
                'status': 'inappropriate_request',
                'message': joke_result.friendly_message,
                'error_code': 'JOKE_DETECTED',
                'category': joke_result.category,
                'confidence': joke_result.confidence,
                'reason': joke_result.reason,
                'original_input': user_input
            }

        # 2) 언어/입력 검증
        if not validation_result.is_valid:
            if validation_result.category in ['english_only', 'no_korean']:
                return {
                    'status': 'inappropriate_request',
                    'message': validation_result.message,
                    'error_code': 'LANGUAGE_VALIDATION_ERROR',
                    'original_input': user_input
                }
            else:
                return {
                    'status': 'inappropriate_request',
                    'message': validation_result.message,
                    'error_code': 'INAPPROPRIATE_REQUEST',
                    'original_input': user_input
                }

        # 3) 비속어 검사
        if profanity_result.has_profanity:
            return {
                'status': 'profanity_retry',
                'message': "비속어가 검출되었습니다. 다시 입력해주세요.",
                'retry_type': 'profanity',
                'original_input': user_input
            }

        # 2. 비동기 질의 분석
        analysis_result = await self.analyze_query_async(user_input)
        if not analysis_result['success']:
            return {
                'status': 'analysis_failed',
                'message': "질의 분석에 실패했습니다. 다시 시도해주세요.",
                'error': analysis_result.get('error'),
                'original_input': user_input
            }

        new_variables = analysis_result['variables']

        # 3. 변수 업데이트
        if is_follow_up:
            self.smart_variable_update(user_input, new_variables)
        else:
            self.conversation_state = self.conversation_manager.update_variables(
                self.conversation_state, new_variables
            )

        # 4. 완성도 판단
        completeness_result = self.conversation_manager.ai_judge_completeness(
            self.conversation_state, analysis_result['intent']
        )

        # 5. 변수 완성도 체크 - reask_required 제거, 항상 진행 (비동기 버전)
        # 부족한 변수가 있어도 Agent2에서 추론하도록 변경
        if not self.conversation_manager.should_proceed_to_template_generation(completeness_result):
            # 기본값으로 채우거나 빈 값으로 진행
            for needed_var in completeness_result.needed_variables:
                if needed_var not in analysis_result['variables'] or not analysis_result['variables'][needed_var]:
                    analysis_result['variables'][needed_var] = "추론 필요"  # Agent2에서 처리하도록 마킹

        # 6. 최종 정책 검사
        confirmed_vars = self.conversation_manager.get_confirmed_variables(self.conversation_state)
        combined_text = " ".join([v for v in confirmed_vars.values()])

        comprehensive_policy_result = await asyncio.to_thread(
            self.policy_checker.check_comprehensive,
            combined_text, self.conversation_state.variables
        )

        # 7. 위반사항 처리
        if not comprehensive_policy_result.is_compliant:
            violation_msg = self.policy_checker.get_violation_message(comprehensive_policy_result)
            self.conversation_state = None  # 상태 초기화
            return {
                'status': 'policy_violation',
                'message': f"정책 위반이 감지되었습니다:\n\n{violation_msg}\n\n프롬프트를 다시 작성해주세요.",
                'violations': comprehensive_policy_result.violations,
                'restart_required': True
            }

        # 8. 모든 검사 통과 - 성공 결과 반환
        return {
            'status': 'success',
            'message': "모든 검사를 통과했습니다. 템플릿 생성이 가능합니다.",
            'analysis': analysis_result,
            'variables': self.conversation_state.variables,
            'intent': analysis_result['intent'],
            'policy_result': {'is_compliant': True, 'violations': [], 'risk_level': 'LOW'},
            'selected_variables': confirmed_vars
        }

    def _parse_completion_response(self, response_text: str, expected_field: str = 'needed_variables') -> Dict[str, Any]:
        """AI 응답 파싱 (하위 호환성)"""
        # ConversationManager로 위임
        return self.conversation_manager._parse_completion_response(response_text)

    def reset_conversation(self):
        """대화 상태 리셋"""
        self.conversation_state = None

    def get_conversation_status(self) -> Dict[str, Any]:
        """현재 대화 상태 조회"""
        if not self.conversation_state:
            return {
                'has_active_conversation': False,
                'completion_percentage': 0.0,
                'confirmed_variables': {},
                'missing_variables': []
            }

        return {
            'has_active_conversation': True,
            'completion_percentage': self.conversation_state.completion_percentage,
            'confirmed_variables': self.conversation_manager.get_confirmed_variables(self.conversation_state),
            'missing_variables': self.conversation_manager.get_missing_variables(self.conversation_state)
        }