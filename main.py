import json
import re
from pathlib import Path
from typing import Dict

from config import GEMINI_API_KEY
from src.core import EntityExtractor, TemplateGenerator
from src.core.index_manager import get_index_manager
from src.core.template_matcher import TemplateMatcher
from src.utils import DataProcessor
from src.agents.agent2 import Agent2


class TemplateSystem:

    def __init__(self):
        print(" TemplateSystem 초기화 시작...")

        # 공통 초기화 모듈 사용
        from src.utils.common_init import (
            initialize_core_components,
            setup_guidelines_and_indexes,
        )

        (
            self.index_manager,
            self.entity_extractor,
            self.template_generator,
            self.data_processor,
            self.agent2,
        ) = initialize_core_components()

        # 키워드 기반 템플릿 검색 모듈 추가
        self.template_matcher = TemplateMatcher(
            api_key=GEMINI_API_KEY, index_manager=self.index_manager
        )

        self.templates = self._load_sample_templates()
        self.guidelines = setup_guidelines_and_indexes(
            self.index_manager, self.entity_extractor, self.template_generator
        )

        print(" TemplateSystem 준비 완료")

    def _load_sample_templates(self) -> list:
        """샘플 템플릿 로드"""
        from src.utils.sample_templates import get_sample_templates

        return get_sample_templates()

    def generate_template(self, user_input: str) -> dict:
        """
        완전한 워크플로우 기반 템플릿 생성
        1. Agent1 검증 (merge 후 활성화)
        2. 키워드 기반 기존 템플릿 검색
        3. 기존 템플릿이 없으면 Agent2로 새 템플릿 생성

        Args:
            user_input: 사용자 입력
        """
        print(f"\n 전체 워크플로우 시작: '{user_input}'")

        # Step 1: Agent1 가이드라인 검증 (TODO: Agent1 merge 후 활성화)
        # agent1_result = self.agent1.validate_guidelines(user_input)
        # if not agent1_result["passed"]:
        #     return {
        #         "status": "rejected",
        #         "reason": "guideline_violation",
        #         "feedback": agent1_result["feedback"]
        #     }

        # Step 2: 키워드 기반 기존 템플릿 검색
        print("\n 기존 템플릿 검색 중...")
        similar_templates = self.template_matcher.find_similar_templates(
            user_input, top_k=3
        )

        # if similar_templates:
        #     # 높은 유사도 템플릿이 있으면 기존 템플릿 추천
        #     best_match = similar_templates[0]
        #     if best_match['similarity_score'] >= 0.75:  # 75% 이상 유사
        #         print(f" 기존 템플릿 발견 (유사도: {best_match['similarity_score']:.1%})")
        #         return {
        #             "status": "existing_template_found",
        #             "recommendation": "기존 템플릿 수정 권장",
        #             "similar_templates": similar_templates,
        #             "best_match": best_match,
        #             "user_input": user_input
        #         }

        # Step 3: 기존 템플릿이 없거나 유사도가 낮으면 Agent2로 새 템플릿 생성
        print("\n 기존 템플릿 없음 - 새 템플릿 생성...")

        # Agent2로 가이드라인과 법령 완벽 준수 템플릿 생성
        template, metadata = self.agent2.generate_compliant_template(user_input)

        # 엔티티 추출 (기존 시스템과의 호환성)
        entities = self.entity_extractor.extract_entities(user_input)

        # 변수 추출
        variables = self.template_generator.extract_variables(template)

        # 새로 생성된 템플릿을 데이터베이스에 추가 (향후 검색용)
        self.template_matcher.add_new_template(
            template_content=template,
            metadata={
                "user_input": user_input,
                "quality_score": 0.8,
                "method": "Agent2",
            },
        )

        return {
            "status": "new_template_generated",
            "user_input": user_input,
            "generated_template": template,
            "entities": entities,
            "variables": variables,
            "generation_method": "Agent2_Compliant_Generation",
            "agent2_metadata": metadata,
            "similar_templates": similar_templates,  # 참고용으로 포함
            "quality_assured": True,
            "guidelines_compliant": True,
            "legal_compliant": True,
        }


def main():
    """메인 실행 함수 - 간단한 템플릿 생성"""
    print(" 알림톡 템플릿 생성기")
    print("=" * 50)

    try:
        system = TemplateSystem()
        print(" 시스템 준비 완료\n")
    except Exception as e:
        print(f" 시스템 초기화 실패: {e}")
        return

    while True:
        print(" 알림톡 내용을 설명해주세요:")

        user_input = input("\n ").strip()

        if user_input.lower() in ["quit", "exit", "종료"]:
            print(" 시스템을 종료합니다.")
            break

        if user_input:
            try:
                print(f"\n 사용자 입력: '{user_input}'")
                result = system.generate_template(user_input)

                # if result["status"] == "existing_template_found":
                #     # 기존 템플릿 발견 시
                #     print("\n 기존 유사 템플릿 발견!")
                #     print("=" * 50)
                #     best_match = result["best_match"]
                #     print(f" 유사도: {best_match['similarity_score']:.1%}")
                #     print(f" 카테고리: {best_match['category']}")
                #     print(f" 키워드: {', '.join(best_match['keywords'])}")
                #     print("\n 추천: 아래 기존 템플릿을 수정해서 사용하세요:")
                #     print("-" * 30)
                #     print(best_match['content'][:200] + "...")
                #     print("=" * 50)

                if result["status"] == "new_template_generated":
                    # 새 템플릿 생성 시
                    print("\n 새 템플릿 생성 완료!")
                    print("=" * 50)
                    print(result["generated_template"])
                    print("=" * 50)

                    print(f"\n 추출된 변수 ({len(result['variables'])}개):")
                    print(f"   {', '.join(result['variables'])}")

                    print(f"\n 추출된 정보:")
                    entities = result["entities"]
                    extracted = entities.get("extracted_info", {})
                    if extracted.get("dates"):
                        print(f"    날짜: {', '.join(extracted['dates'])}")
                    if extracted.get("names"):
                        print(f"    이름: {', '.join(extracted['names'])}")
                    if extracted.get("locations"):
                        print(f"    장소: {', '.join(extracted['locations'])}")
                    if extracted.get("events"):
                        print(f"    이벤트: {', '.join(extracted['events'])}")

                    # 유사 템플릿 정보도 표시
                    if result["similar_templates"]:
                        print(
                            f"\n 참고용 유사 템플릿 ({len(result['similar_templates'])}개):"
                        )
                        for i, template in enumerate(result["similar_templates"], 1):
                            print(
                                f"   {i}. {template['category']} (유사도: {template['similarity_score']:.1%})"
                            )

                elif result["status"] == "rejected":
                    # Agent1 검증 실패 시 (Agent1 merge 후 활성화)
                    print("\n 입력이 가이드라인에 위배됩니다:")
                    print(result["feedback"])

                print("\n" + "=" * 50 + "\n")

            except Exception as e:
                print(f" 오류: {e}\n")
        else:
            print(" 입력이 비어있습니다.\n")


if __name__ == "__main__":
    main()
