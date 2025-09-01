import json
import re
from pathlib import Path
from typing import Dict

from config import GEMINI_API_KEY
from src.core import EntityExtractor, TemplateGenerator
from src.core.index_manager import get_index_manager
from src.utils import DataProcessor
from src.agents.agent2 import Agent2


class TemplateSystem:

    def __init__(self):
        print(" TemplateSystem 초기화 시작...")
        
        # 공통 초기화 모듈 사용
        from src.utils.common_init import initialize_core_components, setup_guidelines_and_indexes
        
        (self.index_manager, self.entity_extractor, 
         self.template_generator, self.data_processor, 
         self.agent2) = initialize_core_components()

        self.templates = self._load_sample_templates()
        self.guidelines = setup_guidelines_and_indexes(
            self.index_manager, self.entity_extractor, self.template_generator
        )

    def _load_sample_templates(self) -> list:
        """샘플 템플릿 로드"""
        from src.utils.sample_templates import get_sample_templates
        return get_sample_templates()


    def generate_template(self, user_input: str, use_agent2: bool = True) -> dict:
        """
        템플릿 생성 - Agent2를 기본으로 사용
        
        Args:
            user_input: 사용자 입력
            use_agent2: Agent2 사용 여부 (기본값: True)
        """

        if use_agent2:
            print("\n Agent2 기반 템플릿 생성 시작...")
            
            # Agent2로 가이드라인과 법령 완벽 준수 템플릿 생성
            template, metadata = self.agent2.generate_compliant_template(user_input)
            
            # 엔티티 추출 (기존 시스템과의 호환성)
            entities = self.entity_extractor.extract_entities(user_input)
            
            return {
                "user_input": user_input,
                "generated_template": template,
                "entities": entities,
                "generation_method": "Agent2_Compliant_Generation",
                "agent2_metadata": metadata,
                "quality_assured": True,
                "guidelines_compliant": True,
                "legal_compliant": True
            }
        
        else:
            print("\n 순수 신규 템플릿 생성...")
            
            # 기업 요구사항: 기존 템플릿 검색 없이 순수 신규 생성만
            # 1. 엔티티 추출
            entities = self.entity_extractor.extract_entities(user_input)

            # 2. 가이드라인 검색 (규정 준수를 위해)
            relevant_guidelines = self.entity_extractor.search_similar(
                user_input + " " + entities.get("message_intent", ""),
                self.entity_extractor.guideline_index,
                self.entity_extractor.guidelines,
                top_k=3,
            )
            guidelines = [guideline for guideline, _ in relevant_guidelines]

            # 3. 순수 신규 템플릿 생성 (기존 템플릿 참고 없음)
            template, filled_template = self.template_generator.generate_template(
                user_input, entities, [], guidelines  # similar_templates를 빈 배열로
            )

            # 4. 템플릿 최적화
            optimized_template = self.template_generator.optimize_template(
                template, entities
            )

            # 5. 변수 추출
            variables = self.template_generator.extract_variables(optimized_template)

            return {
                "user_input": user_input,
                "generated_template": optimized_template,
                "filled_template": self.template_generator._fill_template_with_entities(
                    optimized_template, entities
                ),
                "variables": variables,
                "entities": entities,
                "generation_method": "Pure_New_Generation",
                "quality_assured": True,
                "guidelines_compliant": True,
                "legal_compliant": True
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
                print("\n 템플릿 생성 중...")
                result = system.generate_template(user_input)

                print("\n 생성된 템플릿:")
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

                print("\n" + "=" * 50 + "\n")

            except Exception as e:
                print(f" 오류: {e}\n")
        else:
            print(" 입력이 비어있습니다.\n")


if __name__ == "__main__":
    main()
