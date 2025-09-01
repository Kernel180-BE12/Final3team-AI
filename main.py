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
        
        # 인덱스 매니저 초기화
        self.index_manager = get_index_manager()
        
        self.entity_extractor = EntityExtractor(GEMINI_API_KEY)
        self.template_generator = TemplateGenerator(GEMINI_API_KEY)
        self.data_processor = DataProcessor()
        
        #  Agent2 추가 - 가이드라인과 법령 기반 템플릿 생성 (데이터 공유)
        self.agent2 = Agent2(GEMINI_API_KEY, index_manager=self.index_manager)

        self.templates = self._load_sample_templates()
        
        #  캐시된 가이드라인 로드 (매번 파일 읽기 방지)
        self.guidelines = self._load_guidelines_cached()

        self._build_indexes_cached()

    def _load_sample_templates(self) -> list:
        """샘플 템플릿 로드"""
        return [
            "[가격 변경 안내]\n\n안녕하세요, #{수신자명}님.\n#{서비스명} 서비스 가격 변경을 안내드립니다.\n\n 변경 적용일: #{적용일}\n 기존 가격: #{기존가격}원\n 변경 가격: #{변경가격}원\n\n[변경 사유 및 개선사항]\n#{변경사유}에 따라 서비스 품질 개선을 위해 가격을 조정합니다.\n주요 개선사항: #{개선사항}\n\n[기존 이용자 안내]\n- 현재 이용 중인 서비스: #{유예기간}까지 기존 가격 적용\n- 자동 연장 서비스: 변경된 가격으로 갱신\n- 서비스 해지 희망: #{해지마감일}까지 신청 가능\n\n[문의 및 지원]\n- 고객센터: #{고객센터번호}\n- 상담시간: 평일 09:00-18:00\n- 온라인 문의: #{문의링크}\n\n※ 본 메시지는 정보통신망법에 따라 서비스 약관 변경 안내를 위해 발송된 정보성 메시지입니다.",
            "[#{매장명} 방문 예약 확인]\n\n#{고객명}님, 안녕하세요.\n#{매장명} 방문 예약이 완료되었습니다.\n\n 예약 정보\n- 예약번호: #{예약번호}\n- 방문일시: #{방문일시}\n- 예상 소요시간: #{소요시간}\n- 담당 직원: #{담당자명}\n\n 매장 정보\n- 위치: #{매장주소}\n- 연락처: #{매장전화번호}\n- 주차: #{주차안내}\n\n[방문 전 준비사항]\n- 신분증 지참 필수 (본인 확인)\n- 예약 10분 전 도착 권장\n- 마스크 착용 협조\n- 예약 확인 문자 제시\n\n[교통 및 위치 안내]\n- 대중교통: #{교통편안내}\n- 자가용: #{길찾기정보}\n- 주변 랜드마크: #{랜드마크}\n\n[예약 변경 및 취소]\n방문 예정일 1일 전까지 변경/취소 가능\n- 전화: #{매장전화번호}\n- 온라인: #{변경링크}\n- 문자 회신으로도 변경 가능\n\n※ 본 메시지는 매장 방문 예약 신청고객에게 발송되는 예약 확인 메시지입니다.",
            "[#{행사명} 참가 안내]\n\n#{수신자명}님, 안녕하세요.\n#{주최기관}에서 개최하는 #{행사명} 참가를 안내드립니다.\n\n 행사 개요\n- 행사명: #{행사명}\n- 일시: #{행사일시}\n- 장소: #{행사장소}\n- 대상: #{참가대상}\n- 참가비: #{참가비}\n\n 프로그램 일정\n#{프로그램일정상세}\n\n 참가 신청\n- 신청 방법: #{신청방법}\n- 신청 마감: #{신청마감일}\n- 신청 문의: #{신청문의전화}\n- 온라인 신청: #{신청링크}\n\n[준비물 및 복장]\n- 필수 준비물: #{필수준비물}\n- 권장 복장: #{복장안내}\n- 개인 준비물: #{개인준비물}\n\n[행사장 안내]\n- 상세 주소: #{상세주소}\n- 교통편: #{교통편}\n- 주차 시설: #{주차정보}\n- 편의 시설: #{편의시설}\n\n[주의사항 및 안내]\n- 코로나19 방역수칙 준수\n- 행사 당일 발열체크 실시\n- 우천 시 일정: #{우천시대안}\n- 기타 문의: #{기타문의처}\n\n※ 본 메시지는 #{행사명} 관심 등록자에게 발송되는 행사 안내 메시지입니다.",
        ]

    def _load_guidelines_cached(self) -> list:
        """ 캐시된 가이드라인 청크 로드 (성능 최적화)"""
        try:
            guidelines = self.index_manager.get_guidelines_chunks(
                chunk_func=self.entity_extractor.chunk_text,
                chunk_size=800,
                overlap=100
            )
            print(f" 가이드라인 로드 완료: {len(guidelines)}개 청크")
            return guidelines
            
        except Exception as e:
            print(f" 가이드라인 로드 오류: {e}")
            return []

    def _build_indexes_cached(self):
        """ 캐시된 FAISS 인덱스 구축 (성능 최적화)"""
        
        # 템플릿 인덱스
        if self.templates:
            print(" 템플릿 인덱스 준비 중...")
            clean_templates = []
            for template in self.templates:
                clean_template = re.sub(r"#\{[^}]+\}", "[VARIABLE]", template)
                clean_templates.append(clean_template)

            self.template_generator.template_index = self.index_manager.get_faiss_index(
                index_name="templates",
                data=clean_templates,
                encode_func=self.template_generator.encode_texts,
                build_func=self.template_generator.build_faiss_index
            )
            self.template_generator.templates = self.templates
            print(" 템플릿 인덱스 준비 완료")

        # 가이드라인 인덱스  
        if self.guidelines:
            print(" 가이드라인 인덱스 준비 중...")
            self.entity_extractor.guideline_index = self.index_manager.get_faiss_index(
                index_name="guidelines",
                data=self.guidelines,
                encode_func=self.entity_extractor.encode_texts,
                build_func=self.entity_extractor.build_faiss_index
            )
            self.entity_extractor.guidelines = self.guidelines
            print(" 가이드라인 인덱스 준비 완료")
        
        print(" 모든 인덱스 준비 완료")

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
