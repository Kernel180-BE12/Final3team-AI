"""
공통 초기화 모듈
main.py와 api.py의 중복된 초기화 패턴 통합
"""

from config import GEMINI_API_KEY
from ..core import EntityExtractor
from ..core.index_manager import get_index_manager
from ..utils import DataProcessor
from ..agents.agent2 import Agent2


def initialize_core_components(api_key: str = None):
    """
    핵심 컴포넌트들을 초기화하고 반환
    
    Returns:
        튜플: (index_manager, entity_extractor, data_processor, agent2)
    """
    if api_key is None:
        api_key = GEMINI_API_KEY
    
    print(" 핵심 컴포넌트 초기화 시작...")
    
    # 인덱스 매니저 초기화
    index_manager = get_index_manager()
    
    # 컴포넌트 초기화 (TemplateGenerator 제거)
    entity_extractor = EntityExtractor(api_key)
    data_processor = DataProcessor()
    
    # Agent2 추가 - 가이드라인과 법령 기반 템플릿 생성 (데이터 공유)
    agent2 = Agent2(api_key, index_manager=index_manager)
    
    print(" 핵심 컴포넌트 초기화 완료")
    
    return index_manager, entity_extractor, data_processor, agent2


def setup_guidelines_and_indexes(index_manager, entity_extractor):
    """
    가이드라인 로드 및 인덱스 구축
    
    Args:
        index_manager: 인덱스 매니저 인스턴스
        entity_extractor: 엔티티 추출기 인스턴스  
        template_generator: 템플릿 생성기 인스턴스
    """
    # 캐시된 가이드라인 로드
    guidelines = index_manager.get_guidelines_chunks(
        chunk_func=entity_extractor.chunk_text,
        chunk_size=800,
        overlap=100
    )
    print(f" 가이드라인 로드 완료: {len(guidelines)}개 청크")
    
    # 샘플 템플릿 로드
    from .sample_templates import get_sample_templates
    templates = get_sample_templates()
    
    # 인덱스 구축
    if templates:
        print(" 템플릿 인덱스 준비 중...")
        import re
        clean_templates = []
        for template in templates:
            clean_template = re.sub(r"#\{[^}]+\}", "{VARIABLE}", template)
            clean_templates.append(clean_template)

        template_generator.template_collection = index_manager.get_chroma_collection(
            collection_name="templates",
            data=clean_templates,
            encode_func=template_generator.encode_texts
        )
        template_generator.templates = templates
        print(" 템플릿 인덱스 준비 완료")

    # 가이드라인 인덱스  
    if guidelines:
        print(" 가이드라인 인덱스 준비 중...")
        entity_extractor.guideline_collection = index_manager.get_chroma_collection(
            collection_name="guidelines",
            data=guidelines,
            encode_func=entity_extractor.encode_texts
        )
        entity_extractor.guidelines = guidelines
        print(" 가이드라인 인덱스 준비 완료")
    
    print(" 모든 인덱스 준비 완료")
    
    return guidelines