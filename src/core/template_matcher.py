#!/usr/bin/env python3
"""
Template Matcher - AI.png 구조에 맞는 키워드 기반 기존 템플릿 검색 모듈
사용자 입력과 기존 템플릿을 비교하여 유사한 템플릿을 찾고 추천하는 시스템
"""
import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from .base_processor import BaseTemplateProcessor


class TemplateMatcher(BaseTemplateProcessor):
    """키워드 기반 기존 템플릿 검색 및 매칭 시스템"""
    
    def __init__(self, api_key: str, gemini_model: str = "gemini-2.0-flash-exp", index_manager=None):
        super().__init__(api_key, gemini_model)
        
        # 인덱스 매니저 연동 (캐싱 최적화)
        self.index_manager = index_manager
        
        # 템플릿 인덱스 파일 경로
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.template_index_file = os.path.join(base_path, "cache", "templates.faiss")
        self.template_vectors_file = os.path.join(base_path, "cache", "templates_vectors.npy")
        self.template_metadata_file = os.path.join(base_path, "cache", "metadata.json")
        
        # 템플릿 데이터베이스
        self.template_database = []
        self.template_embeddings = None
        self.template_index = None
        
        # 유사도 임계값 설정
        self.similarity_threshold = 0.75  # 75% 이상 유사시 기존 템플릿 추천
        self.keyword_threshold = 0.6     # 60% 이상 유사시 관련 템플릿으로 분류
        
        print("TemplateMatcher 초기화 완료")
        
        # 초기화 시 기존 템플릿 로드
        self._initialize_template_database()
    
    def _initialize_template_database(self):
        """템플릿 데이터베이스 초기화"""
        try:
            # 1. 샘플 템플릿 로드
            self._load_sample_templates()
            
            # 2. 캐시된 임베딩 로드 또는 생성
            self._load_or_create_embeddings()
            
            print(f"템플릿 데이터베이스 초기화 완료: {len(self.template_database)}개 템플릿")
            
        except Exception as e:
            print(f"템플릿 데이터베이스 초기화 실패: {e}")
            # 폴백: 빈 데이터베이스로 시작
            self.template_database = []
            self.template_embeddings = None
    
    def _load_sample_templates(self):
        """샘플 템플릿을 데이터베이스에 로드"""
        # 1. 기존 sample_templates.py에서 로드
        from ..utils.sample_templates import get_sample_templates
        
        sample_templates = get_sample_templates()
        
        for idx, template_content in enumerate(sample_templates):
            # 템플릿에서 키워드 추출
            keywords = self._extract_keywords_from_template(template_content)
            
            # 카테고리 자동 분류
            category = self._classify_template_category(template_content)
            
            template_data = {
                "id": f"sample_{idx}",
                "content": template_content,
                "category": category,
                "keywords": keywords,
                "source": "sample_templates",
                "usage_count": 0,
                "last_used": None,
                "quality_score": 0.9  # 샘플 템플릿은 높은 품질
            }
            
            self.template_database.append(template_data)
        
        # 2. JSON 파일에서 기존 템플릿 로드
        self._load_templates_from_json()
    
    def _load_templates_from_json(self):
        """JSON 파일에서 기존 템플릿 로드"""
        try:
            # JSON 파일 경로
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            json_file_path = os.path.join(base_path, "data", "existing_templates.json")
            
            if not os.path.exists(json_file_path):
                print("existing_templates.json 파일이 없습니다. 샘플 템플릿만 사용합니다.")
                return
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_templates = json.load(f)
            
            print(f"JSON에서 {len(json_templates)}개 템플릿 로드 중...")
            
            for template_data in json_templates:
                # JSON 데이터 검증
                if not all(key in template_data for key in ['id', 'content', 'category']):
                    print(f"템플릿 데이터 누락: {template_data.get('id', 'unknown')}")
                    continue
                
                # 키워드가 없으면 자동 추출
                if 'keywords' not in template_data or not template_data['keywords']:
                    template_data['keywords'] = self._extract_keywords_from_template(template_data['content'])
                
                # 기본값 설정
                template_data.setdefault('usage_count', 0)
                template_data.setdefault('last_used', None)
                template_data.setdefault('quality_score', 0.8)
                template_data.setdefault('source', 'json_file')
                
                self.template_database.append(template_data)
            
            print(f"JSON 템플릿 로드 완료: {len(json_templates)}개")
            
        except FileNotFoundError:
            print("existing_templates.json 파일이 없습니다.")
        except json.JSONDecodeError as e:
            print(f"JSON 파일 형식 오류: {e}")
        except Exception as e:
            print(f"JSON 템플릿 로드 실패: {e}")
    
    def _extract_keywords_from_template(self, template_content: str) -> List[str]:
        """템플릿에서 핵심 키워드 추출"""
        import re
        
        # 제목에서 키워드 추출
        title_match = re.search(r'\[(.*?)\]', template_content)
        title_keywords = []
        if title_match:
            title = title_match.group(1)
            title_keywords = [word.strip() for word in title.split() if len(word.strip()) > 1]
        
        # 본문에서 주요 키워드 추출
        content_keywords = []
        key_patterns = [
            r'예약', r'방문', r'안내', r'확인', r'변경', r'취소',
            r'행사', r'이벤트', r'참가', r'신청', r'등록',
            r'가격', r'할인', r'특가', r'혜택', r'쿠폰',
            r'포인트', r'적립', r'결제', r'구매', r'주문',
            r'배송', r'택배', r'도착', r'완료'
        ]
        
        for pattern in key_patterns:
            if re.search(pattern, template_content):
                content_keywords.append(pattern)
        
        return list(set(title_keywords + content_keywords))
    
    def _classify_template_category(self, template_content: str) -> str:
        """템플릿 카테고리 자동 분류"""
        category_patterns = {
            "예약확인": [r"예약", r"방문", r"확인", r"예약번호"],
            "가격변경": [r"가격", r"변경", r"조정", r"인상"],
            "행사안내": [r"행사", r"이벤트", r"참가", r"개최"],
            "결제안내": [r"결제", r"구매", r"주문", r"포인트"],
            "배송안내": [r"배송", r"택배", r"도착", r"발송"],
            "일반안내": []  # 기본 카테고리
        }
        
        for category, patterns in category_patterns.items():
            if patterns and any(re.search(pattern, template_content) for pattern in patterns):
                return category
        
        return "일반안내"
    
    def _load_or_create_embeddings(self):
        """캐시된 임베딩 로드 또는 새로 생성"""
        try:
            # 캐시된 임베딩 로드 시도
            if (os.path.exists(self.template_vectors_file) and 
                os.path.exists(self.template_metadata_file)):
                
                print("캐시된 템플릿 임베딩 로드 중...")
                self.template_embeddings = np.load(self.template_vectors_file)
                
                with open(self.template_metadata_file, 'r', encoding='utf-8') as f:
                    cached_metadata = json.load(f)
                
                # 캐시 유효성 검증
                if len(cached_metadata) == len(self.template_database):
                    print("캐시된 임베딩 로드 성공")
                    return
            
            # 캐시가 없거나 유효하지 않으면 새로 생성
            print("새로운 템플릿 임베딩 생성 중...")
            self._create_new_embeddings()
            
        except Exception as e:
            print(f"임베딩 로드/생성 실패: {e}")
            self.template_embeddings = None
    
    def _create_new_embeddings(self):
        """새로운 템플릿 임베딩 생성 및 저장"""
        if not self.template_database:
            print("템플릿 데이터베이스가 비어있습니다.")
            return
        
        # 템플릿 내용을 임베딩용 텍스트로 변환
        template_texts = []
        for template_data in self.template_database:
            # 제목 + 키워드 + 일부 내용을 조합하여 임베딩
            content = template_data["content"]
            keywords = " ".join(template_data["keywords"])
            category = template_data["category"]
            
            # 임베딩용 텍스트 구성 (제목과 핵심 내용 위주)
            embedding_text = f"{category} {keywords} {content[:200]}"
            template_texts.append(embedding_text)
        
        try:
            # 배치로 임베딩 생성
            embeddings_result = self.encode_texts(template_texts)
            
            if embeddings_result is not None:
                self.template_embeddings = np.array(embeddings_result)
                
                # 캐시 저장
                self._save_embeddings_cache()
                print("새로운 임베딩 생성 및 저장 완료")
            else:
                print("임베딩 생성 실패")
                
        except Exception as e:
            print(f"임베딩 생성 중 오류: {e}")
            self.template_embeddings = None
    
    def _save_embeddings_cache(self):
        """임베딩 캐시 저장"""
        try:
            # 캐시 디렉토리 생성
            cache_dir = os.path.dirname(self.template_vectors_file)
            os.makedirs(cache_dir, exist_ok=True)
            
            # 임베딩 벡터 저장
            np.save(self.template_vectors_file, self.template_embeddings)
            
            # 메타데이터 저장
            metadata = []
            for template_data in self.template_database:
                metadata.append({
                    "id": template_data["id"],
                    "category": template_data["category"],
                    "keywords": template_data["keywords"],
                    "source": template_data["source"]
                })
            
            with open(self.template_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print("임베딩 캐시 저장 완료")
            
        except Exception as e:
            print(f"캐시 저장 실패: {e}")
    
    def find_similar_templates(self, user_input: str, top_k: int = 5) -> List[Dict]:
        """
        사용자 입력과 유사한 기존 템플릿 검색 (AI.png 핵심 기능)
        
        Args:
            user_input: 사용자 입력 텍스트
            top_k: 반환할 최대 템플릿 수
            
        Returns:
            유사한 템플릿 리스트 (유사도 순으로 정렬)
        """
        if not self.template_database or self.template_embeddings is None:
            print("템플릿 데이터베이스가 초기화되지 않았습니다.")
            return []
        
        try:
            print(f"키워드 기반 유사 템플릿 검색: '{user_input}'")
            
            # 1. 사용자 입력 임베딩
            user_embedding = self.encode_texts([user_input])
            if user_embedding is None:
                print("사용자 입력 임베딩 실패")
                return []
            
            user_vector = np.array(user_embedding[0]).reshape(1, -1)
            
            # 2. 코사인 유사도 계산
            similarities = self._calculate_cosine_similarities(user_vector, self.template_embeddings)
            
            # 3. 유사도 순으로 정렬
            similar_indices = np.argsort(similarities)[::-1][:top_k]
            
            # 4. 결과 구성
            similar_templates = []
            for idx in similar_indices:
                similarity_score = float(similarities[idx])
                
                # 임계값 이상인 것만 포함
                if similarity_score >= self.keyword_threshold:
                    template_data = self.template_database[idx].copy()
                    template_data['similarity_score'] = similarity_score
                    template_data['recommendation_type'] = self._get_recommendation_type(similarity_score)
                    
                    similar_templates.append(template_data)
            
            print(f"검색 결과: {len(similar_templates)}개 유사 템플릿 발견")
            return similar_templates
            
        except Exception as e:
            print(f"유사 템플릿 검색 중 오류: {e}")
            return []
    
    def _calculate_cosine_similarities(self, user_vector: np.ndarray, template_vectors: np.ndarray) -> np.ndarray:
        """코사인 유사도 계산"""
        try:
            # 벡터 정규화
            user_norm = user_vector / np.linalg.norm(user_vector, axis=1, keepdims=True)
            template_norms = template_vectors / np.linalg.norm(template_vectors, axis=1, keepdims=True)
            
            # 코사인 유사도 계산
            similarities = np.dot(user_norm, template_norms.T).flatten()
            
            return similarities
            
        except Exception as e:
            print(f"유사도 계산 오류: {e}")
            return np.zeros(len(template_vectors))
    
    def _get_recommendation_type(self, similarity_score: float) -> str:
        """유사도 점수에 따른 추천 유형 결정"""
        if similarity_score >= self.similarity_threshold:
            return "기존_템플릿_수정_권장"  # 75% 이상: 기존 템플릿 수정해서 사용
        elif similarity_score >= self.keyword_threshold:
            return "참고용_유사_템플릿"      # 60~75%: 참고용으로 제시
        else:
            return "관련성_낮음"            # 60% 미만: 관련성 낮음
    
    def check_duplicate_template(self, user_input: str, strict_threshold: float = 0.9) -> Optional[Dict]:
        """
        중복 템플릿 검사 (매우 높은 유사도)
        
        Args:
            user_input: 사용자 입력
            strict_threshold: 중복 판정 임계값 (기본 90%)
            
        Returns:
            중복된 템플릿 데이터 또는 None
        """
        similar_templates = self.find_similar_templates(user_input, top_k=1)
        
        if (similar_templates and 
            similar_templates[0]['similarity_score'] >= strict_threshold):
            
            print(f"중복 템플릿 감지: {similar_templates[0]['similarity_score']:.2%} 유사")
            return similar_templates[0]
        
        return None
    
    def add_new_template(self, template_content: str, metadata: Dict = None) -> bool:
        """
        새로운 템플릿을 데이터베이스에 추가
        
        Args:
            template_content: 템플릿 내용
            metadata: 추가 메타데이터
            
        Returns:
            추가 성공 여부
        """
        try:
            # 새 템플릿 데이터 구성
            new_template = {
                "id": f"user_{len(self.template_database)}",
                "content": template_content,
                "category": self._classify_template_category(template_content),
                "keywords": self._extract_keywords_from_template(template_content),
                "source": "user_generated",
                "usage_count": 1,
                "last_used": None,
                "quality_score": metadata.get("quality_score", 0.8) if metadata else 0.8
            }
            
            if metadata:
                new_template.update(metadata)
            
            # 데이터베이스에 추가
            self.template_database.append(new_template)
            
            # 임베딩 재계산 (새 템플릿 포함)
            self._create_new_embeddings()
            
            print(f"새 템플릿 추가 완료: {new_template['id']}")
            return True
            
        except Exception as e:
            print(f"템플릿 추가 실패: {e}")
            return False
    
    def get_template_statistics(self) -> Dict:
        """템플릿 데이터베이스 통계 정보"""
        if not self.template_database:
            return {"total": 0, "categories": {}, "sources": {}}
        
        # 카테고리별 집계
        categories = {}
        sources = {}
        
        for template in self.template_database:
            category = template.get("category", "unknown")
            source = template.get("source", "unknown")
            
            categories[category] = categories.get(category, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total": len(self.template_database),
            "categories": categories,
            "sources": sources,
            "embedding_status": "loaded" if self.template_embeddings is not None else "not_loaded"
        }


def test_template_matcher():
    """TemplateMatcher 테스트 함수"""
    from config import GEMINI_API_KEY
    
    print("TemplateMatcher 테스트 시작")
    print("=" * 50)
    
    try:
        matcher = TemplateMatcher(GEMINI_API_KEY)
        
        # 통계 정보 출력
        stats = matcher.get_template_statistics()
        print(f"템플릿 DB 통계: {stats}")
        
        # 테스트 쿼리들
        test_queries = [
            "카페 예약 확인 메시지 만들어줘",
            "상품 가격이 올랐다고 안내해줘", 
            "이벤트 참가 신청 안내 템플릿",
            "완전히 새로운 서비스 소개",
        ]
        
        for query in test_queries:
            print(f"\n검색: '{query}'")
            similar = matcher.find_similar_templates(query, top_k=3)
            
            if similar:
                for i, template in enumerate(similar, 1):
                    print(f"{i}. 유사도: {template['similarity_score']:.2%}")
                    print(f"   카테고리: {template['category']}")
                    print(f"   추천유형: {template['recommendation_type']}")
                    print(f"   키워드: {', '.join(template['keywords'])}")
            else:
                print("유사한 템플릿 없음 (신규 생성 필요)")
        
        print("\nTemplateMatcher 테스트 완료")
        return True
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        return False


if __name__ == "__main__":
    test_template_matcher()