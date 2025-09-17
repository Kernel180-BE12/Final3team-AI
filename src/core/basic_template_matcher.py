#!/usr/bin/env python3
"""
BasicTemplateMatcher - 기존 템플릿 검색 시스템
ChromaDB + 임베딩 기반 유사도 검색
"""
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

class BasicTemplateMatcher:
    """기본 템플릿 매칭 시스템 (ChromaDB 기반)"""

    def __init__(self, index_manager=None, entity_extractor=None):
        self.index_manager = index_manager
        self.entity_extractor = entity_extractor
        self.templates = []
        self.template_collection = None
        self.chroma_id_to_template_id = {}  # ChromaDB ID 매핑

        # 템플릿 로드 및 임베딩
        self._load_templates()
        self._create_embeddings()

    def _load_templates(self):
        """temp_fix.json에서 템플릿 로드"""
        try:
            template_path = Path(__file__).parent.parent.parent / "data" / "temp_fix.json"

            with open(template_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            templates = data.get("data", {}).get("templates", [])

            # 템플릿 정규화
            for template in templates:
                processed = self._process_template(template)
                if processed:
                    self.templates.append(processed)

            print(f"✅ {len(self.templates)}개 기본 템플릿 로드 완료")

        except Exception as e:
            print(f"⚠️ 템플릿 로드 실패: {e}")
            self.templates = []

    def _process_template(self, template: Dict) -> Optional[Dict]:
        """템플릿 전처리 및 키워드 추출"""
        try:
            content = template.get("templateContent", "")
            name = template.get("templateName", "")

            if not content or not name:
                return None

            # 변수를 표준 형식으로 변환: #{변수} → ${변수}
            standardized_content = re.sub(r'#\{([^}]+)\}', r'${\1}', content)

            # 키워드 추출 (템플릿명 + 내용에서)
            keywords = self._extract_keywords(name + " " + content)

            # 카테고리 분류
            category = self._classify_template(name)

            return {
                "id": template.get("templateCode", "unknown"),
                "name": name,
                "content": standardized_content,
                "original_content": content,
                "keywords": keywords,
                "category": category,
                "variables": self._extract_variables(template.get("variables", [])),
                "search_text": f"{name} {content}".replace('\n', ' ')  # 검색용 텍스트
            }

        except Exception as e:
            print(f"⚠️ 템플릿 처리 실패: {e}")
            return None

    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 기본 키워드 추출 (한글, 영어)
        keywords = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', text)

        # 불용어 제거
        stopwords = {'안내', '드립니다', '입니다', '있습니다', '감사합니다', '주세요', '확인'}
        keywords = [k for k in keywords if k not in stopwords and len(k) >= 2]

        return list(set(keywords))[:10]  # 최대 10개

    def _classify_template(self, name: str) -> str:
        """템플릿 카테고리 분류"""
        if any(keyword in name for keyword in ["AS", "문의", "상담"]):
            return "고객지원"
        elif any(keyword in name for keyword in ["예약", "일정", "알림"]):
            return "예약관리"
        elif any(keyword in name for keyword in ["수업", "강의", "교육", "세미나"]):
            return "교육"
        elif any(keyword in name for keyword in ["이벤트", "프로모션", "할인"]):
            return "마케팅"
        elif any(keyword in name for keyword in ["결제", "납부", "수수료"]):
            return "결제"
        else:
            return "일반"

    def _extract_variables(self, variables: List[Dict]) -> List[Dict]:
        """변수 정보 추출 및 표준화"""
        extracted = []
        for var in variables:
            key = var.get("key", "")
            if key:
                extracted.append({
                    "name": f"${{{key}}}",
                    "description": key,
                    "required": True,
                    "original_key": key
                })
        return extracted

    def _create_embeddings(self):
        """ChromaDB에 템플릿 임베딩 저장"""
        if not self.index_manager or not self.entity_extractor or not self.templates:
            print("⚠️ 임베딩 생성 건너뛰기 (컴포넌트 부족)")
            return

        try:
            # 검색용 텍스트 추출
            search_texts = [template["search_text"] for template in self.templates]

            # ChromaDB 컬렉션 생성
            self.template_collection = self.index_manager.get_chroma_collection(
                collection_name="basic_templates",
                data=search_texts,
                encode_func=self.entity_extractor.encode_texts
            )

            # ChromaDB ID와 template ID 매핑 생성
            self.chroma_id_to_template_id = {}
            for i, template in enumerate(self.templates):
                chroma_id = f"basic_templates_{i}"
                self.chroma_id_to_template_id[chroma_id] = template["id"]

            print("✅ 템플릿 임베딩 저장 완료")

        except Exception as e:
            print(f"⚠️ 임베딩 생성 실패: {e}")
            self.template_collection = None

    def find_matching_template(self, user_input: str, threshold: float = 0.8) -> Optional[Dict]:
        """사용자 입력에 맞는 템플릿 검색"""
        if not self.template_collection:
            print("⚠️ 임베딩 컬렉션이 없음, 키워드 기반 검색 사용")
            return self._keyword_based_search(user_input, threshold)

        try:
            # 임베딩 기반 검색
            query_embedding = self.entity_extractor.encode_texts([user_input])

            results = self.template_collection.query(
                query_embeddings=query_embedding,
                n_results=5,
                include=['distances']
            )

            if not results['ids'] or not results['ids'][0]:
                return None

            # 가장 유사한 템플릿 찾기
            best_match = None
            best_similarity = 0

            for i, (chroma_id, distance) in enumerate(zip(
                results['ids'][0],
                results['distances'][0]
            )):
                similarity = 1 - distance  # 거리를 유사도로 변환

                # ChromaDB ID를 실제 template ID로 변환
                actual_template_id = self.chroma_id_to_template_id.get(chroma_id)
                template = self._find_template_by_id(actual_template_id) if actual_template_id else None


                if similarity > threshold and similarity > best_similarity:
                    # 원본 템플릿 찾기
                    if template:
                        best_match = {
                            **template,
                            "similarity": similarity,
                            "match_type": "embedding"
                        }
                        best_similarity = similarity

            return best_match

        except Exception as e:
            print(f"⚠️ 임베딩 검색 실패, 키워드 검색으로 대체: {e}")
            return self._keyword_based_search(user_input, threshold)

    def _keyword_based_search(self, user_input: str, threshold: float = 0.3) -> Optional[Dict]:
        """키워드 기반 백업 검색"""
        user_keywords = set(self._extract_keywords(user_input))

        best_match = None
        best_score = 0

        for template in self.templates:
            template_keywords = set(template["keywords"])

            if not user_keywords or not template_keywords:
                continue

            # Jaccard 유사도 계산
            intersection = len(user_keywords.intersection(template_keywords))
            union = len(user_keywords.union(template_keywords))
            similarity = intersection / union if union > 0 else 0

            if similarity > threshold and similarity > best_score:
                best_match = {
                    **template,
                    "similarity": similarity,
                    "match_type": "keyword"
                }
                best_score = similarity

        return best_match

    def _find_template_by_id(self, template_id: str) -> Optional[Dict]:
        """ID로 템플릿 찾기"""
        for template in self.templates:
            if template["id"] == template_id:
                return template
        return None

    def get_templates_by_category(self, category: str) -> List[Dict]:
        """카테고리별 템플릿 조회"""
        return [t for t in self.templates if t["category"] == category]

    def get_statistics(self) -> Dict[str, Any]:
        """템플릿 통계 정보"""
        categories = {}
        for template in self.templates:
            cat = template["category"]
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_templates": len(self.templates),
            "categories": categories,
            "has_embeddings": self.template_collection is not None
        }


# 편의 함수
def get_basic_template_matcher(index_manager=None, entity_extractor=None):
    """BasicTemplateMatcher 인스턴스 반환"""
    return BasicTemplateMatcher(index_manager, entity_extractor)


if __name__ == "__main__":
    # 테스트
    print("=== BasicTemplateMatcher 테스트 ===")

    matcher = BasicTemplateMatcher()
    stats = matcher.get_statistics()

    print(f"총 템플릿: {stats['total_templates']}개")
    print(f"카테고리별:")
    for cat, count in stats['categories'].items():
        print(f"  - {cat}: {count}개")

    # 검색 테스트
    test_queries = [
        "패스트캠퍼스 강의 설명회",
        "수업 일정 안내",
        "예약 확인",
        "AS 문의"
    ]

    for query in test_queries:
        result = matcher.find_matching_template(query)
        if result:
            print(f"\n'{query}' → '{result['name']}' (유사도: {result['similarity']:.3f})")
        else:
            print(f"\n'{query}' → 매칭된 템플릿 없음")