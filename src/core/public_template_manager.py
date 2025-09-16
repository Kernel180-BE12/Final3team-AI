"""
공용 템플릿 검색 및 매칭 시스템
3단계 템플릿 선택 로직의 2단계 담당
"""
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import logging

@dataclass
class PublicTemplate:
    """공용 템플릿 데이터 구조"""
    template_code: str
    template_name: str
    template_content: str
    template_type: str
    variables: List[Dict[str, Any]]
    buttons: List[Dict[str, Any]]
    industries: List[str] = None
    purposes: List[str] = None
    image_url: Optional[str] = None
    
class PublicTemplateManager:
    """
    공용 템플릿 검색 및 매칭 관리자
    """
    
    def __init__(self, template_data_path: str = "data/temp_fix.json"):
        """
        Args:
            template_data_path: 공용 템플릿 데이터 파일 경로
        """
        self.template_data_path = template_data_path
        self.templates: List[PublicTemplate] = []
        self.logger = logging.getLogger(__name__)
        
        # 템플릿 로드
        self._load_templates()
        
    def _load_templates(self):
        """JSON 파일에서 공용 템플릿 로드"""
        try:
            if not os.path.exists(self.template_data_path):
                self.logger.warning(f"템플릿 데이터 파일이 없습니다: {self.template_data_path}")
                return
                
            with open(self.template_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # JSON 구조 파싱
            if isinstance(data, list) and len(data) > 0:
                # data[0]이 response wrapper인 경우
                template_data = data[0].get('data', {}).get('templates', [])
            else:
                template_data = data.get('templates', [])
            
            for template_json in template_data:
                try:
                    template = PublicTemplate(
                        template_code=template_json.get('templateCode', ''),
                        template_name=template_json.get('templateName', ''),
                        template_content=template_json.get('templateContent', ''),
                        template_type=template_json.get('templateEmphasizeType', 'TEXT'),
                        variables=template_json.get('variables', []),
                        buttons=template_json.get('buttons', []),
                        industries=template_json.get('industries', []),
                        purposes=template_json.get('purposes', []),
                        image_url=template_json.get('templateImageUrl')
                    )
                    self.templates.append(template)
                except Exception as e:
                    self.logger.warning(f"템플릿 파싱 실패: {e}")
                    continue
                    
            self.logger.info(f"공용 템플릿 {len(self.templates)}개 로드 완료")
            
        except Exception as e:
            self.logger.error(f"템플릿 로드 실패: {e}")
            self.templates = []
    
    def search_by_keywords(self, user_input: str, similarity_threshold: float = 0.6) -> List[Tuple[PublicTemplate, float]]:
        """
        키워드 기반 템플릿 검색
        
        Args:
            user_input: 사용자 입력
            similarity_threshold: 유사도 임계값
            
        Returns:
            List[(template, similarity_score)] 유사도 순 정렬
        """
        if not self.templates:
            return []
        
        # 키워드 추출
        keywords = self._extract_keywords(user_input)
        
        matches = []
        for template in self.templates:
            # 템플릿명 + 내용에서 유사도 계산
            template_text = f"{template.template_name} {template.template_content}"
            similarity = self._calculate_similarity(keywords, template_text)
            
            if similarity >= similarity_threshold:
                matches.append((template, similarity))
        
        # 유사도 순 정렬
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def search_by_category(self, category_keywords: List[str]) -> List[PublicTemplate]:
        """
        카테고리별 템플릿 검색
        
        Args:
            category_keywords: 카테고리 키워드 리스트
            
        Returns:
            해당 카테고리의 템플릿들
        """
        results = []
        
        for template in self.templates:
            template_text = f"{template.template_name} {template.template_content}".lower()
            
            # 카테고리 키워드 중 하나라도 매칭되면 포함
            for keyword in category_keywords:
                if keyword.lower() in template_text:
                    results.append(template)
                    break
        
        return results
    
    def get_template_by_code(self, template_code: str) -> Optional[PublicTemplate]:
        """템플릿 코드로 특정 템플릿 조회"""
        for template in self.templates:
            if template.template_code == template_code:
                return template
        return None
    
    def get_all_templates(self) -> List[PublicTemplate]:
        """모든 공용 템플릿 반환"""
        return self.templates.copy()
    
    def _extract_keywords(self, user_input: str) -> List[str]:
        """사용자 입력에서 키워드 추출"""
        # 기본적인 키워드 추출 (한국어 처리)
        # 명사 위주로 추출, 불용어 제거
        stopwords = {'을', '를', '이', '가', '은', '는', '의', '에', '로', '으로', '와', '과', '도', '만', '까지', '부터', '에서'}
        
        # 특수문자 제거 및 공백으로 분리
        cleaned = re.sub(r'[^\w\s]', ' ', user_input)
        words = cleaned.split()
        
        # 불용어 제거 및 2글자 이상 키워드만 추출
        keywords = [word for word in words if len(word) >= 2 and word not in stopwords]
        
        return keywords
    
    def _calculate_similarity(self, keywords: List[str], template_text: str) -> float:
        """키워드와 템플릿 텍스트 간 유사도 계산"""
        if not keywords:
            return 0.0
        
        template_text_lower = template_text.lower()
        keyword_text = ' '.join(keywords).lower()
        
        # SequenceMatcher를 사용한 유사도 계산
        similarity1 = SequenceMatcher(None, keyword_text, template_text_lower).ratio()
        
        # 키워드 포함도 계산
        matching_keywords = sum(1 for keyword in keywords if keyword.lower() in template_text_lower)
        keyword_ratio = matching_keywords / len(keywords) if keywords else 0
        
        # 가중평균 (키워드 포함도에 더 높은 가중치)
        final_similarity = (similarity1 * 0.3) + (keyword_ratio * 0.7)
        
        return final_similarity
    
    def convert_to_standard_format(self, template: PublicTemplate) -> Dict[str, Any]:
        """
        공용 템플릿을 표준 생성 템플릿 형식으로 변환
        #{변수명} -> ${변수명} 형식으로 변환
        """
        content = template.template_content
        variables = []
        
        # 기존 변수 추출
        for i, var in enumerate(template.variables, 1):
            key = var.get('key', f'변수{i}')
            placeholder = var.get('placeholder', f'#{{{key}}}')
            
            # 표준 형식으로 변환
            standard_var = f"${{최대15자 {i}}}"
            content = content.replace(placeholder, standard_var)
            
            variables.append({
                "name": standard_var,
                "description": key,
                "required": True
            })
        
        return {
            "template": content,
            "variables": variables,
            "template_type": template.template_type,
            "has_image": template.image_url is not None,
            "has_buttons": len(template.buttons) > 0,
            "source": "public_template",
            "source_code": template.template_code,
            "source_name": template.template_name
        }
    
    def get_template_stats(self) -> Dict[str, Any]:
        """템플릿 통계 정보 반환"""
        if not self.templates:
            return {"total": 0}
        
        # 타입별 분류
        type_count = {}
        has_image = 0
        has_buttons = 0
        
        for template in self.templates:
            template_type = template.template_type
            type_count[template_type] = type_count.get(template_type, 0) + 1
            
            if template.image_url:
                has_image += 1
            if template.buttons:
                has_buttons += 1
        
        return {
            "total": len(self.templates),
            "by_type": type_count,
            "with_image": has_image,
            "with_buttons": has_buttons
        }


# 전역 인스턴스 (싱글톤 패턴)
_global_public_template_manager = None

def get_public_template_manager() -> PublicTemplateManager:
    """전역 공용 템플릿 매니저 인스턴스 반환"""
    global _global_public_template_manager
    if _global_public_template_manager is None:
        _global_public_template_manager = PublicTemplateManager()
    return _global_public_template_manager


if __name__ == "__main__":
    # 테스트
    print("=== 공용 템플릿 매니저 테스트 ===")
    
    try:
        manager = PublicTemplateManager()
        
        # 통계 정보
        stats = manager.get_template_stats()
        print(f"총 템플릿 수: {stats['total']}")
        print(f"타입별: {stats.get('by_type', {})}")
        
        # 키워드 검색 테스트
        print("\n=== 키워드 검색 테스트 ===")
        matches = manager.search_by_keywords("국가 건강검진 안내")
        print(f"검색 결과: {len(matches)}개")
        
        for template, score in matches[:3]:  # 상위 3개만
            print(f"- {template.template_name} (유사도: {score:.3f})")
            
    except Exception as e:
        print(f" 오류: {e}")