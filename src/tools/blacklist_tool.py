#!/usr/bin/env python3
"""
BlackList Tool - 금지어 패턴 기반 안전한 템플릿 생성 가이드
Agent2의 첫 번째 도구
"""
import os
from pathlib import Path
from typing import Dict, List, Any
from langchain.tools import BaseTool

class BlackListTool(BaseTool):
    """
    금지어 및 위험 패턴을 검출하여 안전한 템플릿 생성을 가이드하는 도구
    """
    name = "blacklist_generator_guide"
    description = "금지어 패턴 기반 안전한 템플릿 생성 가이드"
    
    def __init__(self, index_manager=None):
        super().__init__()
        self.index_manager = index_manager
        self._blacklist_data = None
        
    def _get_blacklist_data(self) -> str:
        """BlackList 데이터 로드 (캐시 사용)"""
        if self._blacklist_data is None:
            try:
                if self.index_manager:
                    # IndexManager에서 캐시된 데이터 사용
                    predata = self.index_manager.get_predata_cache()
                    self._blacklist_data = predata.get("cleaned_black_list.md", "")
                else:
                    # 직접 파일 로드
                    base_path = Path(__file__).parent.parent
                    blacklist_path = base_path / "predata" / "cleaned_black_list.md"
                    
                    if blacklist_path.exists():
                        with open(blacklist_path, 'r', encoding='utf-8') as f:
                            self._blacklist_data = f.read()
                    else:
                        self._blacklist_data = ""
            except Exception as e:
                print(f" BlackList 데이터 로드 실패: {e}")
                self._blacklist_data = ""
        
        return self._blacklist_data
    
    def _identify_risk_keywords(self, user_input: str) -> List[str]:
        """위험 키워드 식별"""
        risk_keywords = []
        
        # 고위험 패턴들
        high_risk_patterns = [
            "할인", "이벤트", "무료", "특가", "프로모션", 
            "경품", "추첨", "당첨", "선착순", "한정"
        ]
        
        # 광고성 키워드
        advertising_keywords = [
            "혜택", "쿠폰", "포인트", "적립", "캐시백",
            "세일", "오픈", "신규", "가입"
        ]
        
        # 스팸성 키워드  
        spam_keywords = [
            "긴급", "마지막", "지금", "즉시", "빨리"
        ]
        
        all_patterns = high_risk_patterns + advertising_keywords + spam_keywords
        
        for pattern in all_patterns:
            if pattern in user_input:
                risk_keywords.append(pattern)
        
        return risk_keywords
    
    def _find_relevant_restrictions(self, user_input: str) -> List[str]:
        """관련 제한사항 찾기"""
        restrictions = []
        blacklist_data = self._get_blacklist_data()
        
        if not blacklist_data:
            return restrictions
        
        # 광고 관련 제한사항
        if any(word in user_input for word in ["할인", "이벤트", "무료", "특가"]):
            if "광고" in blacklist_data:
                restrictions.append("[광고] 표기 의무")
            if "동의" in blacklist_data:
                restrictions.append("수신 동의자에게만 발송")
            if "21시" in blacklist_data or "야간" in blacklist_data:
                restrictions.append("야간 전송 금지 (21시~8시)")
        
        # 스팸 관련 제한사항
        if any(word in user_input for word in ["긴급", "마지막", "즉시"]):
            if "금지" in blacklist_data:
                restrictions.append("스팸성 표현 금지")
            if "위반" in blacklist_data:
                restrictions.append("가이드라인 위반 시 계정 차단")
        
        # 개인정보 관련
        if "개인정보" in user_input or "정보" in user_input:
            if "수집" in blacklist_data:
                restrictions.append("개인정보 수집 시 동의 필수")
        
        return restrictions
    
    def _generate_safe_guidelines(self, user_input: str, risk_keywords: List[str]) -> List[str]:
        """안전한 템플릿 작성 가이드라인 생성"""
        guidelines = []
        
        if risk_keywords:
            guidelines.append("광고성 내용 시 [광고] 표기 필수")
            guidelines.append("과장된 표현 지양")
            guidelines.append("명확하고 정확한 정보 제공")
            
        if any(word in user_input for word in ["할인", "이벤트"]):
            guidelines.append("할인율/혜택 정보 정확히 명시")
            guidelines.append("유효기간 명확히 표기")
            guidelines.append("조건 및 제약사항 안내")
            
        if any(word in user_input for word in ["무료", "공짜"]):
            guidelines.append("무료 제공 조건 명시")
            guidelines.append("숨은 비용 없음 명확히 표기")
            
        # 기본 가이드라인
        guidelines.extend([
            "수신거부 방법 안내",
            "연락처 정보 포함",
            "정중하고 예의바른 표현 사용"
        ])
        
        return list(set(guidelines))  # 중복 제거
    
    def _run(self, user_input: str) -> Dict[str, Any]:
        """BlackList Tool 실행"""
        try:
            # 1. 위험 키워드 식별
            risk_keywords = self._identify_risk_keywords(user_input)
            
            # 2. 관련 제한사항 찾기
            restrictions = self._find_relevant_restrictions(user_input)
            
            # 3. 안전한 가이드라인 생성
            guidelines = self._generate_safe_guidelines(user_input, risk_keywords)
            
            # 4. 위험도 평가
            risk_level = "HIGH" if len(risk_keywords) >= 2 else ("MEDIUM" if risk_keywords else "LOW")
            
            # 5. 규정 준수 여부
            compliance_status = "REVIEW_REQUIRED" if risk_keywords else "COMPLIANT"
            
            blacklist_data = self._get_blacklist_data()
            
            return {
                "tool_name": "BlackList",
                "user_input": user_input,
                "risk_keywords": risk_keywords,
                "risk_level": risk_level,
                "restrictions": restrictions,
                "safe_guidelines": guidelines,
                "compliance_status": compliance_status,
                "data_loaded": len(blacklist_data) > 0,
                "data_size": len(blacklist_data),
                "recommendation": "광고성 내용 시 법령 준수 필수" if risk_keywords else "안전한 내용으로 판단됨"
            }
            
        except Exception as e:
            return {
                "tool_name": "BlackList",
                "error": str(e),
                "user_input": user_input,
                "risk_keywords": [],
                "risk_level": "UNKNOWN",
                "restrictions": [],
                "safe_guidelines": [],
                "compliance_status": "ERROR",
                "data_loaded": False,
                "data_size": 0,
                "recommendation": "도구 실행 중 오류 발생"
            }