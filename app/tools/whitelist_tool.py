#!/usr/bin/env python3
"""
WhiteList Tool - 승인된 패턴 기반 템플릿 최적화 도구  
Agent2의 두 번째 도구
"""
import os
from pathlib import Path
from typing import Dict, List, Any
from langchain.tools import BaseTool

class WhiteListTool(BaseTool):
    """
    승인된 패턴과 베스트 프랙티스를 기반으로 템플릿을 최적화하는 도구
    """
    name: str = "whitelist_template_optimizer"
    description: str = "승인된 패턴 기반 템플릿 최적화"
    
    def __init__(self, index_manager=None):
        super().__init__()
        self._index_manager = index_manager
        self._whitelist_data = None
        
    def _get_whitelist_data(self) -> str:
        """WhiteList 데이터 로드 (캐시 사용)"""
        if self._whitelist_data is None:
            try:
                if self._index_manager:
                    # IndexManager에서 캐시된 데이터 사용
                    predata = self._index_manager.get_predata_cache()
                    self._whitelist_data = predata.get("cleaned_white_list.md", "")
                else:
                    # 직접 파일 로드
                    base_path = Path(__file__).parent.parent.parent
                    whitelist_path = base_path / ".." / "data" / "presets" / "cleaned_white_list.md"
                    
                    if whitelist_path.exists():
                        with open(whitelist_path, 'r', encoding='utf-8') as f:
                            self._whitelist_data = f.read()
                    else:
                        self._whitelist_data = ""
            except Exception as e:
                print(f" WhiteList 데이터 로드 실패: {e}")
                self._whitelist_data = ""
        
        return self._whitelist_data
    
    def _categorize_message_type(self, user_input: str) -> str:
        """메시지 유형 분류"""
        user_input_lower = user_input.lower()
        
        # 예약/방문 관련
        if any(word in user_input_lower for word in ["예약", "방문", "진료", "상담"]):
            return "예약확인"
            
        # 결제/구매 관련  
        elif any(word in user_input_lower for word in ["결제", "구매", "주문", "영수증"]):
            return "결제안내"
            
        # 배송/물류 관련
        elif any(word in user_input_lower for word in ["배송", "택배", "발송", "도착"]):
            return "배송안내"
            
        # 포인트/혜택 관련
        elif any(word in user_input_lower for word in ["포인트", "적립", "혜택", "쿠폰"]):
            return "포인트안내"
            
        # 회원/가입 관련
        elif any(word in user_input_lower for word in ["회원", "가입", "등록", "인증"]):
            return "회원관리"
            
        # 이벤트/프로모션
        elif any(word in user_input_lower for word in ["이벤트", "할인", "특가"]):
            return "이벤트안내"
            
        # 알림/공지
        elif any(word in user_input_lower for word in ["알림", "공지", "안내", "변경"]):
            return "공지사항"
            
        else:
            return "일반안내"
    
    def _find_approved_patterns(self, message_type: str, user_input: str) -> List[str]:
        """승인된 패턴 찾기"""
        patterns = []
        whitelist_data = self._get_whitelist_data()
        
        if not whitelist_data:
            return self._get_default_patterns(message_type)
        
        # 메시지 유형별 승인된 패턴
        type_patterns = {
            "예약확인": ["예약번호 안내", "방문일시 확인", "준비사항 안내", "연락처 제공"],
            "결제안내": ["결제금액 명시", "결제수단 안내", "영수증 제공", "문의처 안내"],  
            "배송안내": ["배송상태 안내", "예상도착일 제공", "배송조회 방법", "수령방법 안내"],
            "포인트안내": ["적립내역 안내", "사용방법 설명", "유효기간 명시", "혜택 안내"],
            "회원관리": ["인증절차 안내", "서비스 이용방법", "개인정보 보호", "문의방법"],
            "이벤트안내": ["이벤트 상세정보", "참여방법 안내", "유의사항 명시", "문의처 제공"],
            "공지사항": ["변경사항 안내", "적용일시 명시", "영향범위 설명", "대응방법"],
            "일반안내": ["명확한 정보 전달", "친근한 어조", "연락처 제공", "감사인사"]
        }
        
        patterns.extend(type_patterns.get(message_type, []))
        
        # WhiteList 데이터에서 추가 패턴 추출
        if "승인" in whitelist_data:
            patterns.append("승인된 표현 사용")
        if "허용" in whitelist_data:
            patterns.append("허용된 형식 적용")
        if "권장" in whitelist_data:
            patterns.append("권장사항 준수")
            
        return patterns
    
    def _get_default_patterns(self, message_type: str) -> List[str]:
        """기본 승인 패턴"""
        return [
            "정중한 인사말",
            "명확한 정보 전달", 
            "연락처 정보 포함",
            "감사 표현"
        ]
    
    def _generate_optimization_suggestions(self, message_type: str, user_input: str) -> List[str]:
        """최적화 제안사항 생성"""
        suggestions = []
        
        # 메시지 유형별 최적화 제안
        if message_type == "예약확인":
            suggestions.extend([
                "예약번호를 명확히 표기하세요",
                "방문 시 준비사항을 안내하세요",
                "변경/취소 방법을 포함하세요"
            ])
        elif message_type == "결제안내":
            suggestions.extend([
                "결제금액과 상품명을 명시하세요",
                "결제일시를 포함하세요",
                "환불 정책을 안내하세요"
            ])
        elif message_type == "배송안내":
            suggestions.extend([
                "송장번호를 제공하세요",
                "예상 배송일을 안내하세요",
                "부재 시 처리방법을 설명하세요"
            ])
        elif message_type == "이벤트안내":
            suggestions.extend([
                "[광고] 표기를 확인하세요",
                "이벤트 기간을 명시하세요",
                "참여 조건을 명확히 하세요"
            ])
            
        # 공통 최적화 제안
        suggestions.extend([
            "친근하면서도 정중한 어조 사용",
            "핵심 정보를 먼저 배치",
            "문의사항 연락처 포함",
            "수신거부 방법 안내"
        ])
        
        return list(set(suggestions))  # 중복 제거
    
    def _calculate_optimization_score(self, user_input: str, patterns: List[str]) -> int:
        """최적화 점수 계산 (0-100)"""
        score = 50  # 기본 점수
        
        # 길이 적절성 (+10/-10)
        if 50 <= len(user_input) <= 200:
            score += 10
        elif len(user_input) < 20 or len(user_input) > 500:
            score -= 10
            
        # 패턴 매칭 (+5 per pattern, max +30)
        pattern_bonus = min(len(patterns) * 5, 30)
        score += pattern_bonus
        
        # 키워드 포함 여부
        positive_keywords = ["안내", "확인", "감사", "문의"]
        for keyword in positive_keywords:
            if keyword in user_input:
                score += 5
                
        return min(max(score, 0), 100)  # 0-100 범위 보장
    
    def _run(self, user_input: str) -> Dict[str, Any]:
        """WhiteList Tool 실행"""
        try:
            # 1. 메시지 유형 분류
            message_type = self._categorize_message_type(user_input)
            
            # 2. 승인된 패턴 찾기
            approved_patterns = self._find_approved_patterns(message_type, user_input)
            
            # 3. 최적화 제안사항 생성
            optimization_suggestions = self._generate_optimization_suggestions(message_type, user_input)
            
            # 4. 최적화 점수 계산
            optimization_score = self._calculate_optimization_score(user_input, approved_patterns)
            
            # 5. 승인 상태 결정
            approval_status = "APPROVED" if optimization_score >= 70 else "NEEDS_IMPROVEMENT"
            
            whitelist_data = self._get_whitelist_data()
            
            return {
                "tool_name": "WhiteList",
                "user_input": user_input,
                "message_type": message_type,
                "approved_patterns": approved_patterns,
                "optimization_suggestions": optimization_suggestions,
                "optimization_score": optimization_score,
                "approval_status": approval_status,
                "data_loaded": len(whitelist_data) > 0,
                "data_size": len(whitelist_data),
                "recommendation": f"{message_type} 형태로 템플릿 구성 권장 (점수: {optimization_score}/100)"
            }
            
        except Exception as e:
            return {
                "tool_name": "WhiteList", 
                "error": str(e),
                "user_input": user_input,
                "message_type": "UNKNOWN",
                "approved_patterns": [],
                "optimization_suggestions": [],
                "optimization_score": 0,
                "approval_status": "ERROR",
                "data_loaded": False,
                "data_size": 0,
                "recommendation": "도구 실행 중 오류 발생"
            }