#!/usr/bin/env python3
"""
InfoCommLaw Tool - 정보통신망법 준수 검증 도구
Agent2의 세 번째 도구
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from langchain.tools import BaseTool
from datetime import datetime

class InfoCommLawTool(BaseTool):
    """
    정보통신망 이용촉진 및 정보보호 등에 관한 법률 준수를 검증하는 도구
    """
    name: str = "info_communication_law_compliance"
    description: str = "정보통신망법 준수 검증 및 가이드"
    
    def __init__(self, index_manager=None):
        super().__init__()
        self.index_manager = index_manager
        self._law_data = None
        
    def _get_law_data(self) -> str:
        """법령 데이터 로드 (캐시 사용)"""
        if self._law_data is None:
            try:
                if self.index_manager:
                    # IndexManager에서 캐시된 데이터 사용
                    predata = self.index_manager.get_predata_cache()
                    self._law_data = predata.get("pdf_extraction_results.txt", "")
                else:
                    # 직접 파일 로드
                    base_path = Path(__file__).parent.parent.parent
                    law_path = base_path / "predata" / "pdf_extraction_results.txt"
                    
                    if law_path.exists():
                        with open(law_path, 'r', encoding='utf-8') as f:
                            self._law_data = f.read()
                    else:
                        self._law_data = ""
            except Exception as e:
                print(f" 법령 데이터 로드 실패: {e}")
                self._law_data = ""
        
        return self._law_data
    
    def _classify_message_type(self, user_input: str) -> Tuple[str, str]:
        """메시지 유형 분류 (정보성 vs 광고성)"""
        user_input_lower = user_input.lower()
        
        # 광고성 키워드
        advertising_keywords = [
            "할인", "이벤트", "특가", "세일", "프로모션",
            "무료", "쿠폰", "혜택", "경품", "추첨", "당첨"
        ]
        
        # 정보성 키워드
        informational_keywords = [
            "예약", "확인", "안내", "알림", "공지",
            "결제", "배송", "접수", "승인", "완료"
        ]
        
        advertising_score = sum(1 for keyword in advertising_keywords if keyword in user_input_lower)
        informational_score = sum(1 for keyword in informational_keywords if keyword in user_input_lower)
        
        if advertising_score > informational_score:
            return "광고성정보", "상업적 목적의 광고성 메시지"
        elif informational_score > 0:
            return "정보성", "서비스 이용 관련 정보 제공 메시지"
        else:
            return "정보성", "일반적인 정보 전달 메시지"
    
    def _get_legal_requirements(self, message_type: str, user_input: str) -> List[str]:
        """법적 요구사항 추출"""
        requirements = []
        law_data = self._get_law_data()
        
        # 광고성 정보 전송 시 요구사항
        if message_type == "광고성정보":
            requirements.extend([
                "[광고] 표기 의무 (정보통신망법 제50조)",
                "수신동의자에게만 발송",
                "야간 전송 금지 (21시~8시)",
                "송신자 정보 명시",
                "수신거부 방법 제공"
            ])
            
        # 정보성 메시지 요구사항
        else:
            requirements.extend([
                "명확한 정보 전달",
                "송신자 신원 명시",
                "수신거부 방법 제공"
            ])
        
        # 법령 데이터에서 추가 요구사항 추출
        if law_data:
            if "개인정보" in user_input:
                if "개인정보" in law_data:
                    requirements.append("개인정보처리방침 고지")
                if "동의" in law_data:
                    requirements.append("개인정보 수집 시 동의 필수")
                    
            if "위치" in user_input or "주소" in user_input:
                if "위치정보" in law_data:
                    requirements.append("위치정보 이용 동의 필요")
                    
            if "결제" in user_input or "구매" in user_input:
                if "전자상거래" in law_data:
                    requirements.append("전자상거래법 준수")
        
        return list(set(requirements))  # 중복 제거
    
    def _check_time_restrictions(self, message_type: str) -> Dict[str, Any]:
        """시간 제한 검증"""
        current_hour = datetime.now().hour
        
        # 야간 시간대 (21시~8시)
        is_night_time = current_hour >= 21 or current_hour < 8
        
        if message_type == "광고성정보" and is_night_time:
            return {
                "time_compliant": False,
                "current_time": f"{current_hour}시",
                "restriction": "광고성 정보는 21시~8시 전송 금지",
                "allowed_time": "8시~21시"
            }
        else:
            return {
                "time_compliant": True,
                "current_time": f"{current_hour}시",
                "restriction": "시간 제한 없음" if message_type == "정보성" else "8시~21시 전송 권장",
                "allowed_time": "24시간" if message_type == "정보성" else "8시~21시"
            }
    
    def _generate_compliance_checklist(self, message_type: str, user_input: str) -> List[Dict[str, Any]]:
        """준수사항 체크리스트 생성"""
        checklist = []
        
        if message_type == "광고성정보":
            checklist.extend([
                {
                    "item": "[광고] 표기",
                    "required": True,
                    "description": "메시지 시작 부분에 [광고] 표기 필수",
                    "penalty": "과태료 300만원 이하"
                },
                {
                    "item": "수신동의 확인", 
                    "required": True,
                    "description": "사전 수신동의를 받은 대상에게만 발송",
                    "penalty": "과태료 300만원 이하"
                },
                {
                    "item": "야간 전송 금지",
                    "required": True,
                    "description": "21시~8시 사이 전송 금지",
                    "penalty": "과태료 300만원 이하"
                },
                {
                    "item": "수신거부 방법",
                    "required": True,
                    "description": "수신거부 방법을 메시지에 명시",
                    "penalty": "과태료 300만원 이하"
                }
            ])
        
        # 공통 준수사항
        checklist.extend([
            {
                "item": "송신자 정보",
                "required": True,
                "description": "발송자의 연락처 정보 포함",
                "penalty": "과태료 100만원 이하"
            },
            {
                "item": "정확한 정보",
                "required": True, 
                "description": "허위 또는 과장된 정보 금지",
                "penalty": "형사처벌 가능"
            }
        ])
        
        return checklist
    
    def _calculate_compliance_score(self, message_type: str, user_input: str, checklist: List[Dict]) -> int:
        """준수도 점수 계산 (0-100)"""
        score = 100
        
        # 광고성 메시지 검증
        if message_type == "광고성정보":
            if "[광고]" not in user_input:
                score -= 30
            if "수신거부" not in user_input and "거부" not in user_input:
                score -= 20
                
        # 기본 요소 검증
        if "연락처" not in user_input and "문의" not in user_input:
            score -= 15
            
        # 시간 제한 검증
        time_check = self._check_time_restrictions(message_type)
        if not time_check["time_compliant"]:
            score -= 25
            
        return max(score, 0)
    
    def _run(self, user_input: str) -> Dict[str, Any]:
        """InfoCommLaw Tool 실행"""
        try:
            # 1. 메시지 유형 분류
            message_type, type_description = self._classify_message_type(user_input)
            
            # 2. 법적 요구사항 추출
            legal_requirements = self._get_legal_requirements(message_type, user_input)
            
            # 3. 시간 제한 검증
            time_restrictions = self._check_time_restrictions(message_type)
            
            # 4. 준수사항 체크리스트 생성
            compliance_checklist = self._generate_compliance_checklist(message_type, user_input)
            
            # 5. 준수도 점수 계산
            compliance_score = self._calculate_compliance_score(message_type, user_input, compliance_checklist)
            
            # 6. 전체 준수 상태 결정
            overall_compliance = "COMPLIANT" if compliance_score >= 80 else ("NEEDS_REVIEW" if compliance_score >= 60 else "NON_COMPLIANT")
            
            law_data = self._get_law_data()
            
            return {
                "tool_name": "정보통신법",
                "user_input": user_input,
                "message_type": message_type,
                "type_description": type_description,
                "legal_requirements": legal_requirements,
                "time_restrictions": time_restrictions,
                "compliance_checklist": compliance_checklist,
                "compliance_score": compliance_score,
                "overall_compliance": overall_compliance,
                "data_loaded": len(law_data) > 0,
                "data_size": len(law_data),
                "law_version": "정보통신망 이용촉진 및 정보보호 등에 관한 법률 (2024년 기준)",
                "recommendation": self._get_compliance_recommendation(message_type, compliance_score)
            }
            
        except Exception as e:
            return {
                "tool_name": "정보통신법",
                "error": str(e),
                "user_input": user_input,
                "message_type": "UNKNOWN",
                "legal_requirements": [],
                "compliance_score": 0,
                "overall_compliance": "ERROR",
                "data_loaded": False,
                "data_size": 0,
                "recommendation": "법령 검증 중 오류 발생"
            }
    
    def _get_compliance_recommendation(self, message_type: str, score: int) -> str:
        """준수도에 따른 권장사항"""
        if score >= 80:
            return f"{message_type} 메시지로 법령 준수 양호"
        elif score >= 60:
            if message_type == "광고성정보":
                return "광고성 메시지 - [광고] 표기 및 수신동의 확인 필요"
            else:
                return "정보성 메시지 - 일부 개선사항 있음"
        else:
            if message_type == "광고성정보":
                return "광고성 메시지 - 중대한 법령 위반 위험, 즉시 수정 필요"
            else:
                return "정보성 메시지 - 법령 준수 개선 필요"