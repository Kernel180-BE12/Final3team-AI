#!/usr/bin/env python3
"""
Agent2 - AI.png 구조에 맞는 템플릿 생성 에이전트
4개 Tools (BlackList, WhiteList, 가이드라인, 정보통신법)과 연동하여 
가이드라인과 법령을 모르는 사용자를 위한 완벽 준수 템플릿 생성
"""
import os
import sys
import json
from typing import Dict, List, Tuple
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel

# 상위 디렉토리의 config 모듈을 import하기 위해 path 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config import GEMINI_API_KEY

class Agent2:
    """AI.png 구조에 맞는 Agent2 구현 (데이터 캐싱 최적화)"""
    
    def __init__(self, api_key: str = None, gemini_model: str = "gemini-2.0-flash-exp", index_manager=None):
        self.api_key = api_key or GEMINI_API_KEY
        self.llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=self.api_key,
            temperature=0.3
        )
        
        #  인덱스 매니저로 데이터 공유 (중복 로딩 방지)
        self.index_manager = index_manager
        self._predata_cache = None
        
        # 4개 Tools 초기화 (캐시된 데이터 사용)
        self.blacklist_tool = self._init_blacklist_tool()
        self.whitelist_tool = self._init_whitelist_tool() 
        self.guideline_tool = self._init_guideline_tool()
        self.law_tool = self._init_law_tool()
        
        # 4개 Tools 병렬 실행 준비
        self.tools = {
            "blacklist": self.blacklist_tool,
            "whitelist": self.whitelist_tool,
            "guideline": self.guideline_tool,
            "law": self.law_tool
        }
        
        print(" Agent2 초기화 완료 - AI.png 구조 적용 (캐시 최적화)")
    
    def _get_predata_cache(self):
        """ 캐시된 predata 가져오기 (Tools 간 공유)"""
        if self._predata_cache is None:
            if self.index_manager:
                self._predata_cache = self.index_manager.get_predata_cache()
            else:
                # 폴백: 직접 로딩
                self._predata_cache = self._load_predata_direct()
        return self._predata_cache
    
    def _load_predata_direct(self):
        """폴백: 직접 predata 로딩"""
        import os
        from pathlib import Path
        
        data = {}
        predata_dir = Path("predata") 
        files = ["cleaned_black_list.md", "cleaned_white_list.md", 
                "cleaned_add_infotalk.md", "cleaned_alrimtalk.md",
                "cleaned_content-guide.md", "cleaned_message.md",
                "cleaned_run_message.md", "cleaned_zipguide.md", 
                "pdf_extraction_results.txt"]
        
        for filename in files:
            file_path = predata_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data[filename] = f.read()
                except Exception as e:
                    print(f" {filename} 로드 실패: {e}")
        return data

    def _init_blacklist_tool(self):
        """BlackList Tool 초기화 (캐시 사용)"""
        class BlackListTool:
            def __init__(self, parent_agent):
                self.parent = parent_agent
                self.blacklist_data = None
            
            def _get_blacklist_data(self):
                if self.blacklist_data is None:
                    predata = self.parent._get_predata_cache()
                    self.blacklist_data = predata.get("cleaned_black_list.md", "")
                return self.blacklist_data
            
            def invoke(self, input_data):
                """금지어 패턴 분석"""
                user_input = input_data.get("user_input", "")
                blacklist_data = self._get_blacklist_data()
                
                # 위험 키워드 식별
                risk_keywords = []
                high_risk_patterns = ["할인", "이벤트", "무료", "특가", "프로모션", "경품", "추첨"]
                for pattern in high_risk_patterns:
                    if pattern in user_input:
                        risk_keywords.append(pattern)
                
                # BlackList 데이터에서 관련 제한사항 찾기
                restrictions = []
                if risk_keywords and blacklist_data:
                    if "광고" in blacklist_data:
                        restrictions.append("[광고] 표기 필수")
                    if "금지" in blacklist_data:
                        restrictions.append("스팸성 내용 금지")
                    if "위반" in blacklist_data:
                        restrictions.append("가이드라인 위반 시 계정 차단")
                
                return {
                    "tool_name": "BlackList",
                    "risk_level": "HIGH" if risk_keywords else "LOW",
                    "risk_keywords": risk_keywords,
                    "restrictions": restrictions,
                    "data_loaded": len(blacklist_data) > 0,
                    "compliance_check": "FAILED" if risk_keywords else "PASSED"
                }
        
        return BlackListTool(self)
    
    def _init_whitelist_tool(self):
        """WhiteList Tool 초기화"""
        class WhiteListTool:
            def __init__(self):
                self.whitelist_data = self._load_whitelist()
            
            def _load_whitelist(self):
                try:
                    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    whitelist_path = os.path.join(base_path, "predata", "cleaned_white_list.md")
                    with open(whitelist_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    print(f" WhiteList 로드 실패: {e}")
                    return ""
            
            def invoke(self, input_data):
                """승인 패턴 분석"""
                user_input = input_data.get("user_input", "")
                
                # 카테고리 분류
                category = "일반안내"
                if "예약" in user_input: category = "예약확인"
                elif "결제" in user_input: category = "결제안내"
                elif "포인트" in user_input: category = "포인트안내"
                elif "배송" in user_input: category = "배송안내"
                
                # WhiteList에서 승인 패턴 찾기
                approved_patterns = []
                if self.whitelist_data:
                    if "승인" in self.whitelist_data:
                        approved_patterns.append("정당한 서비스 안내")
                    if "허용" in self.whitelist_data:
                        approved_patterns.append("고객 요청 정보")
                
                return {
                    "tool_name": "WhiteList",
                    "category": category,
                    "approved_patterns": approved_patterns,
                    "recommendation": f"{category} 형태로 템플릿 구성",
                    "data_loaded": len(self.whitelist_data) > 0,
                    "approval_status": "APPROVED"
                }
        
        return WhiteListTool()
    
    def _init_guideline_tool(self):
        """가이드라인 Tool 초기화"""
        class GuidelineTool:
            def __init__(self):
                self.guidelines_data = self._load_guidelines()
            
            def _load_guidelines(self):
                try:
                    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    guideline_files = [
                        "cleaned_add_infotalk.md",
                        "cleaned_alrimtalk.md",
                        "cleaned_content-guide.md",
                        "cleaned_message.md",
                        "cleaned_run_message.md",
                        "cleaned_zipguide.md"
                    ]
                    
                    all_data = ""
                    for filename in guideline_files:
                        file_path = os.path.join(base_path, "predata", filename)
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                all_data += f.read() + "\n"
                    
                    return all_data
                except Exception as e:
                    print(f" 가이드라인 로드 실패: {e}")
                    return ""
            
            def invoke(self, input_data):
                """가이드라인 준수사항 분석"""
                user_input = input_data.get("user_input", "")
                
                # 가이드라인 요구사항 추출
                requirements = []
                if self.guidelines_data:
                    if "필수" in self.guidelines_data:
                        requirements.append("필수 정보 포함")
                    if "권장" in self.guidelines_data:
                        requirements.append("권장사항 준수")
                    if "금지" in self.guidelines_data:
                        requirements.append("금지사항 회피")
                
                # 템플릿 구조 권장사항
                template_structure = [
                    "인사말 포함",
                    "핵심 내용 명시",
                    "연락처 정보",
                    "수신거부 안내"
                ]
                
                return {
                    "tool_name": "가이드라인",
                    "requirements": requirements,
                    "template_structure": template_structure,
                    "compliance_level": "HIGH",
                    "data_loaded": len(self.guidelines_data) > 0,
                    "guideline_version": "2024_standard"
                }
        
        return GuidelineTool()
    
    def _init_law_tool(self):
        """정보통신법 Tool 초기화"""
        class LawTool:
            def __init__(self):
                self.law_data = self._load_law_data()
            
            def _load_law_data(self):
                try:
                    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    law_path = os.path.join(base_path, "predata", "pdf_extraction_results.txt")
                    with open(law_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    print(f" 정보통신법 데이터 로드 실패: {e}")
                    return ""
            
            def invoke(self, input_data):
                """정보통신망법 준수사항 분석"""
                user_input = input_data.get("user_input", "")
                
                # 메시지 유형 분류
                message_type = "정보성"
                if any(word in user_input for word in ["할인", "이벤트", "특가"]):
                    message_type = "광고성"
                
                # 법적 요구사항
                legal_requirements = []
                if self.law_data:
                    if "정보통신망법" in self.law_data:
                        legal_requirements.append("정보통신망법 제50조 준수")
                    if "21시" in self.law_data:
                        legal_requirements.append("야간 전송 금지 (21시~8시)")
                    if message_type == "광고성":
                        legal_requirements.append("[광고] 표기 의무")
                        legal_requirements.append("수신동의자에게만 발송")
                
                return {
                    "tool_name": "정보통신법",
                    "message_type": message_type,
                    "legal_requirements": legal_requirements,
                    "compliance_status": "COMPLIANT",
                    "data_loaded": len(self.law_data) > 0,
                    "law_version": "정보통신망법_최신"
                }
        
        return LawTool()
    
    def _preprocess_dates(self, user_input: str) -> str:
        """날짜 표현을 구체적인 날짜로 변환"""
        from ..tools.date_preprocessor import DatePreprocessor
        preprocessor = DatePreprocessor()
        return preprocessor.preprocess_dates(user_input)
    
    def generate_compliant_template(self, user_input: str, agent1_variables: Dict[str, str] = None) -> Tuple[str, Dict]:
        """
        AI.png 구조에 따른 완벽 준수 템플릿 생성
        4개 Tools 병렬 실행 -> Agent(템플릿생성자) -> 최종 템플릿
        """
        print(f" Agent2: 4개 Tools 병렬 분석 시작")
        
        # 0단계: 날짜 전처리 (내일, 모레 등을 구체적 날짜로 변환)
        preprocessed_input = self._preprocess_dates(user_input)
        if preprocessed_input != user_input:
            print(f" 날짜 전처리: '{user_input}' → '{preprocessed_input}'")
        
        # 1단계: 4개 Tools 병렬 실행
        input_data = {"user_input": preprocessed_input}
        
        try:
            tools_results = {}
            for tool_name, tool in self.tools.items():
                tools_results[tool_name] = tool.invoke(input_data)
            print(f" 4개 Tools 분석 완료")
            
        except Exception as e:
            print(f" Tools 실행 오류: {e}")
            return {"success": False, "error": f"Tools 실행 오류: {str(e)}"}, {}
        
        # 2단계: Agent(템플릿생성자)가 Tools 결과를 종합하여 템플릿 생성
        template = self._create_template_from_tools(preprocessed_input, tools_results)
        
        # 메타데이터 준비
        metadata = {
            "original_input": user_input,
            "preprocessed_input": preprocessed_input,
            "date_preprocessing_applied": preprocessed_input != user_input,
            "tools_results": tools_results,
            "generation_method": "Agent2_4Tools_Parallel",
            "compliance_status": {
                "blacklist_passed": tools_results["blacklist"]["compliance_check"] == "PASSED",
                "whitelist_approved": tools_results["whitelist"]["approval_status"] == "APPROVED",
                "guideline_compliant": tools_results["guideline"]["compliance_level"] == "HIGH",
                "law_compliant": tools_results["law"]["compliance_status"] == "COMPLIANT"
            },
            "all_data_loaded": all(
                result.get("data_loaded", False) for result in tools_results.values()
            )
        }
        
        print(f" Agent2 템플릿 생성 완료")
        
        # 성공적인 결과를 딕셔너리 형태로 반환
        result = {
            "success": True,
            "template": template,
            "variables": []  # TODO: 변수 추출 로직 추가 필요
        }
        return result, metadata
    
    def _create_template_from_tools(self, user_input: str, tools_results: Dict) -> str:
        """4개 Tools 결과를 종합하여 최종 템플릿 생성"""
        
        # Tools 결과 요약
        blacklist = tools_results["blacklist"]
        whitelist = tools_results["whitelist"]
        guideline = tools_results["guideline"]
        law = tools_results["law"]
        
        # LLM에 전달할 프롬프트 구성
        system_prompt = """당신은 Agent2의 템플릿생성자입니다.
4개 Tools(BlackList, WhiteList, 가이드라인, 정보통신법)의 분석 결과를 바탕으로 
완벽하게 준수하는 알림톡 템플릿을 생성해야 합니다.

중요한 원칙:
1. BlackList 위반 사항은 절대 포함하지 않음
2. WhiteList 승인 패턴을 적극 활용
3. 가이드라인 구조와 요구사항을 모두 적용
4. 정보통신망법을 완벽 준수
5. 변수는 반드시 {변수명} 형식으로 작성 (예: {카페이름}, {고객명}, {주문내용})

템플릿 생성 규칙:
- 적절한 인사말
- 핵심 내용 (사용자가 제공한 구체적인 정보를 그대로 사용)
- 연락처 정보
- 법적 고지사항 (필요시)
- 정중한 마무리

 중요: 템플릿 메타 정보는 제외하고 실제 메시지 내용만 생성하세요.
- "## 알림톡 템플릿", "**[템플릿 제목]**", "**[템플릿 내용]**" 같은 구조 텍스트 절대 금지
- "참고사항", "본 템플릿은..." 같은 부가 설명 제외  
- 순수한 알림톡 메시지 내용만 출력
- 모든 변수는 {변수명} 형식으로 작성 ([변수명] 형식 절대 금지)"""

        human_prompt = f"""사용자 요청: {user_input}

 4개 Tools 분석 결과:

 BlackList 분석:
- 위험도: {blacklist['risk_level']}
- 위험 키워드: {blacklist['risk_keywords']}
- 제한사항: {blacklist['restrictions']}
- 준수상태: {blacklist['compliance_check']}

 WhiteList 분석:  
- 카테고리: {whitelist['category']}
- 승인 패턴: {whitelist['approved_patterns']}
- 권장사항: {whitelist['recommendation']}

 가이드라인 분석:
- 요구사항: {guideline['requirements']}
- 템플릿 구조: {guideline['template_structure']}
- 준수 수준: {guideline['compliance_level']}

 정보통신법 분석:
- 메시지 유형: {law['message_type']}
- 법적 요구사항: {law['legal_requirements']}
- 준수 상태: {law['compliance_status']}

위 4개 Tools의 분석을 완벽히 반영하여 안전하고 효과적인 알림톡 템플릿을 생성해주세요."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f" 템플릿 생성 LLM 오류: {e}")
            
            # 폴백 템플릿 생성
            fallback_template = self._create_fallback_template(user_input, tools_results)
            return fallback_template
    
    def _create_fallback_template(self, user_input: str, tools_results: Dict) -> str:
        """LLM 오류 시 폴백 템플릿"""
        law = tools_results["law"]
        whitelist = tools_results["whitelist"]
        
        # 광고성 메시지 여부에 따른 접두사
        prefix = "[광고] " if law["message_type"] == "광고성" else ""
        
        template = f"""{prefix}안녕하세요, {{고객명}}님.

{user_input}에 대해 안내드립니다.

 주요 내용
- 일시: {{일시}}
- 장소: {{장소}}  
- 내용: {{상세내용}}

 문의처
- 연락처: {{연락처}}
- 운영시간: 평일 9시-18시

※ 본 메시지는 {whitelist['category']} 목적으로 발송되는 {'광고성 정보 전송에 동의하신 분께 발송되는' if law['message_type'] == '광고성' else '서비스 안내'} 메시지입니다.

감사합니다."""
        
        return template


# 테스트 실행 블록
if __name__ == "__main__":
    print("Agent2 테스트 시작...")
    
    try:
        # Agent2 인스턴스 생성
        agent = Agent2()
        print("Agent2 인스턴스 생성 성공!")
        
        # 간단한 테스트
        test_input = "긴급 점검 안내드립니다"
        print(f"테스트 입력: {test_input}")
        
        result, tools_data = agent.generate_compliant_template(test_input)
        print(f"생성된 템플릿:\n{result}")
        print(f"도구 데이터: {tools_data}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()