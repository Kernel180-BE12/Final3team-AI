"""
Agent1용 날짜 전처리기
사용자 입력의 자연어 날짜 표현을 구체적인 날짜로 변환
"""

import re
from datetime import date, timedelta


class DatePreprocessor:
    """Agent1용 날짜 전처리 클래스"""
    
    def __init__(self):
        self.date_expressions = {
            "오늘": 0,
            "내일": 1,
            "모레": 2,
            "글피": 3,
            "사흘 뒤": 3,
            "나흘 뒤": 4,
            "닷새 뒤": 5,
            "엿새 뒤": 6,
            "이레 뒤": 7,
            "여드레 뒤": 8,
            "아흐레 뒤": 9,
            "열흘 뒤": 10,
        }
    
    def get_weekday_korean(self, weekday: int) -> str:
        """요일을 한국어로 변환"""
        return ["월", "화", "수", "목", "금", "토", "일"][weekday]
    
    def preprocess_dates(self, query: str) -> str:
        """
        자연어 날짜 표현을 구체적인 날짜로 변환
        
        Args:
            query: 사용자 입력 텍스트
            
        Returns:
            날짜가 변환된 텍스트
        """
        today = date.today()
        processed_query = query
        
        # 기본 날짜 표현 처리
        for expr, days in self.date_expressions.items():
            if expr in processed_query:
                target_date = today + timedelta(days=days)
                replacement = f"{target_date.strftime('%Y년 %m월 %d일')} ({self.get_weekday_korean(target_date.weekday())}요일)"
                processed_query = processed_query.replace(expr, replacement)
        
        # 'N일 뒤' 패턴 처리
        match = re.search(r'(\d+)\s*일\s*뒤', processed_query)
        if match:
            days = int(match.group(1))
            target_date = today + timedelta(days=days)
            replacement = f"{target_date.strftime('%Y년 %m월 %d일')} ({self.get_weekday_korean(target_date.weekday())}요일)"
            processed_query = re.sub(r'(\d+)\s*일\s*뒤', replacement, processed_query)
        
        return processed_query
    
    def extract_date_info(self, query: str) -> dict:
        """
        입력에서 날짜 관련 정보 추출
        
        Args:
            query: 사용자 입력 텍스트
            
        Returns:
            날짜 정보 딕셔너리
        """
        today = date.today()
        date_info = {
            "original_expressions": [],
            "converted_dates": [],
            "days_from_today": []
        }
        
        # 기본 날짜 표현 찾기
        for expr, days in self.date_expressions.items():
            if expr in query:
                target_date = today + timedelta(days=days)
                date_info["original_expressions"].append(expr)
                date_info["converted_dates"].append(target_date.strftime('%Y년 %m월 %d일'))
                date_info["days_from_today"].append(days)
        
        # 'N일 뒤' 패턴 찾기
        match = re.search(r'(\d+)\s*일\s*뒤', query)
        if match:
            days = int(match.group(1))
            target_date = today + timedelta(days=days)
            date_info["original_expressions"].append(match.group(0))
            date_info["converted_dates"].append(target_date.strftime('%Y년 %m월 %d일'))
            date_info["days_from_today"].append(days)
        
        return date_info


def preprocess_dates(query: str) -> str:
    """
    편의 함수: 날짜 전처리 실행
    
    Args:
        query: 사용자 입력 텍스트
        
    Returns:
        날짜가 변환된 텍스트
    """
    preprocessor = DatePreprocessor()
    return preprocessor.preprocess_dates(query)


def get_weekday_korean(weekday: int) -> str:
    """
    편의 함수: 요일을 한국어로 변환
    
    Args:
        weekday: 요일 번호 (0=월요일, 6=일요일)
        
    Returns:
        한국어 요일
    """
    return ["월", "화", "수", "목", "금", "토", "일"][weekday]


if __name__ == "__main__":
    # 테스트
    preprocessor = DatePreprocessor()
    
    test_queries = [
        "내일 회의가 있습니다",
        "모레까지 과제를 제출해주세요", 
        "3일 뒤에 행사가 열립니다",
        "오늘은 좋은 날이네요",
        "글피 저녁에 만나요"
    ]
    
    print("=== 날짜 전처리 테스트 ===")
    for query in test_queries:
        processed = preprocessor.preprocess_dates(query)
        date_info = preprocessor.extract_date_info(query)
        print(f"원본: {query}")
        print(f"변환: {processed}")
        print(f"추출정보: {date_info}")
        print("-" * 50)