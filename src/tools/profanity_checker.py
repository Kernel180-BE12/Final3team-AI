#!/usr/bin/env python3
"""
비속어 검증 도구 (Profanity Checker Tool)

Agent1에서 분리된 독립적인 비속어 검증 도구
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent.parent


class ProfanityChecker:
    """비속어 검증 도구"""

    def __init__(self, keyword_file_path: str = None):
        """
        비속어 체커 초기화

        Args:
            keyword_file_path: 비속어 키워드 파일 경로 (기본값: predata/cleaned_blacklist_keyword.txt)
        """
        self.keyword_file_path = keyword_file_path or str(project_root / "predata" / "cleaned_blacklist_keyword.txt")
        self.profanity_keywords: Set[str] = set()
        self._load_keywords()

    def _load_keywords(self) -> None:
        """비속어 키워드 파일 로드"""
        try:
            if os.path.exists(self.keyword_file_path):
                with open(self.keyword_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        keyword = line.strip()
                        if keyword:
                            self.profanity_keywords.add(keyword.lower())
                print(f"비속어 키워드 {len(self.profanity_keywords)}개 로드 완료")
            else:
                print(f"비속어 키워드 파일을 찾을 수 없습니다: {self.keyword_file_path}")
        except Exception as e:
            print(f"비속어 키워드 로드 실패: {e}")

    def check_text(self, text: str) -> Dict[str, Any]:
        """
        텍스트 비속어 검사

        Args:
            text: 검사할 텍스트

        Returns:
            검사 결과 딕셔너리
        """
        if not text or not text.strip():
            return {
                'is_clean': True,
                'detected_words': [],
                'message': '검사할 텍스트가 없습니다.'
            }

        if not self.profanity_keywords:
            return {
                'is_clean': True,
                'detected_words': [],
                'message': '비속어 키워드가 로드되지 않았습니다.'
            }

        detected_words = []
        text_lower = text.lower()
        text_no_space = text_lower.replace(" ", "")

        # 키워드 검사
        for keyword in self.profanity_keywords:
            if keyword in text_lower or keyword in text_no_space:
                detected_words.append(keyword)

        is_clean = len(detected_words) == 0

        return {
            'is_clean': is_clean,
            'detected_words': detected_words,
            'message': '정상 텍스트입니다.' if is_clean else f'비속어 검출: {", ".join(detected_words)}'
        }

    def check_multiple_texts(self, texts: List[str]) -> Dict[str, Any]:
        """
        여러 텍스트 일괄 검사

        Args:
            texts: 검사할 텍스트 리스트

        Returns:
            일괄 검사 결과
        """
        results = []
        all_detected = []

        for i, text in enumerate(texts):
            result = self.check_text(text)
            results.append({
                'index': i,
                'text': text[:50] + '...' if len(text) > 50 else text,
                'is_clean': result['is_clean'],
                'detected_words': result['detected_words']
            })
            all_detected.extend(result['detected_words'])

        # 중복 제거
        unique_detected = list(set(all_detected))

        return {
            'all_clean': len(unique_detected) == 0,
            'total_texts': len(texts),
            'clean_count': sum(1 for r in results if r['is_clean']),
            'detected_count': len(texts) - sum(1 for r in results if r['is_clean']),
            'all_detected_words': unique_detected,
            'detailed_results': results,
            'summary': f'{len(texts)}개 텍스트 중 {sum(1 for r in results if r["is_clean"])}개 정상'
        }

    def add_keyword(self, keyword: str) -> bool:
        """
        비속어 키워드 추가 (런타임)

        Args:
            keyword: 추가할 키워드

        Returns:
            추가 성공 여부
        """
        if keyword and keyword.strip():
            self.profanity_keywords.add(keyword.lower().strip())
            return True
        return False

    def remove_keyword(self, keyword: str) -> bool:
        """
        비속어 키워드 제거 (런타임)

        Args:
            keyword: 제거할 키워드

        Returns:
            제거 성공 여부
        """
        keyword_lower = keyword.lower().strip()
        if keyword_lower in self.profanity_keywords:
            self.profanity_keywords.remove(keyword_lower)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        체커 통계 정보 반환

        Returns:
            통계 정보
        """
        return {
            'total_keywords': len(self.profanity_keywords),
            'keyword_file_path': self.keyword_file_path,
            'file_exists': os.path.exists(self.keyword_file_path),
            'sample_keywords': list(self.profanity_keywords)[:10] if self.profanity_keywords else []
        }


# 싱글톤 인스턴스
_profanity_checker_instance: Optional[ProfanityChecker] = None


def get_profanity_checker() -> ProfanityChecker:
    """전역 비속어 체커 인스턴스 반환"""
    global _profanity_checker_instance
    if _profanity_checker_instance is None:
        _profanity_checker_instance = ProfanityChecker()
    return _profanity_checker_instance


def check_profanity(text: str) -> Dict[str, Any]:
    """
    편의 함수: 텍스트 비속어 검사

    Args:
        text: 검사할 텍스트

    Returns:
        검사 결과
    """
    checker = get_profanity_checker()
    return checker.check_text(text)


def is_text_clean(text: str) -> bool:
    """
    편의 함수: 텍스트가 깨끗한지 단순 확인

    Args:
        text: 검사할 텍스트

    Returns:
        True: 깨끗함, False: 비속어 포함
    """
    result = check_profanity(text)
    return result['is_clean']


if __name__ == "__main__":
    # 테스트
    print("=== 비속어 검증 도구 테스트 ===")

    checker = ProfanityChecker()

    # 통계 출력
    stats = checker.get_stats()
    print(f"로드된 키워드 수: {stats['total_keywords']}")
    print(f"파일 경로: {stats['keyword_file_path']}")
    print(f"파일 존재: {stats['file_exists']}")

    # 테스트 텍스트들
    test_texts = [
        "안녕하세요 좋은 하루 되세요",
        "부트캠프 안내 메시지입니다",
        "템플릿 생성을 요청합니다"
    ]

    print("\n=== 개별 검사 테스트 ===")
    for text in test_texts:
        result = checker.check_text(text)
        print(f"텍스트: '{text}'")
        print(f"결과: {result['message']}")
        print()

    print("=== 일괄 검사 테스트 ===")
    batch_result = checker.check_multiple_texts(test_texts)
    print(f"요약: {batch_result['summary']}")
    print(f"전체 깨끗함: {batch_result['all_clean']}")