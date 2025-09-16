#!/usr/bin/env python3
"""
Industry/Purpose 자동 분류기
konlpy 기반 형태소 분석과 키워드 매칭을 통한 효율적인 분류 시스템
"""
import re
from typing import Dict, List, Tuple, Optional

class IndustryClassifier:
    """업종 및 목적 자동 분류기"""

    def __init__(self):
        """분류기 초기화"""
        self.okt = None
        self._init_morphological_analyzer()
        self._init_keywords()

        # 캐시 시스템
        self._cache = {}

    def _init_morphological_analyzer(self):
        """형태소 분석기 초기화 (지연 로딩)"""
        try:
            from konlpy.tag import Okt
            self.okt = Okt()
            print("✅ KoNLPy 형태소 분석기 초기화 완료")
        except ImportError:
            print("⚠️ KoNLPy가 설치되지 않았습니다. pip install konlpy 실행 필요")
            self.okt = None
        except Exception as e:
            print(f"⚠️ 형태소 분석기 초기화 실패: {e}")
            self.okt = None

    def _init_keywords(self):
        """키워드 매핑 초기화"""
        # Industry 키워드 매핑 (ID: 키워드 리스트)
        self.industry_keywords = {
            1: ['학원', '수업', '강의', '교육', '학습', '수강', '교실', '강사', '선생님'],
            2: ['온라인', '인터넷', '화상', '웹', '원격', '비대면', '실시간', '스트리밍'],
            3: ['운동', '헬스', '요가', '필라테스', '체육관', 'PT', '트레이닝', '피트니스', '다이어트'],
            4: ['공연', '콘서트', '전시', '페스티벌', '이벤트', '행사', '축제', '무대'],
            5: ['모임', '만남', '회의', '미팅', '정모', '클럽', '동호회', '커뮤니티'],
            6: ['동창회', '동문회', '동기', '선후배', '졸업생', '동창', '모교', '동문'],
            7: ['병원', '진료', '치료', '의원', '검진', '예약', '진찰', '처방', '수술'],
            8: ['부동산', '매매', '임대', '분양', '아파트', '오피스텔', '빌라', '상가'],
            9: ['기타', '일반', '공통', '기본', '표준']
        }

        # Purpose 키워드 매핑 (ID: 키워드 리스트)
        self.purpose_keywords = {
            1: ['쿠폰', '할인', '혜택', '특가', '세일', '프로모션', '적립', '포인트'],
            2: ['공지', '안내', '알림', '알려', '공고', '통지', '전달', '공표'],
            3: ['뉴스레터', '소식', '뉴스', '정보', '업데이트', '발행', '매거진'],
            4: ['설문', '조사', '의견', '피드백', '평가', '후기', '리뷰', '응답'],
            5: ['리포트', '보고서', '결과', '분석', '통계', '데이터', '자료'],
            6: ['결제', '납부', '청구', '요금', '수납', '계산', '비용', '대금'],
            7: ['콘텐츠', '자료', '정보', '소개', '가이드', '매뉴얼', '설명'],
            8: ['예약', '접수', '신청', '예약확인', '부킹', '스케줄', '일정'],
            9: ['회원', '가입', '등록', '멤버십', '회원관리', '사용자', '계정'],
            10: ['행사신청', '참가', '등록', '접수', '참여', '신청서', '지원'],
            11: ['기타', '일반', '공통', '기본', '표준', '평상시']
        }

        # Agent1 의도 → Purpose 매핑 (문맥 기반)
        self.intent_purpose_mapping = {
            '예약알림/리마인드': {
                'primary': 8,      # 예약
                'secondary': [2],  # 공지/안내
                'confidence': 0.85
            },
            '할인/프로모션': {
                'primary': 1,      # 쿠폰
                'secondary': [2],  # 공지/안내
                'confidence': 0.90
            },
            '서비스안내': {
                'primary': 2,      # 공지/안내
                'secondary': [7],  # 콘텐츠 발송
                'confidence': 0.80
            },
            '회원가입': {
                'primary': 9,      # 회원 관리
                'secondary': [2],  # 공지/안내
                'confidence': 0.90
            },
            '결제안내': {
                'primary': 6,      # 결제 안내
                'secondary': [2],  # 공지/안내
                'confidence': 0.95
            },
            '이벤트안내': {
                'primary': 10,     # 행사 신청
                'secondary': [2],  # 공지/안내
                'confidence': 0.85
            },
            '배송안내': {
                'primary': 2,      # 공지/안내
                'secondary': [7],  # 콘텐츠 발송
                'confidence': 0.80
            },
            '고객만족도조사': {
                'primary': 4,      # 설문
                'secondary': [5],  # 리포트
                'confidence': 0.90
            }
        }

        # 엔티티 패턴 → Industry 매핑 (변수 조합 기반)
        self.entity_pattern_mapping = {
            # 동창회/모임 관련
            ('동창회', '모임'): {'industry': 6, 'confidence': 0.95},
            ('동창회', '만남'): {'industry': 6, 'confidence': 0.90},
            ('동문회', '모임'): {'industry': 6, 'confidence': 0.95},

            # 피트니스 관련 (더 유연한 패턴)
            ('헬스장', '예약'): {'industry': 3, 'confidence': 0.90},  # 헬스장 + 예약
            ('헬스장', '회원'): {'industry': 3, 'confidence': 0.85},  # 헬스장 + 회원
            ('헬스', '운동'): {'industry': 3, 'confidence': 0.90},
            ('요가', '수업'): {'industry': 3, 'confidence': 0.85},
            ('피트니스', '회원'): {'industry': 3, 'confidence': 0.90},

            # 병원 관련
            ('병원', '진료'): {'industry': 7, 'confidence': 0.95},
            ('병원', '예약'): {'industry': 7, 'confidence': 0.90},
            ('의원', '진료'): {'industry': 7, 'confidence': 0.95},
            ('치과', '예약'): {'industry': 7, 'confidence': 0.90},

            # 교육 관련
            ('학원', '수업'): {'industry': 1, 'confidence': 0.95},
            ('학원', '강의'): {'industry': 1, 'confidence': 0.90},
            ('교육', '수강'): {'industry': 1, 'confidence': 0.85},

            # 온라인 강의 관련
            ('온라인', '강의'): {'industry': 2, 'confidence': 0.90},
            ('온라인', '수업'): {'industry': 2, 'confidence': 0.85},
            ('화상', '교육'): {'industry': 2, 'confidence': 0.80},

            # 부동산 관련
            ('부동산', '상담'): {'industry': 8, 'confidence': 0.90},
            ('아파트', '매매'): {'industry': 8, 'confidence': 0.95},
            ('오피스텔', '임대'): {'industry': 8, 'confidence': 0.90},

            # 공연/행사 관련
            ('공연', '예약'): {'industry': 4, 'confidence': 0.90},
            ('콘서트', '티켓'): {'industry': 4, 'confidence': 0.95},
            ('전시', '관람'): {'industry': 4, 'confidence': 0.85},

            # 모임 관련 (일반)
            ('모임', '참가'): {'industry': 5, 'confidence': 0.80},
            ('회의', '참석'): {'industry': 5, 'confidence': 0.75},
            ('미팅', '일정'): {'industry': 5, 'confidence': 0.70}
        }

        # 특별 키워드 우선순위 (원본 텍스트에서 확인)
        self.special_keyword_mapping = {
            'PT': {'industry': 3, 'confidence': 0.85, 'name': '피트니스'},  # PT는 피트니스
            '헬스장': {'industry': 3, 'confidence': 0.80, 'name': '피트니스'},
            '동창회': {'industry': 6, 'confidence': 0.90, 'name': '동문회'},
            '동문회': {'industry': 6, 'confidence': 0.90, 'name': '동문회'}
        }

        # Industry 이름 매핑
        self.industry_names = {
            1: '학원',
            2: '온라인 강의',
            3: '피트니스',
            4: '공연/행사',
            5: '모임',
            6: '동문회',
            7: '병원',
            8: '부동산',
            9: '기타'
        }

        # Purpose 이름 매핑
        self.purpose_names = {
            1: '쿠폰',
            2: '공지/안내',
            3: '뉴스레터',
            4: '설문',
            5: '리포트',
            6: '결제 안내',
            7: '콘텐츠 발송',
            8: '예약',
            9: '회원 관리',
            10: '행사 신청',
            11: '기타'
        }

    def extract_nouns(self, text: str) -> List[str]:
        """형태소 분석으로 명사 추출"""
        if not self.okt:
            # 폴백: 단순 키워드 분할
            return self._fallback_extract_keywords(text)

        try:
            # KoNLPy로 명사 추출
            nouns = self.okt.nouns(text)
            return [noun for noun in nouns if len(noun) > 1]  # 2글자 이상만
        except Exception as e:
            print(f"⚠️ 형태소 분석 실패: {e}")
            return self._fallback_extract_keywords(text)

    def _fallback_extract_keywords(self, text: str) -> List[str]:
        """폴백: 단순 키워드 추출"""
        # 모든 키워드 수집
        all_keywords = []
        for keywords_dict in [self.industry_keywords, self.purpose_keywords]:
            for keywords in keywords_dict.values():
                all_keywords.extend(keywords)

        # 텍스트에서 키워드 찾기
        found_keywords = []
        for keyword in all_keywords:
            if keyword in text:
                found_keywords.append(keyword)

        return found_keywords

    def classify_industry(self, text: str) -> Tuple[int, str, float]:
        """업종 분류"""
        nouns = self.extract_nouns(text)
        scores = {}

        # 각 업종별 점수 계산
        for industry_id, keywords in self.industry_keywords.items():
            score = 0
            matched_keywords = []

            for noun in nouns:
                if noun in keywords:
                    # 정확한 매칭에 높은 점수
                    score += 10
                    matched_keywords.append(noun)
                else:
                    # 부분 매칭 점수
                    for keyword in keywords:
                        if noun in keyword or keyword in noun:
                            score += 3
                            matched_keywords.append(f"{noun}~{keyword}")

            if score > 0:
                scores[industry_id] = {
                    'score': score,
                    'matched': matched_keywords
                }

        # 최고 점수 업종 선택
        if scores:
            best_industry = max(scores.keys(), key=lambda x: scores[x]['score'])
            best_score = scores[best_industry]['score']

            # 신뢰도 계산 (최대 100%)
            confidence = min(best_score / 10.0, 1.0)

            return best_industry, self.industry_names[best_industry], confidence

        # 기본값: 기타
        return 9, self.industry_names[9], 0.1

    def classify_purpose(self, text: str) -> Tuple[int, str, float]:
        """목적 분류"""
        nouns = self.extract_nouns(text)
        scores = {}

        # 각 목적별 점수 계산
        for purpose_id, keywords in self.purpose_keywords.items():
            score = 0
            matched_keywords = []

            for noun in nouns:
                if noun in keywords:
                    # 정확한 매칭에 높은 점수
                    score += 10
                    matched_keywords.append(noun)
                else:
                    # 부분 매칭 점수
                    for keyword in keywords:
                        if noun in keyword or keyword in noun:
                            score += 3
                            matched_keywords.append(f"{noun}~{keyword}")

            if score > 0:
                scores[purpose_id] = {
                    'score': score,
                    'matched': matched_keywords
                }

        # 최고 점수 목적 선택
        if scores:
            best_purpose = max(scores.keys(), key=lambda x: scores[x]['score'])
            best_score = scores[best_purpose]['score']

            # 신뢰도 계산 (최대 100%)
            confidence = min(best_score / 10.0, 1.0)

            return best_purpose, self.purpose_names[best_purpose], confidence

        # 기본값: 기타
        return 11, self.purpose_names[11], 0.1

    def classify(self, text: str, use_cache: bool = True) -> Dict:
        """전체 분류 (industry + purpose) - 기존 키워드 기반"""
        # 캐시 확인
        cache_key = text.strip()
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # 분류 실행
        industry_id, industry_name, industry_confidence = self.classify_industry(text)
        purpose_id, purpose_name, purpose_confidence = self.classify_purpose(text)

        result = {
            'industry': {
                'id': industry_id,
                'name': industry_name,
                'confidence': industry_confidence
            },
            'purpose': {
                'id': purpose_id,
                'name': purpose_name,
                'confidence': purpose_confidence
            },
            'overall_confidence': (industry_confidence + purpose_confidence) / 2,
            'method': 'rule_based_keywords',
            'extracted_nouns': self.extract_nouns(text) if self.okt else []
        }

        # 캐시 저장
        if use_cache:
            self._cache[cache_key] = result

        return result

    def classify_with_agent1_context(self, text: str, agent1_result: Dict, use_cache: bool = True) -> Dict:
        """Agent1 분석 결과를 활용한 문맥 기반 분류"""
        # 캐시 키에 Agent1 결과도 포함
        cache_key = f"{text.strip()}|{agent1_result.get('intent', {}).get('intent', '')}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Agent1 의도로 Purpose 분류
        purpose_result = self._classify_purpose_by_intent(agent1_result)

        # 2. Agent1 변수로 Industry 분류
        industry_result = self._classify_industry_by_entities(agent1_result)

        # 3. 키워드 기반 분류로 보완
        keyword_result = self.classify(text, use_cache=False)

        # 4. 최종 결과 통합 및 신뢰도 계산
        final_result = self._combine_classification_results(
            purpose_result, industry_result, keyword_result, agent1_result
        )

        # 캐시 저장
        if use_cache:
            self._cache[cache_key] = final_result

        return final_result

    def _classify_purpose_by_intent(self, agent1_result: Dict) -> Dict:
        """Agent1 의도를 기반으로 Purpose 분류"""
        intent_info = agent1_result.get('intent', {})
        intent = intent_info.get('intent', '')
        intent_confidence = intent_info.get('confidence', 0.0)

        # 의도 매핑 확인
        if intent in self.intent_purpose_mapping:
            mapping = self.intent_purpose_mapping[intent]
            purpose_id = mapping['primary']
            purpose_name = self.purpose_names[purpose_id]

            # Agent1 의도 신뢰도와 매핑 신뢰도 결합
            combined_confidence = min(intent_confidence * mapping['confidence'], 1.0)

            return {
                'id': purpose_id,
                'name': purpose_name,
                'confidence': combined_confidence,
                'method': 'intent_mapping',
                'source_intent': intent,
                'secondary_options': mapping.get('secondary', [])
            }

        # 매핑되지 않으면 기본값
        return {
            'id': 11,
            'name': '기타',
            'confidence': 0.1,
            'method': 'intent_fallback',
            'source_intent': intent
        }

    def _classify_industry_by_entities(self, agent1_result: Dict) -> Dict:
        """Agent1 추출 변수를 기반으로 Industry 분류"""
        variables = agent1_result.get('variables', {})
        original_input = agent1_result.get('analysis', {}).get('user_input', '')

        # 1. 원본 텍스트에서 특별 키워드 우선 확인
        for keyword, mapping in self.special_keyword_mapping.items():
            if keyword in original_input:
                return {
                    'id': mapping['industry'],
                    'name': mapping['name'],
                    'confidence': mapping['confidence'],
                    'method': 'special_keyword',
                    'matched_keyword': keyword,
                    'source_text': original_input
                }

        # 2. 주요 변수에서 키워드 추출
        key_entities = []
        for var_value in variables.values():
            if var_value and var_value != '없음':
                # 형태소 분석으로 키워드 추출
                entities = self.extract_nouns(var_value)
                key_entities.extend(entities)

        # 3. 엔티티 패턴 매칭
        for pattern, mapping in self.entity_pattern_mapping.items():
            if all(entity in ' '.join(key_entities) for entity in pattern):
                return {
                    'id': mapping['industry'],
                    'name': self.industry_names[mapping['industry']],
                    'confidence': mapping['confidence'],
                    'method': 'entity_pattern',
                    'matched_pattern': pattern,
                    'extracted_entities': key_entities
                }

        # 4. 단일 키워드 매칭
        for entity in key_entities:
            for industry_id, keywords in self.industry_keywords.items():
                if entity in keywords:
                    return {
                        'id': industry_id,
                        'name': self.industry_names[industry_id],
                        'confidence': 0.70,  # 단일 매칭은 낮은 신뢰도
                        'method': 'single_entity_match',
                        'matched_entity': entity,
                        'extracted_entities': key_entities
                    }

        # 5. 매칭되지 않으면 기본값
        return {
            'id': 9,
            'name': '기타',
            'confidence': 0.1,
            'method': 'entity_fallback',
            'extracted_entities': key_entities
        }

    def _combine_classification_results(self, purpose_result: Dict, industry_result: Dict,
                                       keyword_result: Dict, agent1_result: Dict) -> Dict:
        """분류 결과들을 통합하고 최종 신뢰도 계산"""

        # Purpose 선택 (Agent1 기반 우선)
        if purpose_result['confidence'] > 0.5:
            final_purpose = purpose_result
        elif keyword_result['purpose']['confidence'] > purpose_result['confidence']:
            final_purpose = keyword_result['purpose']
            final_purpose['method'] = 'keyword_fallback'
        else:
            final_purpose = purpose_result

        # Industry 선택 (엔티티 패턴 기반 우선)
        if industry_result['confidence'] > 0.5:
            final_industry = industry_result
        elif keyword_result['industry']['confidence'] > industry_result['confidence']:
            final_industry = keyword_result['industry']
            final_industry['method'] = 'keyword_fallback'
        else:
            final_industry = industry_result

        # 전체 신뢰도 계산 (가중 평균)
        purpose_weight = 0.6 if final_purpose['method'].startswith('intent') else 0.4
        industry_weight = 0.6 if final_industry['method'].startswith('entity') else 0.4

        overall_confidence = (
            final_purpose['confidence'] * purpose_weight +
            final_industry['confidence'] * industry_weight
        )

        return {
            'industry': {
                'id': final_industry['id'],
                'name': final_industry['name'],
                'confidence': final_industry['confidence'],
                'method': final_industry['method']
            },
            'purpose': {
                'id': final_purpose['id'],
                'name': final_purpose['name'],
                'confidence': final_purpose['confidence'],
                'method': final_purpose['method']
            },
            'overall_confidence': overall_confidence,
            'method': 'agent1_context_enhanced',
            'agent1_analysis': {
                'intent': agent1_result.get('intent', {}),
                'variables': agent1_result.get('variables', {}),
                'completeness': agent1_result.get('analysis', {}).get('mandatory_check', {}).get('completeness_score', 0.0)
            },
            'classification_details': {
                'purpose_classification': purpose_result,
                'industry_classification': industry_result,
                'keyword_fallback': keyword_result
            }
        }

    def get_industry_list(self) -> List[Dict]:
        """전체 업종 목록 반환"""
        return [
            {'id': id, 'name': name}
            for id, name in self.industry_names.items()
        ]

    def get_purpose_list(self) -> List[Dict]:
        """전체 목적 목록 반환"""
        return [
            {'id': id, 'name': name}
            for id, name in self.purpose_names.items()
        ]

    def clear_cache(self):
        """캐시 클리어"""
        self._cache.clear()

    def get_cache_info(self) -> Dict:
        """캐시 정보 반환"""
        return {
            'cache_size': len(self._cache),
            'analyzer_available': self.okt is not None
        }


# 전역 분류기 인스턴스 (싱글톤)
_classifier_instance = None

def get_classifier() -> IndustryClassifier:
    """싱글톤 분류기 인스턴스 반환"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IndustryClassifier()
    return _classifier_instance


# 편의 함수들
def classify_text(text: str) -> Dict:
    """텍스트 분류 편의 함수 (키워드 기반)"""
    classifier = get_classifier()
    return classifier.classify(text)

def classify_with_context(text: str, agent1_result: Dict) -> Dict:
    """Agent1 문맥을 활용한 분류 편의 함수"""
    classifier = get_classifier()
    return classifier.classify_with_agent1_context(text, agent1_result)

def classify_industry(text: str) -> Tuple[int, str, float]:
    """업종 분류 편의 함수"""
    classifier = get_classifier()
    return classifier.classify_industry(text)

def classify_purpose(text: str) -> Tuple[int, str, float]:
    """목적 분류 편의 함수"""
    classifier = get_classifier()
    return classifier.classify_purpose(text)


if __name__ == "__main__":
    # 테스트 코드
    print("=== Industry/Purpose 분류기 테스트 ===")

    classifier = IndustryClassifier()

    test_cases = [
        "내일 오후 2시 카페에서 동창회 모임이 있다고 알림톡 보내줘",
        "헬스장 PT 예약 확인 메시지 만들어줘",
        "온라인 강의 수강신청 안내 템플릿",
        "병원 진료 예약 알림 메시지",
        "할인 쿠폰 발급 안내"
    ]

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{i}. 테스트: {test_text}")
        result = classifier.classify(test_text)

        print(f"   업종: {result['industry']['name']} (신뢰도: {result['industry']['confidence']:.2f})")
        print(f"   목적: {result['purpose']['name']} (신뢰도: {result['purpose']['confidence']:.2f})")
        print(f"   전체 신뢰도: {result['overall_confidence']:.2f}")

        if result['extracted_nouns']:
            print(f"   추출된 명사: {', '.join(result['extracted_nouns'])}")

    print(f"\n캐시 정보: {classifier.get_cache_info()}")