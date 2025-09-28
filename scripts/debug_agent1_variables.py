#!/usr/bin/env python3
"""
Agent1 변수 추출 디버깅
비동기 구현 후 변수 추출이 제대로 작동하는지 확인
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.agent1 import Agent1

async def debug_agent1_variables():
    """Agent1 변수 추출 디버깅"""

    print("Agent1 변수 추출 디버깅")
    print("=" * 50)

    try:
        agent1 = Agent1()
        print("Agent1 초기화 완료")
    except Exception as e:
        print(f"Agent1 초기화 실패: {e}")
        return

    test_inputs = [
        "내일 오후 2시에 강남 스타벅스에서 독서모임이 있습니다. 참가자들에게 알림을 보내주세요",
        "백엔드 부트캠프 설명회가 다음 주 화요일 오후 7시에 온라인으로 진행됩니다",
        "고객님께서 주문하신 상품이 준비되었습니다. 매장에서 픽업 가능합니다",
        "독서모임 알림",
        "이벤트 안내"
    ]

    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n테스트 {i}: {test_input}")
        print("-" * 60)

        try:
            # 동기 버전 테스트
            print(" 동기 분석 결과:")
            sync_result = agent1.analyze_query(test_input)
            print(f"   Variables: {sync_result.get('variables', {})}")
            print(f"   Intent: {sync_result.get('intent', {}).get('intent', 'Unknown')}")
            print(f"   Confidence: {sync_result.get('intent', {}).get('confidence', 0):.2f}")
            print(f"   Mandatory check: {sync_result.get('mandatory_check', {})}")

            # 비동기 버전 테스트
            print("\n 비동기 분석 결과:")
            async_result = await agent1.analyze_query_async(test_input)
            print(f"   Variables: {async_result.get('variables', {})}")
            print(f"   Intent: {async_result.get('intent', {}).get('intent', 'Unknown')}")
            print(f"   Confidence: {async_result.get('intent', {}).get('confidence', 0):.2f}")
            print(f"   Mandatory check: {async_result.get('mandatory_check', {})}")

            # 결과 비교
            sync_vars = sync_result.get('variables', {})
            async_vars = async_result.get('variables', {})

            print(f"\n 동기 vs 비동기 비교:")
            print(f"   Variables 일치: {sync_vars == async_vars}")
            print(f"   Intent 일치: {sync_result.get('intent', {}).get('intent') == async_result.get('intent', {}).get('intent')}")

            # 주요 변수 체크
            what_subject = sync_vars.get('무엇을 (What/Subject)', '없음')
            print(f"   '무엇을 (What/Subject)' 추출됨: {what_subject != '없음'}")
            if what_subject != '없음':
                print(f"   추출된 값: '{what_subject}'")

        except Exception as e:
            print(f"   오류: {e}")

    # 직접 변수 추출 테스트
    print(f"\n직접 변수 추출 테스트")
    print("-" * 40)

    test_text = "내일 오후 2시에 강남 스타벅스에서 독서모임이 있습니다"

    try:
        # 동기 변수 추출
        print(" 동기 변수 추출:")
        sync_vars = agent1.variable_extractor.extract_variables(test_text)
        print(f"   결과: {sync_vars}")

        # 비동기 변수 추출
        print("\n 비동기 변수 추출:")
        async_vars = await agent1.variable_extractor.extract_variables_async(test_text)
        print(f"   결과: {async_vars}")

        # 필수 변수 체크
        print(f"\n 필수 변수 체크:")
        required_vars = agent1.variable_extractor.determine_required_variables_by_context(test_text)
        print(f"   필수 변수: {required_vars}")

        mandatory_check = agent1.variable_extractor.check_mandatory_variables(sync_vars, test_text)
        print(f"   완성도: {mandatory_check}")

    except Exception as e:
        print(f"   변수 추출 오류: {e}")

if __name__ == "__main__":
    asyncio.run(debug_agent1_variables())