#!/usr/bin/env python3
"""
CSV 데이터 전체 테스트
시간이 오래 걸려도 모든 데이터를 테스트
"""

import requests
import csv
import json
import time
import random
from datetime import datetime

def load_csv_data():
    """모든 CSV 데이터 로드"""
    reject_texts = []
    approve_texts = []

    # 반려 케이스
    reject_file = 'csvfile/[보안 자료] 카카오 알림톡 데이터 취합_2차 텍스트 완료 버전_V1.0_250513.xlsx - 자버에서 반려 받으.csv'
    try:
        with open(reject_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('텍스트') and row['텍스트'].strip():
                    reject_texts.append(row['텍스트'].strip())
        print(f"✅ 반려 케이스 로드: {len(reject_texts)}개")
    except Exception as e:
        print(f"❌ 반료 CSV 로드 오류: {e}")

    # 승인 케이스
    approve_file = 'csvfile/[보안 자료] 카카오 알림톡 데이터 취합_2차 텍스트 완료 버전_V1.0_250513.xlsx - 자버에서 승인 받ᄋ.csv'
    try:
        with open(approve_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('텍스트') and row['텍스트'].strip():
                    approve_texts.append(row['텍스트'].strip())
        print(f"✅ 승인 케이스 로드: {len(approve_texts)}개")
    except Exception as e:
        print(f"❌ 승인 CSV 로드 오류: {e}")

    return reject_texts, approve_texts

def test_single_case(text, expected, test_id, retry_count=2):
    """단일 케이스 테스트 (재시도 기능 포함)"""
    for attempt in range(retry_count + 1):
        try:
            start_time = time.time()
            response = requests.post(
                'http://127.0.0.1:8000/ai/templates',
                json={'userId': 999, 'requestContent': text},
                timeout=40  # 타임아웃 증가
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                # content가 있고 비어있지 않으면 APPROVE
                actual = 'APPROVE' if ('content' in result and result['content'] and result['content'].strip()) else 'REJECT'
                error = ''
                break
            elif response.status_code == 504:  # 타임아웃
                if attempt < retry_count:
                    print(f"   🔄 타임아웃 재시도 {attempt + 1}/{retry_count}")
                    time.sleep(5)
                    continue
                actual = 'ERROR'
                error = f'HTTP {response.status_code}: 타임아웃'
            else:
                actual = 'ERROR'
                error = f'HTTP {response.status_code}: {response.text[:100]}'
                break

        except requests.exceptions.Timeout:
            if attempt < retry_count:
                print(f"   🔄 요청 타임아웃 재시도 {attempt + 1}/{retry_count}")
                time.sleep(5)
                continue
            response_time = time.time() - start_time
            actual = 'ERROR'
            error = '요청 타임아웃'
        except Exception as e:
            response_time = time.time() - start_time
            actual = 'ERROR'
            error = str(e)[:100]
            break

    success = actual == expected
    return {
        'id': test_id,
        'text': text,
        'expected': expected,
        'actual': actual,
        'success': success,
        'time': response_time,
        'error': error
    }

def run_comprehensive_test():
    """전체 데이터에 대한 포괄적 테스트"""

    print("🧪 CSV 전체 데이터 포괄적 테스트 시작")
    print("=" * 80)

    # 데이터 로드
    reject_texts, approve_texts = load_csv_data()

    if not reject_texts and not approve_texts:
        print("❌ 테스트할 데이터가 없습니다.")
        return

    # 테스트 케이스 생성
    test_cases = []

    # 반려 케이스 추가
    for i, text in enumerate(reject_texts):
        test_cases.append({
            'id': f'reject_{i+1:03d}',
            'text': text,
            'expected': 'REJECT'
        })

    # 승인 케이스 추가
    for i, text in enumerate(approve_texts):
        test_cases.append({
            'id': f'approve_{i+1:03d}',
            'text': text,
            'expected': 'APPROVE'
        })

    # 랜덤 순서로 섞기
    random.shuffle(test_cases)

    total_cases = len(test_cases)
    print(f"📊 총 테스트 케이스: {total_cases}개")
    print(f"⏱️  예상 소요 시간: {total_cases * 12 / 60:.1f}분 (케이스당 12초)")
    print(f"🕐 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 결과 저장
    results = []

    # 통계 변수
    success_count = 0
    error_count = 0

    # 배치별 저장 (100개씩)
    batch_size = 100

    try:
        for idx, case in enumerate(test_cases, 1):
            print(f"[{idx}/{total_cases}] {case['id']}: {case['text'][:60]}...")

            # 테스트 실행
            result = test_single_case(case['text'], case['expected'], case['id'])
            results.append(result)

            # 결과 출력
            status_icon = "✅" if result['success'] else "❌"
            if result['actual'] == 'ERROR':
                status_icon = "🔥"
                error_count += 1
            elif result['success']:
                success_count += 1

            print(f"   {status_icon} {result['actual']} (예상: {result['expected']}) - {result['time']:.2f}s")

            if result['error']:
                print(f"   ⚠️  오류: {result['error'][:80]}...")

            # 진행률 출력
            progress = (idx / total_cases) * 100
            success_rate = (success_count / idx) * 100
            error_rate = (error_count / idx) * 100

            print(f"   📈 진행률: {progress:.1f}% | 성공률: {success_rate:.1f}% | 오류율: {error_rate:.1f}%")

            # 배치별 저장
            if idx % batch_size == 0:
                save_batch_results(results, idx // batch_size)
                print(f"   💾 중간 저장 완료 (배치 {idx // batch_size})")

            # API 할당량 보호 대기
            if idx < total_cases:
                wait_time = 12  # 분당 5회 제한에 맞춰 12초 대기
                print(f"   ⏳ API 보호 대기... ({wait_time}초)")
                time.sleep(wait_time)

    except KeyboardInterrupt:
        print(f"\n⏹️  사용자에 의해 테스트가 중단되었습니다. (진행률: {idx}/{total_cases})")

    # 최종 결과 저장
    save_final_results(results, reject_texts, approve_texts)

    print(f"\n🏁 테스트 완료!")
    print(f"🕐 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def save_batch_results(results, batch_num):
    """배치별 중간 결과 저장"""
    filename = f'csv_test_batch_{batch_num}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test_id', 'input_text', 'expected', 'actual', 'success', 'response_time', 'error'])

        for r in results[-100:]:  # 마지막 100개만 저장
            writer.writerow([
                r['id'], r['text'], r['expected'], r['actual'],
                r['success'], r['time'], r['error']
            ])

def save_final_results(results, reject_texts, approve_texts):
    """최종 결과 저장 및 분석"""

    # 상세 결과 CSV 저장
    with open('csv_comprehensive_test_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test_id', 'input_text', 'expected', 'actual', 'success', 'response_time', 'error'])

        for r in results:
            writer.writerow([
                r['id'], r['text'], r['expected'], r['actual'],
                r['success'], r['time'], r['error']
            ])

    # 결과 분석
    total = len(results)
    if total == 0:
        return

    successes = sum(1 for r in results if r['success'])
    errors = sum(1 for r in results if r['actual'] == 'ERROR')

    reject_results = [r for r in results if r['expected'] == 'REJECT']
    approve_results = [r for r in results if r['expected'] == 'APPROVE']

    reject_correct = sum(1 for r in reject_results if r['success'])
    approve_correct = sum(1 for r in approve_results if r['success'])

    # 오류 분석
    error_types = {}
    for r in results:
        if r['actual'] == 'ERROR':
            error_type = '타임아웃' if '504' in r['error'] or '타임아웃' in r['error'] else \
                        '서버오류' if '500' in r['error'] else \
                        '할당량초과' if '429' in r['error'] else \
                        '기타'
            error_types[error_type] = error_types.get(error_type, 0) + 1

    # 성능 분석
    successful_results = [r for r in results if r['actual'] != 'ERROR']
    avg_time = sum(r['time'] for r in successful_results) / len(successful_results) if successful_results else 0
    max_time = max(r['time'] for r in successful_results) if successful_results else 0
    min_time = min(r['time'] for r in successful_results) if successful_results else 0

    # 보고서 생성
    report = {
        "테스트_개요": {
            "총_테스트": total,
            "성공": successes,
            "실패": total - successes,
            "오류": errors,
            "전체_성공률": f"{(successes/total*100):.1f}%",
            "오류율": f"{(errors/total*100):.1f}%"
        },
        "분류별_정확도": {
            "반려_케이스": {
                "총개수": len(reject_results),
                "올바른_반려": reject_correct,
                "정확도": f"{(reject_correct/len(reject_results)*100):.1f}%" if reject_results else "N/A"
            },
            "승인_케이스": {
                "총개수": len(approve_results),
                "올바른_승인": approve_correct,
                "정확도": f"{(approve_correct/len(approve_results)*100):.1f}%" if approve_results else "N/A"
            }
        },
        "오류_분석": error_types,
        "성능_지표": {
            "평균_응답시간": f"{avg_time:.2f}초",
            "최대_응답시간": f"{max_time:.2f}초",
            "최소_응답시간": f"{min_time:.2f}초",
            "처리된_케이스": len(successful_results)
        }
    }

    # JSON 보고서 저장
    with open('csv_comprehensive_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 콘솔 출력
    print("\n" + "=" * 80)
    print("🎯 최종 테스트 결과")
    print("=" * 80)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n📁 상세 결과: csv_comprehensive_test_results.csv")
    print(f"📊 분석 보고서: csv_comprehensive_test_report.json")

if __name__ == "__main__":
    run_comprehensive_test()