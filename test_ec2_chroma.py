#!/usr/bin/env python3
"""
EC2 환경에서 Chroma DB 테스트
"""

def test_ec2_environment():
    """EC2 환경 테스트"""
    print("=== EC2 환경 테스트 ===")
    
    try:
        from ec2_config import (
            is_ec2_environment, 
            get_environment_paths, 
            setup_ec2_directories,
            print_environment_info
        )
        
        print("1. 환경 정보 확인")
        print_environment_info()
        
        print("\n2. 디렉토리 설정")
        setup_ec2_directories()
        
        print("\n3. 경로 확인")
        paths = get_environment_paths()
        for name, path in paths.items():
            from pathlib import Path
            exists = Path(path).exists()
            print(f"  {name}: {path} {'✅' if exists else '❌'}")
            
        return True
        
    except Exception as e:
        print(f"❌ EC2 설정 테스트 실패: {e}")
        return False

def test_chroma_with_ec2_paths():
    """EC2 경로를 사용한 Chroma DB 테스트"""
    print("\n=== EC2 Chroma DB 테스트 ===")
    
    try:
        from src.core.index_manager import IndexManager
        import numpy as np
        
        print("1. EC2 설정으로 IndexManager 초기화")
        manager = IndexManager()  # 자동으로 EC2 경로 사용
        
        print("2. 테스트 컬렉션 생성")
        test_data = [
            "EC2에서 Chroma DB 테스트",
            "클라우드 환경 벡터 검색",
            "배포 환경 검증 완료"
        ]
        
        def dummy_encode(texts):
            return np.array([np.random.rand(384) for _ in texts])
        
        collection = manager.get_chroma_collection(
            collection_name="ec2_test",
            data=test_data,
            encode_func=dummy_encode
        )
        
        print(f"✅ 컬렉션 생성 성공: {collection.count()}개 문서")
        
        print("3. 검색 테스트")
        results = manager.query_similar(
            collection_name="ec2_test",
            query_text="클라우드 테스트",
            encode_func=dummy_encode,
            top_k=2
        )
        
        print(f"✅ 검색 성공: {len(results)}개 결과")
        for text, score in results:
            print(f"  - {text} (유사도: {score:.3f})")
        
        print("4. 캐시 정보 확인")
        cache_info = manager.get_cache_info()
        print(f"  캐시 경로: {cache_info['cache_dir']}")
        print(f"  Chroma DB 경로: {cache_info['chroma_db_path']}")
        print(f"  총 크기: {cache_info['total_size_mb']} MB")
        print(f"  컬렉션: {list(cache_info['chroma_collections'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chroma DB 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_system():
    """전체 시스템 테스트"""
    print("\n=== 전체 시스템 테스트 ===")
    
    try:
        from api import get_template_api
        
        print("1. API 초기화 (EC2 환경)")
        api = get_template_api()
        
        print("2. Health Check")
        health = api.health_check()
        if health.get("status") == "healthy":
            print("✅ 시스템 정상")
            print(f"  인덱스: {health['indexes']}")
        else:
            print(f"❌ 시스템 문제: {health}")
            return False
            
        print("3. 간단한 템플릿 생성 테스트")
        result = api.generate_template("EC2 배포 완료 알림")
        
        if result.get("success"):
            print("✅ 템플릿 생성 성공")
        else:
            print(f"❌ 템플릿 생성 실패: {result.get('error')}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 전체 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 EC2 Chroma DB 종합 테스트")
    
    test1 = test_ec2_environment()
    test2 = test_chroma_with_ec2_paths() 
    test3 = test_full_system()
    
    print(f"\n=== 테스트 결과 ===")
    print(f"환경 설정: {'✅' if test1 else '❌'}")
    print(f"Chroma DB: {'✅' if test2 else '❌'}")
    print(f"전체 시스템: {'✅' if test3 else '❌'}")
    
    if test1 and test2 and test3:
        print("\n🎉 EC2 배포 준비 완료!")
        print("배포 명령: ./deploy_ec2.sh")
    else:
        print("\n⚠️  일부 테스트 실패. 문제를 해결 후 재시도하세요.")