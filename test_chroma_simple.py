#!/usr/bin/env python3
"""
Chroma DB 기본 기능 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_chroma_basic():
    """Chroma DB 기본 기능 테스트"""
    try:
        import chromadb
        from chromadb import PersistentClient
        
        # 클라이언트 생성
        client = PersistentClient(path="./test_chroma_db")
        print("✅ Chroma DB 클라이언트 초기화 성공")
        
        # 컬렉션 생성
        collection_name = "test_collection"
        
        # 기존 컬렉션 삭제 (테스트용)
        try:
            existing_collections = [col.name for col in client.list_collections()]
            if collection_name in existing_collections:
                client.delete_collection(name=collection_name)
                print(f"✅ 기존 컬렉션 '{collection_name}' 삭제")
        except:
            pass
        
        # 새 컬렉션 생성
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ 컬렉션 '{collection_name}' 생성 성공")
        
        # 테스트 데이터
        documents = [
            "안녕하세요, 고객님. 카페에서 모임이 있습니다.",
            "긴급 점검 안내드립니다.",
            "신규 게임 아발론이 출시되었습니다."
        ]
        
        # 간단한 임베딩 (랜덤)
        import numpy as np
        embeddings = [
            np.random.rand(384).tolist() for _ in documents
        ]
        
        # 데이터 추가
        ids = [f"doc_{i}" for i in range(len(documents))]
        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        print(f"✅ {len(documents)}개 문서 추가 성공")
        
        # 검색 테스트
        query_embedding = [np.random.rand(384).tolist()]
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=2
        )
        
        print(f"✅ 검색 결과: {len(results['documents'][0])}개 문서 반환")
        for i, doc in enumerate(results['documents'][0]):
            print(f"  - {doc[:50]}...")
        
        # 컬렉션 정보
        print(f"✅ 컬렉션 크기: {collection.count()}개 문서")
        
        # 정리
        client.delete_collection(name=collection_name)
        print(f"✅ 테스트 컬렉션 삭제 완료")
        
        return True
        
    except ImportError as e:
        print(f"❌ ChromaDB import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ Chroma DB 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_index_manager():
    """IndexManager Chroma DB 기능 테스트"""
    try:
        from src.core.index_manager import IndexManager
        
        # IndexManager 초기화
        index_manager = IndexManager(
            cache_dir="test_cache",
            chroma_db_path="test_chroma_index"
        )
        print("✅ IndexManager 초기화 성공")
        
        # 테스트 데이터
        test_data = [
            "안녕하세요 고객님",
            "긴급 점검 안내",
            "게임 출시 알림"
        ]
        
        # 간단한 인코딩 함수 (랜덤)
        def dummy_encode_func(texts):
            import numpy as np
            return np.array([np.random.rand(384) for _ in texts])
        
        # 컬렉션 생성
        collection = index_manager.get_chroma_collection(
            collection_name="test_guidelines",
            data=test_data,
            encode_func=dummy_encode_func
        )
        print("✅ 컬렉션 생성 성공")
        
        # 검색 테스트
        results = index_manager.query_similar(
            collection_name="test_guidelines",
            query_text="고객 안내",
            encode_func=dummy_encode_func,
            top_k=2
        )
        print(f"✅ 검색 결과: {len(results)}개")
        for text, score in results:
            print(f"  - {text} (유사도: {score:.3f})")
        
        # 캐시 정보
        cache_info = index_manager.get_cache_info()
        print(f"✅ 캐시 정보: {cache_info['chroma_collections']}")
        
        # 정리
        index_manager.clear_cache()
        print("✅ 테스트 캐시 정리 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ IndexManager 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Chroma DB 테스트 시작 ===\n")
    
    print("1. 기본 Chroma DB 기능 테스트")
    success1 = test_chroma_basic()
    
    print("\n2. IndexManager Chroma DB 기능 테스트")
    success2 = test_index_manager()
    
    print(f"\n=== 테스트 결과 ===")
    print(f"기본 기능: {'✅ 성공' if success1 else '❌ 실패'}")
    print(f"IndexManager: {'✅ 성공' if success2 else '❌ 실패'}")
    
    if success1 and success2:
        print("\n🎉 모든 테스트 통과! FAISS → Chroma DB 변경 완료")
    else:
        print("\n⚠️  일부 테스트 실패. 문제를 확인해주세요.")