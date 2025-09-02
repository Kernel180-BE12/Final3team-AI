#!/usr/bin/env python3
"""
FAISS 인덱스 및 데이터 캐싱 관리자
한 번 로드하고 임베딩한 데이터를 재사용하여 성능 최적화
"""
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import faiss
import numpy as np
from datetime import datetime

class IndexManager:
    """FAISS 인덱스와 predata 캐싱 관리"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 캐시 파일 경로
        self.templates_cache = self.cache_dir / "templates.cache"
        self.guidelines_cache = self.cache_dir / "guidelines.cache"
        self.predata_cache = self.cache_dir / "predata.cache"
        self.metadata_cache = self.cache_dir / "metadata.json"
        
        print(" IndexManager 초기화 완료")
    
    def _get_files_hash(self, file_paths: List[Path]) -> str:
        """파일들의 해시값 계산 (수정 시간 기반)"""
        hash_input = ""
        for path in file_paths:
            if path.exists():
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
                hash_input += f"{path.name}_{mtime}_{size}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _load_predata_files(self) -> Dict[str, str]:
        """predata 폴더의 모든 파일 로드"""
        predata_dir = Path("predata")
        
        if not predata_dir.exists():
            print(" predata 폴더가 존재하지 않습니다.")
            return {}
        
        predata_files = [
            "cleaned_add_infotalk.md",
            "cleaned_alrimtalk.md",
            "cleaned_black_list.md", 
            "cleaned_content-guide.md",
            "cleaned_info_simsa.md",
            "cleaned_message.md",
            "cleaned_message_yuisahang.md",
            "cleaned_run_message.md",
            "cleaned_white_list.md",
            "cleaned_zipguide.md",
            "pdf_extraction_results.txt"
        ]
        
        data = {}
        for filename in predata_files:
            file_path = predata_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        data[filename] = content
                except Exception as e:
                    print(f" {filename} 로드 실패: {e}")
        
        return data
    
    def get_predata_cache(self) -> Dict[str, str]:
        """predata 캐시 가져오기 (필요시 새로 로드)"""
        predata_dir = Path("predata")
        
        # predata 파일들의 경로 목록
        predata_files = list(predata_dir.glob("*.md")) + list(predata_dir.glob("*.txt"))
        current_hash = self._get_files_hash(predata_files)
        
        # 메타데이터 확인
        metadata = self._load_metadata()
        cached_hash = metadata.get("predata_hash", "")
        
        # 캐시가 유효한지 확인
        if (self.predata_cache.exists() and 
            cached_hash == current_hash):
            print(" predata 캐시에서 로드")
            with open(self.predata_cache, 'rb') as f:
                return pickle.load(f)
        
        # 새로 로드하고 캐시 저장
        print(" predata 파일들을 새로 로드 중...")
        data = self._load_predata_files()
        
        # 캐시 저장
        with open(self.predata_cache, 'wb') as f:
            pickle.dump(data, f)
        
        # 메타데이터 업데이트
        metadata["predata_hash"] = current_hash
        metadata["predata_updated"] = datetime.now().isoformat()
        self._save_metadata(metadata)
        
        print(f" predata 캐시 업데이트 완료: {len(data)}개 파일")
        return data
    
    def get_guidelines_chunks(self, chunk_func, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """가이드라인 청크 캐시 가져오기"""
        predata = self.get_predata_cache()
        
        # 청크 캐시 키 생성
        chunk_key = f"chunks_{chunk_size}_{overlap}"
        metadata = self._load_metadata()
        
        # 캐시된 청크가 있고 predata가 변경되지 않았으면 재사용
        if (self.guidelines_cache.exists() and 
            chunk_key in metadata.get("chunk_configs", {})):
            print(" 가이드라인 청크 캐시에서 로드")
            with open(self.guidelines_cache, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data.get(chunk_key, [])
        
        # 새로 청킹
        print(" 가이드라인 청킹 중...")
        all_chunks = []
        
        for filename, content in predata.items():
            if content.strip():
                chunks = chunk_func(content, chunk_size, overlap)
                all_chunks.extend(chunks)
                print(f"   {filename}: {len(chunks)}개 청크")
        
        # 캐시 저장
        cached_data = {}
        if self.guidelines_cache.exists():
            with open(self.guidelines_cache, 'rb') as f:
                cached_data = pickle.load(f)
        
        cached_data[chunk_key] = all_chunks
        
        with open(self.guidelines_cache, 'wb') as f:
            pickle.dump(cached_data, f)
        
        # 메타데이터 업데이트
        if "chunk_configs" not in metadata:
            metadata["chunk_configs"] = {}
        metadata["chunk_configs"][chunk_key] = {
            "created": datetime.now().isoformat(),
            "chunks_count": len(all_chunks)
        }
        self._save_metadata(metadata)
        
        print(f" 청크 캐시 저장 완료: {len(all_chunks)}개")
        return all_chunks
    
    def get_faiss_index(self, 
                       index_name: str, 
                       data: List[str], 
                       encode_func,
                       build_func) -> Optional[faiss.Index]:
        """FAISS 인덱스 캐시 가져오기 (필요시 새로 구축)"""
        
        index_path = self.cache_dir / f"{index_name}.faiss"
        vectors_path = self.cache_dir / f"{index_name}_vectors.npy"
        
        # 데이터 해시 계산
        data_str = "\n".join(data[:5])  # 처음 5개만으로 해시 계산 (성능)
        current_hash = hashlib.md5(data_str.encode()).hexdigest()
        
        metadata = self._load_metadata()
        cached_hash = metadata.get(f"{index_name}_hash", "")
        
        # 캐시가 유효한지 확인
        if (index_path.exists() and vectors_path.exists() and 
            cached_hash == current_hash):
            print(f" {index_name} FAISS 인덱스 캐시에서 로드")
            try:
                index = faiss.read_index(str(index_path))
                return index
            except Exception as e:
                print(f" {index_name} 캐시 로드 실패: {e}")
        
        # 새로 구축
        print(f" {index_name} FAISS 인덱스 구축 중...")
        
        # 최적화된 배치 처리 (API 호출 최소화)
        batch_size = 100  # 배치 크기 증가로 API 호출 횟수 줄이기
        print(f" 총 {len(data)}개 항목을 {batch_size}개씩 최적화 배치 처리")
        
        all_embeddings = []
        import time
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_end = min(i + batch_size, len(data))
            batch_num = i//batch_size + 1
            total_batches = (len(data)-1)//batch_size + 1
            
            print(f" 배치 {batch_num}/{total_batches}: {i+1}-{batch_end} 처리중...")
            
            try:
                start_time = time.time()
                batch_embeddings = encode_func(batch_data)
                end_time = time.time()
                
                all_embeddings.extend(batch_embeddings)
                print(f" 배치 {batch_num} 완료 ({end_time - start_time:.1f}초, {len(batch_embeddings)}개 임베딩)")
                
            except Exception as e:
                print(f" 배치 {batch_num} 처리 실패: {e}")
                continue
        
        if not all_embeddings:
            print(f" {index_name} 임베딩 생성 실패")
            return None
            
        embeddings = np.array(all_embeddings)
        print(f" 임베딩 완료: {embeddings.shape}")
        
        index = build_func(embeddings)
        
        # 캐시 저장
        try:
            faiss.write_index(index, str(index_path))
            np.save(vectors_path, embeddings)
            
            # 메타데이터 업데이트
            metadata[f"{index_name}_hash"] = current_hash
            metadata[f"{index_name}_updated"] = datetime.now().isoformat()
            metadata[f"{index_name}_size"] = len(data)
            self._save_metadata(metadata)
            
            print(f" {index_name} 인덱스 캐시 저장 완료")
        except Exception as e:
            print(f" {index_name} 캐시 저장 실패: {e}")
        
        return index
    
    def _process_batches_parallel(self, batches, encode_func, index_name):
        """배치를 병렬로 처리하여 임베딩 생성"""
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        def process_single_batch(batch_info):
            batch_idx, batch_data = batch_info
            batch_num = batch_idx // 50 + 1
            try:
                start_time = time.time()
                embeddings = encode_func(batch_data)
                end_time = time.time()
                print(f" 배치 {batch_num} 완료 ({end_time - start_time:.1f}초)")
                return embeddings
            except Exception as e:
                print(f" 배치 {batch_num} 실패: {e}")
                return None
        
        all_embeddings = []
        
        # ThreadPoolExecutor로 병렬 처리 (API 호출이므로 I/O bound)
        with ThreadPoolExecutor(max_workers=3) as executor:  # API 제한을 고려하여 3개로 제한
            print(f" 최대 3개 배치 동시 처리 시작...")
            
            # 모든 배치 작업 제출
            future_to_batch = {
                executor.submit(process_single_batch, batch): batch 
                for batch in batches
            }
            
            # 결과 수집 (완료 순서대로)
            for future in concurrent.futures.as_completed(future_to_batch):
                result = future.result()
                if result is not None:
                    all_embeddings.extend(result)
        
        print(f" 병렬 처리 완료: {len(all_embeddings)}개 임베딩 생성")
        return all_embeddings
    
    def _load_metadata(self) -> Dict:
        """메타데이터 로드"""
        if self.metadata_cache.exists():
            try:
                with open(self.metadata_cache, 'r') as f:
                    data = json.load(f)
                    # 데이터가 리스트인 경우 빈 딕셔너리 반환 (호환성 보장)
                    if isinstance(data, list):
                        print(" 메타데이터 형식 불일치 - 새로 생성")
                        return {}
                    return data
            except:
                pass
        return {}
    
    def _save_metadata(self, metadata: Dict):
        """메타데이터 저장"""
        try:
            with open(self.metadata_cache, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f" 메타데이터 저장 실패: {e}")
    
    def clear_cache(self):
        """모든 캐시 삭제"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        print(" 모든 캐시가 삭제되었습니다.")
    
    def get_cache_info(self) -> Dict:
        """캐시 정보 조회"""
        metadata = self._load_metadata()
        
        cache_files = list(self.cache_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
        
        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "metadata": metadata,
            "cached_items": {
                "predata": self.predata_cache.exists(),
                "guidelines": self.guidelines_cache.exists(),
                "templates": self.templates_cache.exists()
            }
        }

# 전역 인덱스 매니저 인스턴스
_index_manager = None

def get_index_manager() -> IndexManager:
    """싱글톤 인덱스 매니저 가져오기"""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager