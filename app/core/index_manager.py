#!/usr/bin/env python3
"""
Chroma DB 벡터 데이터베이스 및 데이터 캐싱 관리자
한 번 로드하고 임베딩한 데이터를 재사용하여 성능 최적화
"""
import os
import json
import pickle
import hashlib
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import chromadb
from chromadb import PersistentClient
import numpy as np
from datetime import datetime

class IndexManager:
    """Chroma DB 벡터 데이터베이스와 predata 캐싱 관리"""
    
    def __init__(self, cache_dir: str = None, chroma_db_path: str = None):
        # 환경별 경로 설정
        try:
            from ec2_config import get_environment_paths, get_ec2_optimized_settings
            env_paths = get_environment_paths()
            self.ec2_settings = get_ec2_optimized_settings()
            
            self.cache_dir = Path(cache_dir or env_paths["cache_dir"])
            self.chroma_db_path = Path(chroma_db_path or env_paths["chroma_db_path"])
        except ImportError:
            # 폴백: 기본 경로 사용
            self.cache_dir = Path(cache_dir or "cache")
            self.chroma_db_path = Path(chroma_db_path or "chroma_db")
            self.ec2_settings = {"batch_size": 50, "max_workers": 3}
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        # Chroma DB 클라이언트 초기화
        self.chroma_client = PersistentClient(path=str(self.chroma_db_path))
        
        # 캐시 파일 경로
        self.templates_cache = self.cache_dir / "templates.cache"
        self.guidelines_cache = self.cache_dir / "guidelines.cache"
        self.predata_cache = self.cache_dir / "predata.cache"
        self.metadata_cache = self.cache_dir / "metadata.json"
        
        print(" IndexManager 초기화 완료 (Chroma DB)")
    
    def _get_files_hash(self, file_paths: List[Path]) -> str:
        """파일들의 해시값 계산 (수정 시간 기반)"""
        hash_input = ""
        for path in file_paths:
            if path.exists():
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
                hash_input += f"{path.name}_{mtime}_{size}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _parse_metadata_from_md(self, content: str) -> List[Dict]:
        """MD 파일에서 메타데이터를 파싱하여 청크별로 분리"""
        # HTML 주석에서 메타데이터 추출하는 정규식
        metadata_pattern = r'<!--\s*METADATA:(.*?)-->'

        matches = re.findall(metadata_pattern, content, re.DOTALL)
        parsed_metadata = []

        for match in matches:
            try:
                # METADATA: 키워드를 제거하고 YAML 파싱
                yaml_content = match.strip()
                # 만약 yaml_content가 METADATA:로 시작하면 제거
                if yaml_content.startswith('METADATA:'):
                    yaml_content = yaml_content[9:].strip()

                metadata = yaml.safe_load(yaml_content)
                if isinstance(metadata, dict):
                    parsed_metadata.append(metadata)
                else:
                    parsed_metadata.append({})
            except Exception as e:
                print(f"⚠️ 메타데이터 파싱 실패: {e}")
                # 메타데이터 파싱 실패시 빈 dict 추가
                parsed_metadata.append({})

        return parsed_metadata

    def _extract_content_chunks(self, content: str) -> List[str]:
        """MD 파일에서 메타데이터를 제거한 실제 콘텐츠를 청크별로 분리"""
        # HTML 주석 제거
        content_without_metadata = re.sub(r'<!--\s*METADATA:.*?-->', '', content, flags=re.DOTALL)

        # 연속된 빈 줄 제거 및 정리
        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', content_without_metadata).strip()

        # 메타데이터가 있는 구간별로 콘텐츠 분리
        # 각 메타데이터 주석 이후의 콘텐츠를 추출
        metadata_positions = []
        for match in re.finditer(r'<!--\s*METADATA:.*?-->', content, re.DOTALL):
            metadata_positions.append(match.end())

        if not metadata_positions:
            return [cleaned_content] if cleaned_content else []

        chunks = []
        for i, pos in enumerate(metadata_positions):
            # 다음 메타데이터까지의 콘텐츠 추출
            if i + 1 < len(metadata_positions):
                next_metadata_start = content.find('<!--', pos)
                chunk_content = content[pos:next_metadata_start] if next_metadata_start != -1 else content[pos:]
            else:
                chunk_content = content[pos:]

            # HTML 주석 제거 및 정리
            chunk_content = re.sub(r'<!--\s*METADATA:.*?-->', '', chunk_content, flags=re.DOTALL)
            chunk_content = re.sub(r'\n\s*\n\s*\n', '\n\n', chunk_content).strip()

            if chunk_content:
                chunks.append(chunk_content)

        return chunks

    def _load_predata_files(self) -> Dict[str, Dict]:
        """data/presets 폴더의 모든 파일 로드 (메타데이터 포함)"""
        predata_dir = Path("data/presets")

        if not predata_dir.exists():
            print(" data/presets 폴더가 존재하지 않습니다.")
            return {}

        predata_files = [
            "cleaned_add_infotalk.md",        # 알림톡 정보
            "cleaned_black_list.md",          # 블랙리스트
            "cleaned_content-guide.md",       # 콘텐츠 가이드
            "cleaned_info_simsa.md",          # 정보성 메시지 심사
            "cleaned_message.md",             # 메시지 가이드
            "cleaned_message_yuisahang.md",   # 유사행 메시지
            "cleaned_white_list.md",          # 화이트리스트
            "cleaned_zipguide.md",            # 집행가이드
            "info_comm_law_guide.yaml"       # 정보통신망법 가이드
        ]

        data = {}
        for filename in predata_files:
            file_path = predata_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                        if filename.endswith('.md'):
                            # MD 파일인 경우 메타데이터와 콘텐츠 분리
                            metadata_list = self._parse_metadata_from_md(content)
                            content_chunks = self._extract_content_chunks(content)

                            data[filename] = {
                                'content_chunks': content_chunks,
                                'metadata_list': metadata_list,
                                'raw_content': content
                            }
                        elif filename.endswith('.yaml') or filename.endswith('.yml'):
                            # YAML 파일인 경우 특별 처리
                            try:
                                yaml_data = yaml.safe_load(content)
                                if isinstance(yaml_data, dict):
                                    # YAML을 텍스트로 변환하여 청크화
                                    yaml_text = yaml.dump(yaml_data, allow_unicode=True, default_flow_style=False)
                                    data[filename] = {
                                        'content_chunks': [yaml_text],
                                        'metadata_list': [{'file_type': 'yaml_guide', 'source_file': filename}],
                                        'raw_content': content
                                    }
                                else:
                                    # YAML 파싱 실패시 텍스트로 처리
                                    data[filename] = {
                                        'content_chunks': [content],
                                        'metadata_list': [{'file_type': 'text', 'source_file': filename}],
                                        'raw_content': content
                                    }
                            except Exception as e:
                                print(f"⚠️ YAML 파싱 실패 {filename}: {e}")
                                data[filename] = {
                                    'content_chunks': [content],
                                    'metadata_list': [{'file_type': 'text', 'source_file': filename}],
                                    'raw_content': content
                                }
                        else:
                            # TXT 파일 등 기타 파일인 경우 기존 방식
                            data[filename] = {
                                'content_chunks': [content],
                                'metadata_list': [{'file_type': 'text', 'source_file': filename}],
                                'raw_content': content
                            }
                except Exception as e:
                    print(f" {filename} 로드 실패: {e}")

        return data
    
    def get_predata_cache(self) -> Dict[str, Dict]:
        """predata 캐시 가져오기 (필요시 새로 로드)"""
        predata_dir = Path("data/presets")

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
    
    def get_guidelines_chunks_with_metadata(self, chunk_func, chunk_size: int = 800, overlap: int = 100) -> Tuple[List[str], List[Dict]]:
        """가이드라인 청크와 메타데이터를 함께 가져오기"""
        predata = self.get_predata_cache()

        # 청크 캐시 키 생성
        chunk_key = f"chunks_meta_{chunk_size}_{overlap}"
        metadata = self._load_metadata()

        # 캐시된 청크가 있고 predata가 변경되지 않았으면 재사용
        if (self.guidelines_cache.exists() and
            chunk_key in metadata.get("chunk_configs", {})):
            print(" 가이드라인 청크+메타데이터 캐시에서 로드")
            with open(self.guidelines_cache, 'rb') as f:
                cached_data = pickle.load(f)
                cached_result = cached_data.get(chunk_key, {})
                return cached_result.get('chunks', []), cached_result.get('metadata_list', [])

        # 새로 청킹
        print(" 가이드라인 청킹+메타데이터 처리 중...")
        all_chunks = []
        all_metadata = []

        for filename, file_data in predata.items():
            if isinstance(file_data, dict) and 'content_chunks' in file_data:
                content_chunks = file_data['content_chunks']
                metadata_list = file_data['metadata_list']

                for i, content in enumerate(content_chunks):
                    if content.strip():
                        chunks = chunk_func(content, chunk_size, overlap)
                        all_chunks.extend(chunks)

                        # 각 청크에 해당하는 메타데이터 추가
                        chunk_metadata = metadata_list[i] if i < len(metadata_list) else {}
                        chunk_metadata['source_file'] = filename

                        for _ in chunks:
                            all_metadata.append(chunk_metadata.copy())

                print(f"   {filename}: {len(content_chunks)}개 섹션 → 총 청크 수 증가")

        # 캐시 저장
        cached_data = {}
        if self.guidelines_cache.exists():
            with open(self.guidelines_cache, 'rb') as f:
                cached_data = pickle.load(f)

        cached_data[chunk_key] = {
            'chunks': all_chunks,
            'metadata_list': all_metadata
        }

        with open(self.guidelines_cache, 'wb') as f:
            pickle.dump(cached_data, f)

        # 메타데이터 업데이트
        if "chunk_configs" not in metadata:
            metadata["chunk_configs"] = {}
        metadata["chunk_configs"][chunk_key] = {
            "created": datetime.now().isoformat(),
            "chunks_count": len(all_chunks),
            "metadata_count": len(all_metadata)
        }
        self._save_metadata(metadata)

        print(f" 청크+메타데이터 캐시 저장 완료: {len(all_chunks)}개 청크, {len(all_metadata)}개 메타데이터")
        return all_chunks, all_metadata

    def get_guidelines_chunks(self, chunk_func, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """가이드라인 청크만 가져오기 (기존 호환성 유지)"""
        chunks, _ = self.get_guidelines_chunks_with_metadata(chunk_func, chunk_size, overlap)
        return chunks
    
    def get_chroma_collection(self,
                             collection_name: str,
                             data: List[str],
                             encode_func,
                             metadata_list: List[Dict] = None) -> Optional[chromadb.Collection]:
        """Chroma DB 컬렉션 가져오기 (필요시 새로 구축)"""
        
        # 데이터 해시 계산
        data_str = "\n".join(data[:5])  # 처음 5개만으로 해시 계산 (성능)
        current_hash = hashlib.md5(data_str.encode()).hexdigest()
        
        metadata = self._load_metadata()
        cached_hash = metadata.get(f"{collection_name}_hash", "")
        
        # 기존 컬렉션 확인
        try:
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            collection_exists = collection_name in existing_collections
            
            if collection_exists and cached_hash == current_hash:
                print(f" {collection_name} Chroma DB 컬렉션 캐시에서 로드")
                return self.chroma_client.get_collection(name=collection_name)
        except Exception as e:
            print(f" 기존 컬렉션 확인 실패: {e}")
        
        # 새로 구축
        print(f" {collection_name} Chroma DB 컬렉션 구축 중...")
        
        # 기존 컬렉션 삭제 (있다면)
        try:
            if collection_name in [col.name for col in self.chroma_client.list_collections()]:
                self.chroma_client.delete_collection(name=collection_name)
        except Exception as e:
            print(f" 기존 컬렉션 삭제 중 오류: {e}")
        
        # 새 컬렉션 생성
        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 환경에 맞는 배치 크기 설정
        batch_size = self.ec2_settings.get("batch_size", 50)
        print(f" 총 {len(data)}개 항목을 {batch_size}개씩 병렬 배치 처리")
        
        # 배치 데이터 준비 (메타데이터 포함)
        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size] if metadata_list else [{} for _ in batch_data]
            batches.append((i, batch_data, batch_metadata))

        # 병렬 처리로 임베딩 생성 및 저장
        self._process_batches_parallel_chroma(batches, encode_func, collection, collection_name)
        
        # 메타데이터 업데이트
        metadata[f"{collection_name}_hash"] = current_hash
        metadata[f"{collection_name}_updated"] = datetime.now().isoformat()
        metadata[f"{collection_name}_size"] = len(data)
        self._save_metadata(metadata)
        
        print(f" {collection_name} 컬렉션 구축 완료")
        return collection
    
    def _process_batches_parallel_chroma(self, batches, encode_func, collection, collection_name):
        """배치를 병렬로 처리하여 임베딩 생성 및 Chroma DB에 저장"""
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        def process_single_batch(batch_info):
            if len(batch_info) == 3:
                batch_idx, batch_data, batch_metadata = batch_info
            else:
                # 기존 호환성을 위해
                batch_idx, batch_data = batch_info
                batch_metadata = [{} for _ in batch_data]

            batch_num = batch_idx // 50 + 1
            total_batches = len(batches)
            try:
                start_time = time.time()
                embeddings = encode_func(batch_data)

                # Chroma DB에 배치 데이터 추가 (메타데이터 포함)
                ids = [f"{collection_name}_{batch_idx + i}" for i in range(len(batch_data))]
                collection.add(
                    embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                    documents=batch_data,
                    metadatas=batch_metadata,
                    ids=ids
                )

                end_time = time.time()
                print(f" 배치 {batch_num}/{total_batches} 완료 ({end_time - start_time:.1f}초, {len(embeddings)}개 임베딩, {len(batch_metadata)}개 메타데이터)")
                return True
            except Exception as e:
                print(f" 배치 {batch_num}/{total_batches} 실패: {e}")
                return False
        
        # 환경에 맞는 워커 수 설정
        max_workers = min(self.ec2_settings.get("max_workers", 3), len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f" 최대 {max_workers}개 배치 동시 처리 시작 ({len(batches)}개 배치)...")
            
            # 모든 배치 작업 제출
            future_to_batch = {
                executor.submit(process_single_batch, batch): batch 
                for batch in batches
            }
            
            # 결과 수집
            completed_count = 0
            success_count = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                result = future.result()
                completed_count += 1
                if result:
                    success_count += 1
                print(f" 진행상황: {completed_count}/{len(batches)} 배치 완료")
        
        print(f" 병렬 처리 완료: {success_count}/{len(batches)} 배치 성공")
    
    def query_similar(self, collection_name: str, query_text: str, encode_func, top_k: int = 5) -> List[Tuple[str, float]]:
        """Chroma DB에서 유사한 텍스트 검색"""
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            
            # 쿼리 임베딩 생성
            query_embedding = encode_func([query_text])
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            # 유사도 검색
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            # 결과 포맷팅
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # 거리를 유사도로 변환 (cosine distance -> cosine similarity)
            similarities = [(doc, 1.0 - dist) for doc, dist in zip(documents, distances)]
            
            return similarities
            
        except Exception as e:
            print(f" 유사도 검색 실패 ({collection_name}): {e}")
            return []
    
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
        """모든 캐시 및 Chroma DB 삭제"""
        import shutil
        
        # 파일 캐시 삭제
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        
        # Chroma DB 삭제
        if self.chroma_db_path.exists():
            shutil.rmtree(self.chroma_db_path)
            self.chroma_db_path.mkdir(exist_ok=True)
        
        # 클라이언트 재초기화
        self.chroma_client = PersistentClient(path=str(self.chroma_db_path))
        
        print(" 모든 캐시 및 Chroma DB가 삭제되었습니다.")
    
    def get_cache_info(self) -> Dict:
        """캐시 정보 조회"""
        metadata = self._load_metadata()
        
        cache_files = list(self.cache_dir.glob("*"))
        cache_size = sum(f.stat().st_size for f in cache_files if f.is_file())
        
        chroma_files = list(self.chroma_db_path.glob("**/*"))
        chroma_size = sum(f.stat().st_size for f in chroma_files if f.is_file())
        
        # Chroma DB 컬렉션 정보
        collections_info = {}
        try:
            collections = self.chroma_client.list_collections()
            for collection in collections:
                collections_info[collection.name] = {
                    "count": collection.count(),
                    "metadata": collection.metadata
                }
        except Exception as e:
            collections_info["error"] = str(e)
        
        return {
            "cache_dir": str(self.cache_dir),
            "chroma_db_path": str(self.chroma_db_path),
            "cache_files": len(cache_files),
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
            "chroma_size_mb": round(chroma_size / (1024 * 1024), 2),
            "total_size_mb": round((cache_size + chroma_size) / (1024 * 1024), 2),
            "metadata": metadata,
            "cached_items": {
                "predata": self.predata_cache.exists(),
                "guidelines": self.guidelines_cache.exists(),
                "templates": self.templates_cache.exists()
            },
            "chroma_collections": collections_info
        }

# 전역 인덱스 매니저 인스턴스
_index_manager = None

def get_index_manager() -> IndexManager:
    """싱글톤 인덱스 매니저 가져오기"""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager