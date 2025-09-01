import re
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
from pathlib import Path

class BaseTemplateProcessor:
    """템플릿 처리 기본 클래스"""
    
    def __init__(self, api_key: str, gemini_model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key
        
        # AI 모델 초기화
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(gemini_model)
        
        print(f" Gemini 모델 초기화: {gemini_model}")
        print(" Gemini Embedding API 사용 준비 완료")
        
        # 데이터 저장소
        self.templates = []
        self.guidelines = []
        
        # FAISS 인덱스
        self.template_index = None
        self.guideline_index = None
        
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트를 Gemini Embedding으로 변환"""
        try:
            # Gemini Embedding API 사용
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            return np.array(embeddings)
        except Exception as e:
            print(f" Gemini Embedding 오류: {e}")
            # 폴백: 간단한 TF-IDF 기반 임베딩
            return self._fallback_embedding(texts)
    
    def _fallback_embedding(self, texts: List[str]) -> np.ndarray:
        """폴백 임베딩 (간단한 TF-IDF 기반)"""
        from collections import Counter
        import math
        
        try:
            # 모든 단어 수집 및 어휘 생성
            all_words = []
            for text in texts:
                words = text.lower().split()
                all_words.extend(words)
            
            word_counter = Counter(all_words)
            vocab = {word: idx for idx, (word, _) in enumerate(word_counter.most_common(1000))}
            
            # 간단한 TF-IDF 계산
            embeddings = []
            for text in texts:
                words = text.lower().split()
                word_counts = Counter(words)
                
                # TF-IDF 벡터 생성 (384차원으로 고정)
                vector = np.zeros(384)
                for word, count in word_counts.items():
                    if word in vocab and vocab[word] < 384:
                        tf = count / len(words)
                        idf = math.log(len(texts) / (1 + sum(1 for t in texts if word in t.lower().split())))
                        vector[vocab[word]] = tf * idf
                
                embeddings.append(vector)
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"폴백 임베딩 실패: {e}")
            return np.random.rand(len(texts), 384)
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """FAISS 인덱스 구축"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # 정규화
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        index.add(normalized_embeddings.astype('float32'))
        
        return index
    
    def search_similar(self, query: str, index: faiss.Index, texts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Gemini Embedding 기반 유사도 검색"""
        if index is None or not texts:
            return []
        
        try:
            # Gemini Embedding으로 쿼리 임베딩
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = np.array([result['embedding']])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            scores, indices = index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(texts):
                    results.append((texts[idx], float(score)))
            
            return results
        except Exception as e:
            print(f" 검색 오류: {e}")
            return []
    
    def extract_variables(self, template: str) -> List[str]:
        """템플릿에서 #{변수명} 형태의 변수 추출"""
        pattern = r'#\{([^}]+)\}'
        variables = re.findall(pattern, template)
        return list(set(variables))  # 중복 제거
    
    def generate_with_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Gemini로 텍스트 생성"""
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                continue
    
    def parse_json_response(self, response_text: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            # JSON 코드 블록 제거
            if '```json' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return {}
    
    def load_text_file(self, file_path: Path, encoding: str = 'utf-8') -> str:
        """텍스트 파일 로드"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            print(f"파일 로드 오류 {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        """텍스트를 청크로 분할"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]