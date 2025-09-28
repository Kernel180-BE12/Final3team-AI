import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from .index_manager import get_index_manager

class BaseTemplateProcessor:
    """템플릿 처리 기본 클래스"""
    
    def __init__(self, api_key: str, gemini_model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key
        
        # AI 모델 초기화
        self.gemini_model = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=api_key)
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        print(f" Gemini 모델 초기화: {gemini_model}")
        print(" Gemini Embedding API 사용 준비 완료")
        
        # 데이터 저장소
        self.templates = []
        self.guidelines = []
        
        # Chroma DB 컬렉션
        self.template_collection = None
        self.guideline_collection = None
        
        # 인덱스 매니저
        self.index_manager = get_index_manager()
        
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트를 Gemini Embedding으로 변환"""
        try:
            # Gemini Embedding API 사용
            embeddings = self.embeddings_model.embed_documents(texts)
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
    
    def search_similar_chroma(self, query: str, collection_name: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Chroma DB를 통한 유사도 검색"""
        return self.index_manager.query_similar(
            collection_name=collection_name,
            query_text=query,
            encode_func=self.encode_texts,
            top_k=top_k
        )
    
    def search_similar(self, query: str, collection_name: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """유사도 검색 (Chroma DB 사용)"""
        return self.search_similar_chroma(query, collection_name, top_k)
    
    def extract_variables(self, template: str) -> List[str]:
        """템플릿에서 #{변수명} 형태의 변수 추출"""
        pattern = r'#\{([^}]+)\}'
        variables = re.findall(pattern, template)
        return list(set(variables))  # 중복 제거
    
    def generate_with_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Gemini로 텍스트 생성"""
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.invoke(prompt)
                return response.content.strip()
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