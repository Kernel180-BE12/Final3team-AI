"""
통합된 데이터 처리 유틸리티
기존 htmlloader.py, txtloader.py, deleteImg.py 기능을 통합
"""

import re
import os
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import TextLoader

class DataProcessor:
    """데이터 로딩 및 전처리를 위한 통합 클래스"""

    def __init__(self, data_dir: str = "data", output_dir: str = "predata"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # 출력 디렉토리가 없으면 생성
        self.output_dir.mkdir(exist_ok=True)

    def load_markdown(self, filename: str, encoding: str = "utf-8") -> str:
        """마크다운 파일 로드"""
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        loader = TextLoader(str(file_path), encoding=encoding)
        docs = loader.load()

        print(f" {filename} 로드 완료 - 길이: {len(docs[0].page_content)}자")
        return docs[0].page_content

    def remove_images(self, text: str) -> str:
        """HTML 이미지 태그 제거"""
        # <figure>...</figure> 전체 블록 제거
        text = re.sub(r"<figure>.*?</figure>", "", text, flags=re.DOTALL)

        # 남은 <img> 태그들도 제거
        text = re.sub(r"<img[^>]*>", "", text)

        # 빈 줄 정리
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

        return text.strip()

    def clean_markdown(self, filename: str, save_cleaned: bool = True) -> str:
        """마크다운 파일 로드 및 이미지 제거"""
        print(f" {filename} 처리 중...")

        # 원본 로드
        original_content = self.load_markdown(filename)
        original_length = len(original_content)

        # 이미지 제거
        cleaned_content = self.remove_images(original_content)
        cleaned_length = len(cleaned_content)

        # 정리된 파일 저장 (predata 폴더에)
        if save_cleaned:
            name_part = Path(filename).stem
            ext_part = Path(filename).suffix
            cleaned_filename = f"cleaned_{name_part}{ext_part}"
            cleaned_path = self.output_dir / cleaned_filename

            with open(cleaned_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            print(f" 정리된 파일 저장: {cleaned_path}")

        return cleaned_content

    def list_markdown_files(self) -> List[str]:
        """데이터 디렉토리의 마크다운 파일 목록"""
        if not self.data_dir.exists():
            print(f" 데이터 디렉토리가 존재하지 않습니다: {self.data_dir}")
            return []

        md_files = list(self.data_dir.glob("*.md"))
        return [f.name for f in md_files]

    def process_all_markdown(self) -> None:
        """모든 마크다운 파일 처리"""
        md_files = self.list_markdown_files()

        if not md_files:
            print(" 처리할 마크다운 파일이 없습니다.")
            return

        print(f" 발견된 마크다운 파일: {len(md_files)}개")

        for filename in md_files:
            if not filename.startswith("cleaned_"):  # 이미 처리된 파일 제외
                try:
                    self.clean_markdown(filename)
                    print()
                except Exception as e:
                    print(f" {filename} 처리 실패: {e}")

        print(" 모든 파일 처리 완료!")


if __name__ == "__main__":
    # DataProcessor 인스턴스 생성 (data 폴더에서 읽어서 predata 폴더에 저장)
    processor = DataProcessor(data_dir="data", output_dir="predata")
    
    print(" 데이터 처리를 시작합니다...")
    print(f"📂 입력 폴더: {processor.data_dir}")
    print(f" 출력 폴더: {processor.output_dir}")
    print()
    
    # 모든 마크다운 파일 처리
    processor.process_all_markdown()
