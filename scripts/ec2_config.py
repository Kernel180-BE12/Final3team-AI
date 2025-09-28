#!/usr/bin/env python3
"""
EC2 환경 설정
"""
import os
from pathlib import Path

# EC2 환경 감지
def is_ec2_environment():
    """EC2 환경인지 확인"""
    try:
        # EC2 메타데이터 서비스 확인
        import urllib.request
        urllib.request.urlopen('http://169.254.169.254/latest/meta-data/', timeout=2)
        return True
    except:
        return False

# 환경별 경로 설정
def get_environment_paths():
    """환경에 맞는 경로 반환"""
    if is_ec2_environment():
        # EC2 환경
        base_path = Path("/home/ec2-user")
        return {
            "cache_dir": str(base_path / "app_cache"),
            "chroma_db_path": str(base_path / "chroma_data"),
            "predata_path": str(base_path / "Jober_ai" / "predata"),
            "log_path": str(base_path / "logs")
        }
    else:
        # 로컬 개발 환경
        return {
            "cache_dir": "cache",
            "chroma_db_path": "chroma_db", 
            "predata_path": "predata",
            "log_path": "logs"
        }

# EC2 전용 설정
EC2_SETTINGS = {
    # Chroma DB 설정
    "chroma_collection_settings": {
        "hnsw_space": "cosine",
        "allow_reset": True
    },
    
    # 성능 최적화
    "batch_size": 100,  # EC2에서는 더 큰 배치 사이즈
    "max_workers": 4,   # EC2 인스턴스에 맞게 조정
    
    # 메모리 관리
    "max_cache_size_mb": 512,
    "enable_disk_cache": True,
    
    # 로깅
    "log_level": "INFO",
    "enable_performance_logging": True
}

def setup_ec2_directories():
    """EC2에서 필요한 디렉토리 생성"""
    if not is_ec2_environment():
        return
    
    paths = get_environment_paths()
    
    for path_name, path_value in paths.items():
        try:
            Path(path_value).mkdir(parents=True, exist_ok=True)
            os.chmod(path_value, 0o755)
            print(f" {path_name}: {path_value}")
        except Exception as e:
            print(f"  {path_name} 생성 실패: {e}")

def get_ec2_optimized_settings():
    """EC2 최적화된 설정 반환"""
    return EC2_SETTINGS

# 환경 정보 출력
def print_environment_info():
    """현재 환경 정보 출력"""
    is_ec2 = is_ec2_environment()
    paths = get_environment_paths()
    
    print(f" 환경: {'EC2' if is_ec2 else '로컬'}")
    print(f" 캐시 디렉토리: {paths['cache_dir']}")
    print(f"Chroma DB: {paths['chroma_db_path']}")
    print(f" Predata: {paths['predata_path']}")
    
    if is_ec2:
        settings = get_ec2_optimized_settings()
        print(f" 배치 크기: {settings['batch_size']}")
        print(f" 워커 수: {settings['max_workers']}")

if __name__ == "__main__":
    print("=== EC2 환경 설정 테스트 ===")
    print_environment_info()
    print("\n=== 디렉토리 설정 ===")
    setup_ec2_directories()