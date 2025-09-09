#!/bin/bash
# EC2 배포 스크립트

echo "🚀 EC2에 Jober_AI (Chroma DB) 배포 시작"

# 1. 시스템 업데이트
echo "📦 시스템 패키지 업데이트..."
sudo yum update -y

# 2. Python 3.11 설치 (Amazon Linux 2023)
echo "🐍 Python 3.11 설치..."
sudo yum install -y python3.11 python3.11-pip

# 3. 애플리케이션 디렉토리 생성
echo "📁 애플리케이션 디렉토리 설정..."
mkdir -p /home/ec2-user/Jober_ai
mkdir -p /home/ec2-user/app_cache
mkdir -p /home/ec2-user/chroma_data
mkdir -p /home/ec2-user/logs

# 권한 설정
chmod 755 /home/ec2-user/app_cache
chmod 755 /home/ec2-user/chroma_data
chmod 755 /home/ec2-user/logs

# 4. 애플리케이션 코드 업로드 (Git 사용 권장)
cd /home/ec2-user
echo "📥 코드 다운로드..."
# git clone [your-repo-url] Jober_ai
# cd Jober_ai

# 5. Python 의존성 설치
echo "📚 Python 패키지 설치..."
python3.11 -m pip install --user -r requirements.txt

# 6. 환경 설정 확인
echo "🔧 환경 설정 확인..."
python3.11 ec2_config.py

# 7. Chroma DB 초기화 테스트
echo "🗄️  Chroma DB 테스트..."
python3.11 test_chroma_simple.py

# 8. Systemd 서비스 파일 생성
echo "⚙️  서비스 설정..."
sudo tee /etc/systemd/system/jober-ai.service > /dev/null <<EOF
[Unit]
Description=Jober AI Template Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/Jober_ai
Environment=PATH=/home/ec2-user/.local/bin:/usr/bin:/bin
Environment=PYTHONPATH=/home/ec2-user/Jober_ai
ExecStart=/usr/bin/python3.11 server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 9. 서비스 활성화
echo "🔥 서비스 시작..."
sudo systemctl daemon-reload
sudo systemctl enable jober-ai.service
sudo systemctl start jober-ai.service

# 10. 상태 확인
echo "✅ 배포 완료! 서비스 상태 확인:"
sudo systemctl status jober-ai.service

echo ""
echo "📋 유용한 명령어:"
echo "  서비스 상태: sudo systemctl status jober-ai"
echo "  로그 확인: sudo journalctl -u jober-ai -f"
echo "  서비스 재시작: sudo systemctl restart jober-ai"
echo "  서비스 중지: sudo systemctl stop jober-ai"