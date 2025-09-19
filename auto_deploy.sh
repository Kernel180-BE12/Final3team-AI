#!/bin/bash
# EC2 자동 배포 스크립트 - Git Clone & Auto Deploy

set -e  # 에러 발생시 스크립트 중단

echo "🚀 Jober AI 자동 배포 시작"

# 설정 변수
REPO_URL="https://github.com/david1-p/Jober_ai.git"
APP_DIR="/home/ec2-user/Jober_ai"
SERVICE_NAME="jober-ai"

# 1. 시스템 패키지 설치
echo "📦 시스템 패키지 업데이트 및 설치..."
sudo yum update -y
sudo yum install -y git curl gcc g++ python3.11 python3.11-pip python3.11-devel

# 2. 기존 서비스 중지
echo "⏹️ 기존 서비스 중지..."
sudo systemctl stop $SERVICE_NAME.service 2>/dev/null || true

# 3. 기존 코드 제거 및 새로 클론
echo "🔄 코드 업데이트..."
sudo rm -rf $APP_DIR
cd /home/ec2-user
git clone $REPO_URL Jober_ai

# 4. 디렉토리 권한 설정
echo "🔐 권한 설정..."
sudo chown -R ec2-user:ec2-user $APP_DIR
cd $APP_DIR

# 5. Poetry 설치
echo "📝 Poetry 설치..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3.11 -
fi
export PATH="/home/ec2-user/.local/bin:$PATH"

# 6. 의존성 설치
echo "📚 Python 패키지 설치..."
poetry install --only=main

# 7. 환경 변수 설정 확인
echo "⚙️ 환경 설정 확인..."
if [ ! -f ".env" ]; then
    echo "❌ .env 파일이 없습니다. 수동으로 생성해주세요."
    cp .env.sample .env || true
fi

# 8. Systemd 서비스 파일 생성
echo "🔧 서비스 설정..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Jober AI Template Service
After=network.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=$APP_DIR
Environment=PATH=/home/ec2-user/.local/bin:/usr/bin:/bin
Environment=PYTHONPATH=$APP_DIR
ExecStart=/home/ec2-user/.local/bin/poetry run python server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 9. 서비스 시작
echo "🚀 서비스 시작..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME.service
sudo systemctl start $SERVICE_NAME.service

# 10. 배포 결과 확인
echo "✅ 배포 완료! 서비스 상태:"
sudo systemctl status $SERVICE_NAME.service --no-pager -l

echo ""
echo "📋 유용한 명령어:"
echo "  서비스 상태: sudo systemctl status $SERVICE_NAME"
echo "  로그 확인: sudo journalctl -u $SERVICE_NAME -f"
echo "  서비스 재시작: sudo systemctl restart $SERVICE_NAME"
echo "  재배포: bash auto_deploy.sh"

# 11. 헬스체크
echo ""
echo "🔍 헬스체크 (30초 후)..."
sleep 30
if curl -f http://localhost:8000/ 2>/dev/null; then
    echo "✅ 서버가 정상적으로 실행되고 있습니다!"
else
    echo "⚠️ 서버 상태를 확인해주세요."
    echo "로그 확인: sudo journalctl -u $SERVICE_NAME -n 50"
fi