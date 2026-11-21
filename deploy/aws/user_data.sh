#!/usr/bin/env bash
# ============================================================================
# EC2 User Data Bootstrap Script (Amazon Linux 2023)
# Real-Time Fraud Detection & Risk Scoring Platform
# ============================================================================
# Purpose:
#   - Installs Docker, Docker Compose plugin, AWS CLI, and jq.
#   - Prepares /opt/fraud-detection directory with restricted permissions.
#   - Does NOT contain any hardcoded AWS keys or database passwords.
# ============================================================================

set -euo pipefail

exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

echo "[1/4] Updating system packages..."
dnf update -y
dnf install -y docker jq git

echo "[2/4] Enabling and starting Docker daemon..."
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user || true

echo "[3/4] Installing Docker Compose v2 plugin..."
DOCKER_CONFIG="${DOCKER_CONFIG:-/usr/local/lib/docker}"
mkdir -p "${DOCKER_CONFIG}/cli-plugins"
COMPOSE_VERSION="v2.27.0"
curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
  -o "${DOCKER_CONFIG}/cli-plugins/docker-compose"
chmod +x "${DOCKER_CONFIG}/cli-plugins/docker-compose"

echo "[4/4] Creating application runtime directories..."
mkdir -p /opt/fraud-detection/artifacts/models \
         /opt/fraud-detection/artifacts/plots \
         /opt/fraud-detection/artifacts/reports
chown -R ec2-user:ec2-user /opt/fraud-detection
chmod 750 /opt/fraud-detection

cat <<'EOF' > /opt/fraud-detection/.env.template
# Populate these values securely (e.g., from AWS SSM Parameter Store) into /opt/fraud-detection/.env
POSTGRES_HOST=<PRIVATE_RDS_ENDPOINT_OR_LOCAL_DB>
POSTGRES_PORT=5432
POSTGRES_DB=fraud_detection
POSTGRES_USER=fraud_user
POSTGRES_PASSWORD=<FETCH_FROM_SSM_OR_SECRETS_MANAGER>
APP_ENV=production
LOG_LEVEL=INFO
EOF
chmod 600 /opt/fraud-detection/.env.template

echo "Bootstrap complete. Follow deploy/aws/deployment_guide.md to pull the ECR image and start the API."
