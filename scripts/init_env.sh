#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

if [[ -f "${ENV_FILE}" && "${FORCE}" -ne 1 ]]; then
  echo ".env already exists at ${ENV_FILE}"
  echo "Use --force to overwrite."
  exit 0
fi

cat > "${ENV_FILE}" <<'EOF'
# OANDA credentials
OANDA_API_TOKEN=
OANDA_ACCOUNT_ID=
OANDA_ENV=practice

# Data defaults
DATA_DIR=data/backtesting
INSTRUMENT=XAU_USD
TF=M15

# Optional: align with downloader fallback
TRADING_ENVIRONMENT=practice
EOF

echo "Created ${ENV_FILE}"
echo "Next:"
echo "  1) Edit ${ENV_FILE} and set OANDA_API_TOKEN / OANDA_ACCOUNT_ID"
echo "  2) Load env in current shell:"
echo "       set -a; source .env; set +a"
