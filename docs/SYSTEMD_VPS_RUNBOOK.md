# Systemd + VPS Runbook

This runbook covers:
1. Option 2: persistent `systemd` services on your machine.
2. Option 3: migrate to a VPS for 24/7 runtime.

## 1. Systemd User Services (Laptop or Linux Host)

### Prerequisites
- Repo checked out at a stable path.
- `.env` filled with required values:
  - `OANDA_ACCOUNT_ID`, `OANDA_API_TOKEN`, `INFLUXDB_TOKEN`
  - `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`
  - `DISCORD_EXEC_BOT_TOKEN` (optional fallback: `DISCORD_BOT_TOKEN`)
  - `DISCORD_EXEC_CHANNEL_ID` (optional fallback: `DISCORD_CHANNEL_ID`, default `1477609642258337954`)
- Runtime model JSON available under `data/research/`.

### Install Unit Files
```bash
cd /home/joe/Desktop/Algo_trading/oanda-trading-system
chmod +x scripts/install_systemd_user_services.sh
make systemd-user-install PROJECT_ROOT=/home/joe/Desktop/Algo_trading/oanda-trading-system
```

### Enable and Start
```bash
make systemd-user-enable
# optional: only if you want systemd to manage docker-compose infra
make systemd-user-enable-infra
make systemd-user-status
```

Services installed:
- `oanda-infra.service` (docker compose infra)
- `oanda-trading-supervisor.service`
- `oanda-discord-operator.service`
- `oanda-discord-execution-notifier.service`

If `oanda-infra.service` fails with port binding errors (for example `6379 already allocated`), keep infra service disabled and run infra manually or free the conflicting port/container first.

### Persist After Reboot
```bash
sudo loginctl enable-linger $USER
```

### View Logs
```bash
journalctl --user -u oanda-trading-supervisor.service -f
journalctl --user -u oanda-discord-operator.service -f
journalctl --user -u oanda-discord-execution-notifier.service -f
```

## 2. VPS Deployment (Recommended for Live)

### Baseline Host
- Ubuntu 22.04/24.04 LTS
- 2 vCPU, 4-8 GB RAM, 60+ GB SSD
- Static public IP

### Hardening
```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install git python3-venv python3-pip docker.io docker-compose-plugin fail2ban ufw
sudo usermod -aG docker $USER
newgrp docker
sudo ufw allow OpenSSH
sudo ufw enable
```

### Deploy App
```bash
git clone <your_repo_url> oanda-trading-system
cd oanda-trading-system
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env
# fill .env secrets
```

### Install/Enable Services
```bash
make systemd-user-install PROJECT_ROOT=$PWD
make systemd-user-enable
make systemd-user-enable-infra
sudo loginctl enable-linger $USER
```

### Operational Checks
```bash
make systemd-user-status
curl -s http://127.0.0.1:8010/health || true
```

### Journal Outputs
Execution journal files:
- `data/reports/trading_journal/executions_daily_YYYY-MM-DD.csv`
- `data/reports/trading_journal/executions_monthly_YYYY-MM.csv`
- `data/reports/trading_journal/summary_daily_YYYY-MM-DD.json`
- `data/reports/trading_journal/summary_monthly_YYYY-MM.json`

## Notes
- Supervisor auto-starts the `journal` agent as part of `!app start`.
- If `REGIME_MODEL_JSON` is unset, supervisor uses latest `multiframe_regime_model_*.json`.
- Rotate Discord token immediately if it has been exposed.
