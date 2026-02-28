# Deployment Guide

## Prerequisites

- Python 3.11+
- Redis 7.0+
- InfluxDB 2.0+
- Oanda Practice or Live Account
- Linux/macOS environment

## Installation Steps

### 1. Clone Repository

```bash
git clone <repository-url>
cd oanda-trading-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file:

```bash
# Oanda Configuration
OANDA_API_TOKEN=your_api_token_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice  # or 'live'

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# InfluxDB Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=your_org
INFLUXDB_BUCKET=trading_data
```

### 4. Start Services

```bash
# Start Redis
redis-server

# Start InfluxDB
influxd
```

### 5. Start Agents

```bash
# Start all agents
python -m agents.market_data.agent > /tmp/market_data.log 2>&1 &
echo $! > /tmp/market_data.pid

python -m agents.monitoring.agent > /tmp/monitoring.log 2>&1 &
echo $! > /tmp/monitoring.pid

python -m agents.strategy.agent > /tmp/strategy.log 2>&1 &
echo $! > /tmp/strategy.pid

python -m agents.risk.agent > /tmp/risk.log 2>&1 &
echo $! > /tmp/risk.pid

python -m agents.execution.agent > /tmp/execution.log 2>&1 &
echo $! > /tmp/execution.pid
```

### 6. Verify System

```bash
# Check agent status
bash scripts/monitor_system.sh

# Check logs
tail -f /tmp/*.log
```

## Production Deployment

For production deployment:

1. Use systemd or supervisor for process management
2. Configure log rotation
3. Set up monitoring and alerting
4. Use production Oanda account
5. Implement backup procedures
6. Configure firewall rules
7. Use TLS for Redis/InfluxDB if networked

## Security Checklist

- [ ] API tokens stored in environment variables
- [ ] .env file in .gitignore
- [ ] Redis authentication enabled
- [ ] InfluxDB authentication enabled
- [ ] Firewall configured
- [ ] Log files secured
- [ ] Regular security updates applied
