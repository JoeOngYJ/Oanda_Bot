# Troubleshooting Guide

## Common Issues

### Agent Won't Start

**Symptoms:** Agent exits immediately after starting

**Causes:**
- Missing environment variables
- Redis/InfluxDB not running
- Port conflicts

**Solutions:**
```bash
# Check environment variables
env | grep OANDA

# Check Redis
redis-cli ping

# Check InfluxDB
curl http://localhost:8086/health

# Check logs
tail -f /tmp/agent_name.log
```

### No Market Data

**Symptoms:** No ticks in stream:market_data

**Causes:**
- Markets closed (weekends)
- Oanda API connection issue
- Invalid API credentials

**Solutions:**
```bash
# Check market hours (Forex: Sun 5pm - Fri 5pm ET)
date

# Test Oanda connection
python -c "from agents.market_data.oanda_client import OandaStreamClient; from shared.config import Config; client = OandaStreamClient(Config.load()); print('OK')"

# Check API credentials
echo $OANDA_API_TOKEN
```

### Signals Not Generated

**Symptoms:** No signals in stream:signals

**Causes:**
- Insufficient tick data for indicators
- Strategy conditions not met
- Strategy agent not running

**Solutions:**
```bash
# Check strategy agent
ps -p $(cat /tmp/strategy.pid)

# Check tick count
redis-cli XLEN stream:market_data

# Check strategy logs
grep -i signal /tmp/strategy.log
```

### Orders Rejected

**Symptoms:** Risk agent rejecting all signals

**Causes:**
- Circuit breaker triggered
- Risk limits exceeded
- Insufficient account balance

**Solutions:**
```bash
# Check circuit breaker status
redis-cli GET circuit_breaker:state

# Check risk logs
grep -i reject /tmp/risk.log

# Check account balance (via Oanda dashboard)
```

### High CPU/Memory Usage

**Symptoms:** System slow, high resource usage

**Causes:**
- Too many ticks being processed
- Memory leak
- Inefficient strategy

**Solutions:**
```bash
# Check resource usage
top -p $(cat /tmp/*.pid | tr '\n' ',')

# Restart agents
bash scripts/stop_all_agents.sh
# Then restart

# Check for memory leaks in logs
grep -i memory /tmp/*.log
```

### Redis Connection Issues

**Symptoms:** "Connection refused" errors

**Causes:**
- Redis not running
- Wrong host/port
- Network issues

**Solutions:**
```bash
# Start Redis
redis-server

# Check Redis status
redis-cli ping

# Check Redis config
grep REDIS .env
```

### Stale Tick Warnings

**Symptoms:** "Stale tick" warnings in logs

**Causes:**
- Markets closed (weekends)
- Oanda API delay
- System clock skew

**Solutions:**
```bash
# Check if markets are open
# Forex: Sunday 5pm ET - Friday 5pm ET

# Check system time
date

# This is normal on weekends - ignore
```

## Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m agents.agent_name.agent
```

## Getting Help

1. Check logs: `/tmp/*.log`
2. Check system status: `bash scripts/monitor_system.sh`
3. Review this guide
4. Check Oanda API status
5. Contact support with logs and error messages
