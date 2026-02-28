# Operations Guide

## Daily Operations

### Morning Checklist

1. Check all agents are running:
   ```bash
   bash scripts/monitor_system.sh
   ```

2. Review overnight logs:
   ```bash
   grep -i error /tmp/*.log
   ```

3. Check system metrics:
   ```bash
   curl http://localhost:8000/metrics
   ```

4. Verify market data flowing:
   ```bash
   redis-cli XLEN stream:market_data
   ```

## Monitoring

### Key Metrics to Watch

- **Market Data**: Tick rate, data freshness
- **Signals**: Signal generation rate, confidence levels
- **Risk**: Approval rate, rejection reasons
- **Execution**: Fill rate, slippage
- **System**: CPU, memory, disk usage

### Alert Response

**Critical Alerts:**
- Circuit breaker triggered → Review positions, check logs
- Execution failures → Check Oanda API status
- Agent crashed → Restart agent, investigate logs

**Warning Alerts:**
- High rejection rate → Review risk parameters
- Stale data → Check market hours, Oanda connection
- High latency → Check system resources

## Manual Interventions

### Reset Circuit Breaker

```bash
redis-cli DEL circuit_breaker:state
```

### Stop All Agents

```bash
bash scripts/stop_all_agents.sh
```

### Restart Single Agent

```bash
kill $(cat /tmp/execution.pid)
python -m agents.execution.agent > /tmp/execution.log 2>&1 &
echo $! > /tmp/execution.pid
```

## Trading Hours

- **Forex Market**: Sunday 5pm ET - Friday 5pm ET
- **System Maintenance**: Saturday (markets closed)
- **Strategy Updates**: Deploy during low-volume periods

## Incident Response

1. **Identify**: Check alerts, logs, metrics
2. **Assess**: Determine severity and impact
3. **Contain**: Stop affected agents if needed
4. **Resolve**: Fix issue, restart services
5. **Document**: Record incident details
6. **Review**: Post-mortem analysis

## Backup Procedures

- **Daily**: Export Redis data
- **Weekly**: Backup InfluxDB data
- **Monthly**: Full system backup
