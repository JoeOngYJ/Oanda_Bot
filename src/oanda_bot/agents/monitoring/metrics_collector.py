"""
Collects system and application metrics for monitoring.
Provides data for Prometheus and alerting.
"""

import psutil
import logging
from datetime import datetime
from typing import List
from oanda_bot.utils.models import HealthMetric
from oanda_bot.utils.config import Config

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects system and application metrics"""

    def __init__(self, config: Config):
        self.config = config

    async def collect(self) -> List[HealthMetric]:
        """Collect all metrics and return list"""
        metrics = []

        # System metrics
        metrics.extend(await self._collect_system_metrics())

        # Market data metrics (would be populated by market data agent)
        metrics.extend(await self._collect_market_data_metrics())

        # Trading metrics (would be populated by other agents)
        metrics.extend(await self._collect_trading_metrics())

        return metrics

    async def _collect_system_metrics(self) -> List[HealthMetric]:
        """CPU, memory, disk, network"""
        metrics = []
        now = datetime.utcnow()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(HealthMetric(
            component="system",
            metric_name="cpu_usage",
            value=cpu_percent,
            timestamp=now,
            status="healthy"
        ))

        # Memory usage
        memory = psutil.virtual_memory()
        metrics.append(HealthMetric(
            component="system",
            metric_name="memory_usage",
            value=memory.percent,
            timestamp=now,
            status="healthy"
        ))

        # Disk usage
        disk = psutil.disk_usage('/')
        metrics.append(HealthMetric(
            component="system",
            metric_name="disk_usage",
            value=disk.percent,
            timestamp=now,
            status="healthy"
        ))

        # Network I/O
        net_io = psutil.net_io_counters()
        metrics.append(HealthMetric(
            component="system",
            metric_name="network_bytes_sent",
            value=float(net_io.bytes_sent),
            timestamp=now,
            status="healthy"
        ))
        metrics.append(HealthMetric(
            component="system",
            metric_name="network_bytes_recv",
            value=float(net_io.bytes_recv),
            timestamp=now,
            status="healthy"
        ))

        return metrics

    async def _collect_market_data_metrics(self) -> List[HealthMetric]:
        """Feed latency, staleness, gaps"""
        metrics = []
        # This would be populated by market data agent
        # For now, return empty list
        return metrics

    async def _collect_trading_metrics(self) -> List[HealthMetric]:
        """Positions, P&L, order stats"""
        metrics = []
        # This would be populated by trading agents
        # For now, return empty list
        return metrics
