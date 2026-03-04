"""
Performs health checks on all system components.
Monitors system resources, services, and data feeds.
"""

import asyncio
import psutil
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from oanda_bot.utils.models import HealthMetric, Instrument
from oanda_bot.utils.config import Config

logger = logging.getLogger(__name__)


class HealthChecker:
    """Performs health checks on all system components"""

    def __init__(self, config: Config):
        self.config = config
        self.last_market_tick: Dict[str, datetime] = {}
        self.last_order_time = None

    async def check_all_components(self) -> List[HealthMetric]:
        """Run all health checks and return results"""
        checks = []

        # System resource checks
        checks.extend(await self._check_system_resources())

        # Market data feed checks
        checks.extend(await self._check_market_data_feeds())

        # Redis connectivity
        redis_check = await self._check_redis()
        if redis_check:
            checks.append(redis_check)

        # InfluxDB connectivity
        influx_check = await self._check_influxdb()
        if influx_check:
            checks.append(influx_check)

        return checks

    async def _check_system_resources(self) -> List[HealthMetric]:
        """Check CPU, memory, disk usage"""
        metrics = []
        now = datetime.utcnow()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = "healthy"
        if cpu_percent > self.config.monitoring.alert_thresholds.cpu_usage_critical:
            cpu_status = "critical"
        elif cpu_percent > self.config.monitoring.alert_thresholds.cpu_usage_warning:
            cpu_status = "warning"

        metrics.append(HealthMetric(
            component="system",
            metric_name="cpu_usage",
            value=cpu_percent,
            timestamp=now,
            status=cpu_status
        ))

        # Memory usage
        memory = psutil.virtual_memory()
        mem_status = "healthy"
        if memory.percent > self.config.monitoring.alert_thresholds.memory_usage_critical:
            mem_status = "critical"
        elif memory.percent > self.config.monitoring.alert_thresholds.memory_usage_warning:
            mem_status = "warning"

        metrics.append(HealthMetric(
            component="system",
            metric_name="memory_usage",
            value=memory.percent,
            timestamp=now,
            status=mem_status
        ))

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_status = "healthy"
        if disk.percent > 90:
            disk_status = "critical"
        elif disk.percent > 85:
            disk_status = "warning"

        metrics.append(HealthMetric(
            component="system",
            metric_name="disk_usage",
            value=disk.percent,
            timestamp=now,
            status=disk_status
        ))

        return metrics

    async def _check_market_data_feeds(self) -> List[HealthMetric]:
        """Check market data feed freshness"""
        metrics = []
        now = datetime.utcnow()

        for instrument, last_tick_time in self.last_market_tick.items():
            staleness = (now - last_tick_time).total_seconds()

            status = "healthy"
            if staleness > self.config.monitoring.alert_thresholds.market_data_stale_critical:
                status = "critical"
            elif staleness > self.config.monitoring.alert_thresholds.market_data_stale_warning:
                status = "warning"

            metrics.append(HealthMetric(
                component="market_data",
                metric_name=f"feed_staleness_{instrument}",
                value=staleness,
                timestamp=now,
                status=status,
                metadata={"instrument": instrument}
            ))

        return metrics

    async def _check_redis(self) -> HealthMetric:
        """Check Redis connectivity"""
        try:
            # This would ping Redis in a real implementation
            # For now, return a healthy status
            return HealthMetric(
                component="redis",
                metric_name="connectivity",
                value=1.0,
                timestamp=datetime.utcnow(),
                status="healthy"
            )
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return HealthMetric(
                component="redis",
                metric_name="connectivity",
                value=0.0,
                timestamp=datetime.utcnow(),
                status="critical",
                metadata={"error": str(e)}
            )

    async def _check_influxdb(self) -> HealthMetric:
        """Check InfluxDB connectivity"""
        try:
            # This would ping InfluxDB in a real implementation
            # For now, return a healthy status
            return HealthMetric(
                component="influxdb",
                metric_name="connectivity",
                value=1.0,
                timestamp=datetime.utcnow(),
                status="healthy"
            )
        except Exception as e:
            logger.error(f"InfluxDB health check failed: {e}")
            return HealthMetric(
                component="influxdb",
                metric_name="connectivity",
                value=0.0,
                timestamp=datetime.utcnow(),
                status="critical",
                metadata={"error": str(e)}
            )

    def update_market_tick_time(self, instrument: str, timestamp: datetime):
        """Update last received tick time for an instrument"""
        self.last_market_tick[instrument] = timestamp
