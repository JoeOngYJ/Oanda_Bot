"""
Monitoring Agent - Provides real-time observability and alerting.
Operates independently without interfering with trading logic.
"""

import asyncio
import logging
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from typing import Dict, List
from shared.message_bus import MessageBus
from shared.models import HealthMetric
from shared.config import Config
from .metrics_collector import MetricsCollector
from .alerting import AlertManager
from .health_checker import HealthChecker

logger = logging.getLogger(__name__)


class MonitoringAgent:
    """
    Continuously monitors system health, performance, and security.
    Operates independently without interfering with trading logic.
    """

    def __init__(self, config: Config, message_bus: MessageBus):
        self.config = config
        self.message_bus = message_bus

        # Prometheus metrics
        self.market_data_latency = Histogram(
            'market_data_latency_seconds',
            'Market data feed latency'
        )
        self.order_execution_time = Histogram(
            'order_execution_seconds',
            'Time from order submission to fill'
        )
        self.active_positions = Gauge(
            'active_positions',
            'Number of open positions'
        )
        self.daily_pnl = Gauge(
            'daily_pnl_usd',
            'Daily profit/loss in USD'
        )
        self.error_count = Counter(
            'errors_total',
            'Total errors by component',
            ['component', 'error_type']
        )
        self.component_health = Gauge(
            'component_health',
            'Component health status (1=healthy, 0=unhealthy)',
            ['component']
        )
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
        self.execution_kill_switch = Gauge(
            'execution_kill_switch_active',
            'Execution kill switch state (1=active,0=inactive)'
        )
        self.execution_shadow_mode = Gauge(
            'execution_shadow_mode_active',
            'Execution shadow mode state (1=active,0=inactive)'
        )

        # Components
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config)
        self.health_checker = HealthChecker(config)

        self.running = False

    async def start(self):
        """Start monitoring agent"""
        logger.info("Starting Monitoring Agent")
        self.running = True

        # Start Prometheus HTTP server
        try:
            start_http_server(self.config.monitoring.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.config.monitoring.prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._monitor_health()),
            asyncio.create_task(self._collect_metrics()),
            asyncio.create_task(self._process_alerts()),
            asyncio.create_task(self._monitor_message_streams()),
            asyncio.create_task(self._monitor_execution_controls()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Monitoring agent error: {e}", exc_info=True)
            await self.stop()

    async def _monitor_health(self):
        """Continuously check component health"""
        while self.running:
            try:
                health_checks = await self.health_checker.check_all_components()

                for check in health_checks:
                    # Update Prometheus gauge
                    health_value = 1.0 if check.status == "healthy" else 0.0
                    self.component_health.labels(component=check.component).set(health_value)

                    # Send alerts for issues
                    if check.status == "critical":
                        await self.alert_manager.send_alert(
                            severity="critical",
                            component=check.component,
                            message=f"Health check failed: {check.metric_name}",
                            value=check.value,
                            metadata=check.metadata
                        )
                    elif check.status == "warning":
                        await self.alert_manager.send_alert(
                            severity="warning",
                            component=check.component,
                            message=f"Health check warning: {check.metric_name}",
                            value=check.value
                        )

            except Exception as e:
                logger.error(f"Health monitoring error: {e}", exc_info=True)
                self.error_count.labels(
                    component='monitoring',
                    error_type='health_check'
                ).inc()

            await asyncio.sleep(self.config.monitoring.health_check_interval)

    async def _collect_metrics(self):
        """Collect and aggregate system metrics"""
        while self.running:
            try:
                metrics = await self.metrics_collector.collect()

                # Update Prometheus metrics
                for metric in metrics:
                    if metric.metric_name == "cpu_usage":
                        self.cpu_usage.set(metric.value)
                    elif metric.metric_name == "memory_usage":
                        self.memory_usage.set(metric.value)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}", exc_info=True)

            await asyncio.sleep(self.config.monitoring.collection_interval)

    async def _process_alerts(self):
        """Process alert stream"""
        try:
            async for message in self.message_bus.subscribe('alerts'):
                if not self.running:
                    break
                try:
                    await self.alert_manager.process_alert(message)
                except Exception as e:
                    logger.error(f"Alert processing error: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Alert stream error: {e}", exc_info=True)

    async def _monitor_message_streams(self):
        """Monitor Redis stream backlogs and throughput"""
        while self.running:
            try:
                stream_stats = await self.message_bus.get_stream_stats()

                for stream_name, stats in stream_stats.items():
                    if 'lag' in stats and stats['lag'] > 1000:
                        await self.alert_manager.send_alert(
                            severity="warning",
                            component="message_bus",
                            message=f"Stream backlog detected: {stream_name}",
                            value=stats['lag']
                        )

            except Exception as e:
                logger.error(f"Stream monitoring error: {e}", exc_info=True)

            await asyncio.sleep(10)

    async def _monitor_execution_controls(self):
        """Track execution control state from control commands and alerts."""
        # Initialize conservative defaults.
        self.execution_kill_switch.set(0.0)
        self.execution_shadow_mode.set(0.0)

        # Track explicit control commands.
        async for message in self.message_bus.subscribe('execution_control'):
            if not self.running:
                break
            action = str(message.get("action", ""))
            if action == "kill_switch_on":
                self.execution_kill_switch.set(1.0)
            elif action == "kill_switch_off":
                self.execution_kill_switch.set(0.0)
            elif action == "shadow_mode_on":
                self.execution_shadow_mode.set(1.0)
            elif action == "shadow_mode_off":
                self.execution_shadow_mode.set(0.0)

    async def stop(self):
        """Gracefully stop monitoring agent"""
        logger.info("Stopping Monitoring Agent")
        self.running = False


async def main():
    """Main entry point for Monitoring Agent"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()

    agent = MonitoringAgent(config, message_bus)

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await agent.stop()
        await message_bus.disconnect()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
