"""
Alert manager for routing and delivering notifications.
Handles deduplication and severity-based routing.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from shared.config import Config

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alert routing and delivery"""

    def __init__(self, config: Config):
        self.config = config
        self.last_alert_time: Dict[str, datetime] = {}
        self.deduplication_interval = 300  # 5 minutes

    async def send_alert(
        self,
        severity: str,
        component: str,
        message: str,
        value: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Send alert to configured channels.

        Args:
            severity: Alert severity (info, warning, critical)
            component: Component generating the alert
            message: Alert message
            value: Optional metric value
            metadata: Optional additional context
        """
        # Create alert key for deduplication
        alert_key = f"{component}:{message}"

        # Check if we should deduplicate
        if self._should_deduplicate(alert_key):
            logger.debug(f"Deduplicated alert: {alert_key}")
            return

        # Update last alert time
        self.last_alert_time[alert_key] = datetime.utcnow()

        # Build alert payload
        alert = {
            "severity": severity,
            "component": component,
            "message": message,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        # Route to appropriate channels
        await self._route_alert(alert)

    async def process_alert(self, message: dict) -> None:
        """Process alert from stream:alerts"""
        try:
            severity = message.get("severity", "info")
            component = message.get("component", "unknown")
            alert_message = message.get("message", "")
            value = message.get("value")
            metadata = message.get("metadata", {})

            await self.send_alert(
                severity=severity,
                component=component,
                message=alert_message,
                value=value,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error processing alert: {e}", exc_info=True)

    async def _route_alert(self, alert: dict) -> None:
        """Route alert to configured channels"""
        severity = alert["severity"]

        # Always log
        await self._send_to_log(alert)

        # Send to email for critical alerts
        if severity == "critical":
            await self._send_email(alert)

    async def _send_to_log(self, alert: dict) -> None:
        """Write alert to logs"""
        severity = alert["severity"]
        component = alert["component"]
        message = alert["message"]
        value = alert.get("value")

        log_message = f"{component}: {message}"
        if value is not None:
            log_message += f" (value: {value})"

        if severity == "critical":
            logger.critical(log_message, extra={"alert": alert})
        elif severity == "warning":
            logger.warning(log_message, extra={"alert": alert})
        else:
            logger.info(log_message, extra={"alert": alert})

    async def _send_email(self, alert: dict) -> None:
        """Send email notification"""
        # Email sending would be implemented here
        # For now, just log that we would send an email
        logger.info(
            f"Would send email alert: {alert['component']} - {alert['message']}",
            extra={"alert": alert}
        )

    def _should_deduplicate(self, alert_key: str) -> bool:
        """Check if alert should be deduplicated"""
        if alert_key not in self.last_alert_time:
            return False

        last_time = self.last_alert_time[alert_key]
        time_since_last = (datetime.utcnow() - last_time).total_seconds()

        return time_since_last < self.deduplication_interval
