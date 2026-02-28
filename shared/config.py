"""
Configuration loading system with YAML files and environment variable support.
Uses Pydantic for validation and type safety.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import re
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class RedisConfig(BaseModel):
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    streams: Dict[str, str] = Field(default_factory=dict)


class InfluxDBConfig(BaseModel):
    """InfluxDB configuration"""
    url: str = "http://localhost:8086"
    token: str
    org: str = "trading_org"
    bucket: str = "market_data"


class PrometheusConfig(BaseModel):
    """Prometheus configuration"""
    port: int = 8000
    push_gateway: str = "localhost:9091"


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    elasticsearch: Optional[Dict[str, Any]] = None


class SystemConfig(BaseModel):
    """System-wide configuration"""
    environment: str = "production"
    timezone: str = "UTC"


class OandaEndpoints(BaseModel):
    """Oanda API endpoints"""
    api: str
    stream: str


class OandaConfig(BaseModel):
    """Oanda API configuration"""
    account_id: str
    api_token: str
    environment: str = "practice"
    endpoints: Dict[str, OandaEndpoints]
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0
    instruments: list[str] = Field(default_factory=list)
    requests_per_second: int = 100


class PerInstrumentLimits(BaseModel):
    """Per-instrument risk limits"""
    max_position_size: int
    max_order_size: int


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration"""
    consecutive_losses: int
    loss_velocity_1h: float
    volatility_spike_threshold: float


class RiskLimitsConfig(BaseModel):
    """Risk management limits"""
    max_daily_loss: float
    max_drawdown: float
    max_total_exposure: float
    per_instrument: PerInstrumentLimits
    max_leverage: float
    min_account_balance: float
    require_stop_loss: bool
    max_stop_loss_distance: float
    max_correlated_exposure: float
    circuit_breaker: CircuitBreakerConfig
    max_orders_per_minute: int
    max_open_positions: int


class StrategyConfig(BaseModel):
    """Strategy configuration"""
    name: str
    version: str
    enabled: bool
    strategy_class: str
    instruments: list[str]
    parameters: Dict[str, Any]
    backtest_results: Optional[Dict[str, Any]] = None


class AlertThresholds(BaseModel):
    """Monitoring alert thresholds"""
    market_data_latency_warning: float
    market_data_latency_critical: float
    market_data_stale_warning: float
    market_data_stale_critical: float
    order_fill_time_warning: float
    order_fill_time_critical: float
    order_rejection_rate_warning: float
    order_rejection_rate_critical: float
    cpu_usage_warning: float
    cpu_usage_critical: float
    memory_usage_warning: float
    memory_usage_critical: float
    strategy_error_rate_warning: float
    strategy_error_rate_critical: float


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    health_check_interval: int = 30
    alert_thresholds: AlertThresholds
    collection_interval: int = 10
    retention_days: int = 30
    prometheus_port: int = 8000
    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 75.0
    memory_critical: float = 90.0


class Config:
    """Main configuration class that loads and validates all config files"""

    def __init__(
        self,
        system: SystemConfig,
        redis: RedisConfig,
        influxdb: InfluxDBConfig,
        prometheus: PrometheusConfig,
        logging_config: LoggingConfig,
        oanda: OandaConfig,
        risk: RiskLimitsConfig,
        strategies: list[StrategyConfig],
        monitoring: MonitoringConfig
    ):
        self.system = system
        self.redis = redis
        self.influxdb = influxdb
        self.prometheus = prometheus
        self.logging = logging_config
        self.oanda = oanda
        self.risk = risk
        self.strategies = strategies
        self.monitoring = monitoring

    @classmethod
    def load(cls, config_dir: Optional[str] = None) -> "Config":
        """
        Load configuration from YAML files with environment variable substitution.

        Args:
            config_dir: Path to config directory (defaults to ./config)

        Returns:
            Config instance with all settings loaded
        """
        load_dotenv()
        if config_dir is None:
            # Default to config directory relative to project root
            config_dir = Path(__file__).parent.parent / "config"
        else:
            config_dir = Path(config_dir)

        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        logger.info(f"Loading configuration from {config_dir}")

        # Load each config file
        system_data = cls._load_yaml(config_dir / "system.yaml")
        oanda_data = cls._load_yaml(config_dir / "oanda.yaml")
        risk_data = cls._load_yaml(config_dir / "risk_limits.yaml")
        strategies_data = cls._load_yaml(config_dir / "strategies.yaml")
        monitoring_data = cls._load_yaml(config_dir / "monitoring.yaml")

        # Parse and validate configurations
        try:
            system_config = SystemConfig(**system_data.get("system", {}))
            redis_config = RedisConfig(**system_data.get("redis", {}))
            influxdb_config = InfluxDBConfig(**system_data.get("influxdb", {}))
            prometheus_config = PrometheusConfig(**system_data.get("prometheus", {}))
            logging_config = LoggingConfig(**system_data.get("logging", {}))

            oanda_config = OandaConfig(**oanda_data.get("oanda", {}))
            risk_config = RiskLimitsConfig(**risk_data.get("risk_limits", {}))

            strategies = [
                StrategyConfig(**s) for s in strategies_data.get("strategies", [])
            ]

            monitoring_config = MonitoringConfig(**monitoring_data.get("monitoring", {}))

            logger.info("Configuration loaded and validated successfully")

            return cls(
                system=system_config,
                redis=redis_config,
                influxdb=influxdb_config,
                prometheus=prometheus_config,
                logging_config=logging_config,
                oanda=oanda_config,
                risk=risk_config,
                strategies=strategies,
                monitoring=monitoring_config
            )

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}", exc_info=True)
            raise ValueError(f"Invalid configuration: {e}")

    @staticmethod
    def _load_yaml(file_path: Path) -> dict:
        """Load YAML file with environment variable substitution"""
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, 'r') as f:
            content = f.read()

        # Substitute environment variables (${VAR_NAME} format)
        content = Config._substitute_env_vars(content)

        return yaml.safe_load(content)

    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """Replace ${VAR_NAME} with environment variable values"""
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' not set but required in config"
                )
            return value

        return re.sub(pattern, replacer, content)
