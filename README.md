# oanda-trading-system

Lightweight skeleton for an OANDA backtesting / research system. Fill modules under `backtesting/` with real implementations.

See `scripts/` for quick runners.
# Multi-Agent Forex Trading System - Phase 1

A production-ready multi-agent algorithmic trading system for forex markets using the Oanda v20 REST API.

## Phase 1: Core Infrastructure

This phase implements the foundational components:
- Complete project directory structure
- Shared data models with Pydantic validation
- Redis-based message bus for inter-agent communication
- Configuration system using YAML + environment variables
- Centralized logging with structured output
- Docker Compose for infrastructure services
- Comprehensive unit tests

## Project Structure

```
oanda-trading-system/
├── agents/                 # Agent implementations (future phases)
│   ├── monitoring/
│   ├── market_data/
│   ├── strategy/
│   ├── risk/
│   └── execution/
├── shared/                 # Shared modules
│   ├── models.py          # Pydantic data models
│   ├── message_bus.py     # Redis Streams wrapper
│   ├── config.py          # Configuration loading
│   ├── logging_config.py  # Logging setup
│   └── utils.py           # Utility functions
├── config/                 # Configuration files
│   ├── system.yaml
│   ├── oanda.yaml
│   ├── risk_limits.yaml
│   ├── strategies.yaml
│   └── monitoring.yaml
├── tests/                  # Unit tests
├── scripts/                # Utility scripts
├── data/historical/        # Historical data cache
├── logs/                   # Log files
├── docker-compose.yml      # Infrastructure services
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variables template
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Oanda practice account (for testing)

## Installation

### 1. Install Python Dependencies

```bash
cd oanda-trading-system
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Oanda credentials:
```bash
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_API_TOKEN=your_api_token_here
INFLUXDB_TOKEN=my-super-secret-token
```

### 3. Start Infrastructure Services

```bash
docker-compose up -d
```

This starts:
- **Redis** (port 6379): Message bus
- **InfluxDB** (port 8086): Time-series database
- **Grafana** (port 3000): Visualization (admin/admin)
- **Prometheus** (port 9090): Metrics collection

Verify services are running:
```bash
docker-compose ps
```

## Running Tests

Run all unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=shared --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_models.py -v
```

## Core Components

### Data Models (`shared/models.py`)

Pydantic models for type-safe data handling:
- `MarketTick`: Normalized market data
- `TradeSignal`: Strategy-generated signals
- `RiskCheckResult`: Risk approval results
- `Order`: Order to be executed
- `Execution`: Fill reports
- `Position`: Current positions
- `HealthMetric`: System health metrics

### Message Bus (`shared/message_bus.py`)

Redis Streams-based pub/sub system:
- Async publish/subscribe
- Consumer groups for reliable delivery
- Named streams for different message types
- Automatic reconnection

### Configuration (`shared/config.py`)

YAML-based configuration with environment variable substitution:
- System settings
- Oanda API configuration
- Risk limits
- Strategy parameters
- Monitoring thresholds

### Logging (`shared/logging_config.py`)

Structured JSON logging:
- Console output (human-readable in dev)
- File rotation
- Error-specific log file
- Ready for ELK stack integration

## Testing the Message Bus

Test the message bus manually:

```python
import asyncio
from shared.message_bus import MessageBus
from shared.config import Config

async def test():
    config = Config.load()
    bus = MessageBus(config)
    await bus.connect()

    # Publish a message
    await bus.publish('market_data', {'test': 'hello', 'value': 123})

    # Subscribe and receive
    async for msg in bus.subscribe('market_data'):
        print(f"Received: {msg}")
        break

    await bus.disconnect()

asyncio.run(test())
```

## Configuration Files

### System Configuration (`config/system.yaml`)
- Redis connection settings
- InfluxDB connection
- Prometheus settings
- Logging configuration

### Oanda Configuration (`config/oanda.yaml`)
- API credentials (via environment variables)
- Practice/live environment selection
- Instruments to trade
- Connection settings

### Risk Limits (`config/risk_limits.yaml`)
- Daily loss limits
- Position size limits
- Stop loss requirements
- Circuit breaker settings

### Strategies (`config/strategies.yaml`)
- Strategy definitions
- Parameters
- Backtest results

### Monitoring (`config/monitoring.yaml`)
- Health check intervals
- Alert thresholds
- Metrics collection settings

## Acceptance Criteria

Phase 1 is complete when:
- [x] All directory structure created
- [x] All Pydantic models defined and validated
- [x] Redis message bus working (can publish/subscribe)
- [x] Configuration loads from YAML + env vars
- [x] Logging outputs structured JSON
- [x] Docker Compose brings up all services successfully
- [x] All unit tests pass
- [x] Can create a test message flow
- [x] README documents setup and usage

## Next Steps (Phase 2+)

Future phases will implement:
- **Phase 2**: Market Data Agent (Oanda streaming, normalization, storage)
- **Phase 3**: Monitoring Agent (health checks, metrics, alerting)
- **Phase 4**: Strategy Agent (signal generation, backtesting)
- **Phase 5**: Risk Agent (pre-trade checks, position monitoring)
- **Phase 6**: Execution Agent (order management, Oanda execution)
- **Phase 7**: Integration testing and optimization
- **Phase 8**: Production deployment

## Troubleshooting

### Redis Connection Issues
```bash
# Check Redis is running
docker-compose ps redis

# View Redis logs
docker-compose logs redis

# Test Redis connection
redis-cli ping
```

### InfluxDB Connection Issues
```bash
# Check InfluxDB is running
docker-compose ps influxdb

# View InfluxDB logs
docker-compose logs influxdb

# Access InfluxDB UI
open http://localhost:8086
```

### Test Failures
```bash
# Run tests with verbose output
pytest tests/ -vv

# Run specific test
pytest tests/test_models.py::TestMarketTick::test_valid_market_tick -v

# Show print statements
pytest tests/ -v -s
```

## Security Notes

- Never commit `.env` file with real credentials
- Use practice environment for development
- Rotate API tokens regularly
- Monitor API usage to avoid rate limits
- Use encrypted connections in production

## License

Proprietary - For authorized use only

## Support

For issues or questions, refer to the implementation guide or contact the development team.
