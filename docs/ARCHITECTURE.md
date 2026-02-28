# System Architecture

## Overview

The Oanda Trading System is a multi-agent forex trading platform built on event-driven architecture using Redis Streams for inter-agent communication.

## Architecture Diagram

```
┌─────────────────┐
│  Oanda API      │
│  (Market Data)  │
└────────┬────────┘
         │
         v
┌─────────────────┐      ┌──────────────┐
│  Market Data    │─────>│   InfluxDB   │
│     Agent       │      │  (Storage)   │
└────────┬────────┘      └──────────────┘
         │
         v
    Redis Streams
         │
         v
┌─────────────────┐
│   Strategy      │
│     Agent       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│     Risk        │
│     Agent       │
└────────┬────────┘
         │
         v
┌─────────────────┐      ┌──────────────┐
│   Execution     │─────>│  Oanda API   │
│     Agent       │      │  (Trading)   │
└─────────────────┘      └──────────────┘
         │
         v
┌─────────────────┐
│   Monitoring    │
│     Agent       │
└─────────────────┘
```

## Components

### 1. Market Data Agent
- **Purpose**: Stream live market data from Oanda
- **Input**: Oanda streaming API
- **Output**: stream:market_data
- **Key Features**: Data validation, tick normalization

### 2. Strategy Agent
- **Purpose**: Generate trading signals from market data
- **Input**: stream:market_data
- **Output**: stream:signals
- **Strategies**: MA Crossover, Bollinger Bands, RSI

### 3. Risk Agent
- **Purpose**: Validate signals against risk limits
- **Input**: stream:signals
- **Output**: stream:risk_checks
- **Key Features**: Position sizing, circuit breaker, drawdown limits

### 4. Execution Agent
- **Purpose**: Execute approved trades on Oanda
- **Input**: stream:risk_checks
- **Output**: stream:executions
- **Key Features**: Order management, retry logic, fill tracking

### 5. Monitoring Agent
- **Purpose**: System health monitoring and alerting
- **Input**: All streams
- **Output**: stream:alerts, Prometheus metrics
- **Key Features**: Health checks, performance metrics, alerting

## Data Flow

1. **Tick Reception**: Market Data Agent receives ticks from Oanda
2. **Validation**: Ticks validated and stored in InfluxDB
3. **Publication**: Valid ticks published to stream:market_data
4. **Signal Generation**: Strategy Agent consumes ticks, generates signals
5. **Risk Check**: Risk Agent validates signals against limits
6. **Execution**: Execution Agent executes approved signals
7. **Monitoring**: Monitoring Agent tracks all activity

## Message Bus (Redis Streams)

### Streams

- `stream:market_data` - Market ticks
- `stream:signals` - Trading signals
- `stream:risk_checks` - Risk validation results
- `stream:executions` - Order executions
- `stream:alerts` - System alerts

### Consumer Groups

Each agent uses consumer groups for reliable message delivery and load balancing.

## Data Storage

### InfluxDB
- **Purpose**: Time-series storage for market data
- **Retention**: 90 days
- **Measurements**: ticks, signals, executions

### Redis
- **Purpose**: Message bus and state storage
- **Persistence**: AOF enabled
- **Data**: Streams, circuit breaker state, metrics

## External Dependencies

- **Oanda v20 API**: Market data and trade execution
- **Redis 7.0+**: Message bus
- **InfluxDB 2.0+**: Time-series storage
- **Prometheus**: Metrics collection (optional)

## Design Decisions

### Event-Driven Architecture
- **Why**: Loose coupling, scalability, fault tolerance
- **Trade-off**: Eventual consistency, debugging complexity

### Redis Streams
- **Why**: Reliable, ordered, persistent message delivery
- **Trade-off**: Single point of failure (mitigated by Redis persistence)

### Multi-Agent Design
- **Why**: Separation of concerns, independent scaling
- **Trade-off**: Operational complexity

### Async/Await
- **Why**: High concurrency, efficient I/O handling
- **Trade-off**: Complexity in error handling

## Scalability

- **Horizontal**: Multiple strategy agents can run in parallel
- **Vertical**: Each agent can handle 1000+ messages/second
- **Bottlenecks**: Oanda API rate limits, Redis throughput

## Security

- **Credentials**: Environment variables only
- **Network**: Redis/InfluxDB authentication
- **Logging**: No sensitive data in logs
- **Access**: Principle of least privilege
