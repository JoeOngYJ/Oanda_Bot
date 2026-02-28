# Shared Module

Shared runtime building blocks:
1. `models.py`: Pydantic contracts for stream payloads and domain entities.
2. `message_bus.py`: Redis Streams abstraction (publish/subscribe/consumer groups).
3. `config.py`: YAML configuration loading with environment substitution.
4. `logging_config.py`: logging setup utilities.
5. `utils.py`: common helpers.

Contract changes in `models.py` should be treated as breaking unless backward-compatible.

