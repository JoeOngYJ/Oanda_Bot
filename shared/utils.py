"""
Common utility functions for the trading system.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique identifier.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Unique identifier string
    """
    unique_id = str(uuid.uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id


def utc_now() -> datetime:
    """Get current UTC datetime"""
    return datetime.utcnow()


def pips_to_price(pips: int, pip_value: Decimal = Decimal("0.0001")) -> Decimal:
    """
    Convert pips to price value.

    Args:
        pips: Number of pips
        pip_value: Value of one pip (default 0.0001 for most majors)

    Returns:
        Price value as Decimal
    """
    return Decimal(pips) * pip_value


def price_to_pips(price: Decimal, pip_value: Decimal = Decimal("0.0001")) -> int:
    """
    Convert price value to pips.

    Args:
        price: Price value
        pip_value: Value of one pip (default 0.0001 for most majors)

    Returns:
        Number of pips as integer
    """
    return int(price / pip_value)


def safe_decimal(value: Any) -> Decimal:
    """
    Safely convert value to Decimal.

    Args:
        value: Value to convert

    Returns:
        Decimal value
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        return Decimal(value)
    raise ValueError(f"Cannot convert {type(value)} to Decimal")
