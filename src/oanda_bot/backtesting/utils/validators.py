"""Validation helpers."""

def is_symbol(s: str) -> bool:
    return isinstance(s, str) and len(s) > 0
