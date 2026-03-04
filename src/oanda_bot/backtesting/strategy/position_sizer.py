"""Position sizing utilities (stub)."""

def fixed_size(account_size, fraction=0.01):
    return max(1, int(account_size * fraction))
