"""Portfolio & position tracking (skeleton)."""

class Portfolio:
    def __init__(self, cash=100000):
        self.cash = cash
        self.positions = {}

    def update_on_fill(self, fill):
        pass
