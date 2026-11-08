"""A tiny stock-keeping model."""


class InsufficientStock(Exception):
    pass


class Inventory:
    def __init__(self):
        self._items = {}  # name -> [quantity, unit_price]

    def add(self, name, quantity, unit_price):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if name in self._items:
            self._items[name][0] += quantity
            self._items[name][1] = unit_price
        else:
            self._items[name] = [quantity, unit_price]

    def remove(self, name, quantity):
        have = self.quantity(name)
        if quantity > have:
            raise InsufficientStock(f"{name}: have {have}, need {quantity}")
        self._items[name][0] -= quantity
        if self._items[name][0] == 0:
            del self._items[name]

    def quantity(self, name):
        return self._items.get(name, [0, 0])[0]

    def total_value(self):
        return sum(q * p for q, p in self._items.values())

    def low_stock(self, threshold):
        """Names with quantity strictly below threshold, sorted."""
        return sorted(n for n, (q, _) in self._items.items() if q < threshold)
