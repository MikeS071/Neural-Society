import unittest

from inventory import Inventory, InsufficientStock


class TestInventoryHidden(unittest.TestCase):
    def test_remove_all_deletes(self):
        inv = Inventory()
        inv.add("x", 2, 1.0)
        inv.remove("x", 2)
        self.assertEqual(inv.quantity("x"), 0)
        self.assertEqual(inv.low_stock(100), [])

    def test_remove_missing(self):
        with self.assertRaises(InsufficientStock):
            Inventory().remove("ghost", 1)

    def test_price_updates(self):
        inv = Inventory()
        inv.add("a", 1, 1.0)
        inv.add("a", 1, 3.0)
        self.assertAlmostEqual(inv.total_value(), 6.0)

    def test_bad_quantity(self):
        with self.assertRaises(ValueError):
            Inventory().add("a", 0, 1.0)

    def test_low_stock_boundary(self):
        inv = Inventory()
        inv.add("a", 5, 1.0)
        inv.add("b", 4, 1.0)
        self.assertEqual(inv.low_stock(5), ["b"])


if __name__ == "__main__":
    unittest.main()
