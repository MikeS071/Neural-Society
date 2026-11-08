import unittest

from inventory import Inventory, InsufficientStock


class TestInventory(unittest.TestCase):
    def setUp(self):
        self.inv = Inventory()
        self.inv.add("apple", 10, 0.5)
        self.inv.add("pear", 3, 0.75)

    def test_add_existing(self):
        self.inv.add("apple", 5, 0.6)
        self.assertEqual(self.inv.quantity("apple"), 15)

    def test_total_value(self):
        self.assertAlmostEqual(self.inv.total_value(), 10 * 0.5 + 3 * 0.75)

    def test_remove(self):
        self.inv.remove("apple", 4)
        self.assertEqual(self.inv.quantity("apple"), 6)

    def test_remove_too_many(self):
        with self.assertRaises(InsufficientStock):
            self.inv.remove("pear", 4)

    def test_low_stock(self):
        self.assertEqual(self.inv.low_stock(5), ["pear"])


if __name__ == "__main__":
    unittest.main()
