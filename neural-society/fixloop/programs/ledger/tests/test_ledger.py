import unittest

from ledger import balance, statement

LINES = [
    "# date,kind,amount",
    "2024-01-01,deposit,100.00",
    "2024-01-02,withdrawal,20.5",
    "2024-01-03,deposit,0.05",
    "",
]


class TestLedger(unittest.TestCase):
    def test_balance(self):
        self.assertEqual(balance(LINES), 7955)

    def test_statement(self):
        self.assertEqual(statement(LINES), "Balance: 79.55")

    def test_negative_statement(self):
        self.assertEqual(statement(["2024-01-01,withdrawal,1.10"]), "Balance: -1.10")

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            balance(["2024-01-01,refund,1.00"])


if __name__ == "__main__":
    unittest.main()
