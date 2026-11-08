import unittest

from ledger import balance, format_amount, parse_amount


class TestLedgerHidden(unittest.TestCase):
    def test_parse_amount(self):
        self.assertEqual(parse_amount("12.34"), 1234)
        self.assertEqual(parse_amount("-0.5"), -50)
        self.assertEqual(parse_amount("7"), 700)
        self.assertEqual(parse_amount(" .09 "), 9)
        self.assertEqual(parse_amount("+3.1"), 310)

    def test_parse_amount_bad(self):
        with self.assertRaises(ValueError):
            parse_amount("1.234")
        with self.assertRaises(ValueError):
            parse_amount(".")

    def test_format(self):
        self.assertEqual(format_amount(5), "0.05")
        self.assertEqual(format_amount(-123456), "-1234.56")
        self.assertEqual(format_amount(0), "0.00")

    def test_balance_many(self):
        lines = [f"2024-01-{d:02d},deposit,0.01" for d in range(1, 31)]
        self.assertEqual(balance(lines), 30)


if __name__ == "__main__":
    unittest.main()
