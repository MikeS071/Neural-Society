import unittest

from roman import from_roman, to_roman


class TestRoman(unittest.TestCase):
    def test_to_roman(self):
        self.assertEqual(to_roman(1994), "MCMXCIV")
        self.assertEqual(to_roman(58), "LVIII")

    def test_from_roman(self):
        self.assertEqual(from_roman("MCMXCIV"), 1994)
        self.assertEqual(from_roman("III"), 3)

    def test_round_trip(self):
        for n in (1, 4, 9, 14, 40, 90, 400, 2024, 3999):
            self.assertEqual(from_roman(to_roman(n)), n)

    def test_range(self):
        with self.assertRaises(ValueError):
            to_roman(0)


if __name__ == "__main__":
    unittest.main()
