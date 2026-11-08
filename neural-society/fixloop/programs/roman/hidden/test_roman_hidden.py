import unittest

from roman import from_roman, to_roman


class TestRomanHidden(unittest.TestCase):
    def test_round_trip_all(self):
        for n in range(1, 4000):
            self.assertEqual(from_roman(to_roman(n)), n)

    def test_known(self):
        self.assertEqual(to_roman(3888), "MMMDCCCLXXXVIII")
        self.assertEqual(to_roman(4), "IV")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            from_roman("ABC")
        with self.assertRaises(ValueError):
            from_roman("")
        with self.assertRaises(ValueError):
            to_roman(4000)


if __name__ == "__main__":
    unittest.main()
