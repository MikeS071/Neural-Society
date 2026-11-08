import unittest

from dates import add_days, day_of_year, days_in_month, is_leap


class TestDates(unittest.TestCase):
    def test_leap(self):
        self.assertTrue(is_leap(2024))
        self.assertFalse(is_leap(1900))
        self.assertTrue(is_leap(2000))

    def test_days_in_month(self):
        self.assertEqual(days_in_month(2023, 2), 28)
        self.assertEqual(days_in_month(2024, 2), 29)
        self.assertEqual(days_in_month(2024, 12), 31)

    def test_day_of_year(self):
        self.assertEqual(day_of_year(2024, 3, 1), 61)

    def test_add_days(self):
        self.assertEqual(add_days(2023, 12, 30, 3), (2024, 1, 2))


if __name__ == "__main__":
    unittest.main()
