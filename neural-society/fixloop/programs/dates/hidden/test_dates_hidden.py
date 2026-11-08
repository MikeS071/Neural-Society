import unittest

from dates import add_days, day_of_year, days_in_month, is_leap


class TestDatesHidden(unittest.TestCase):
    def test_leap_more(self):
        self.assertFalse(is_leap(2023))
        self.assertFalse(is_leap(2100))
        self.assertTrue(is_leap(2400))

    def test_day_of_year_edges(self):
        self.assertEqual(day_of_year(2023, 1, 1), 1)
        self.assertEqual(day_of_year(2023, 12, 31), 365)
        self.assertEqual(day_of_year(2024, 12, 31), 366)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            days_in_month(2024, 13)
        with self.assertRaises(ValueError):
            day_of_year(2023, 2, 29)

    def test_add_days_long(self):
        self.assertEqual(add_days(2024, 1, 1, 366), (2025, 1, 1))
        self.assertEqual(add_days(2023, 2, 28, 1), (2023, 3, 1))
        self.assertEqual(add_days(2024, 2, 28, 1), (2024, 2, 29))
        self.assertEqual(add_days(2024, 5, 5, 0), (2024, 5, 5))


if __name__ == "__main__":
    unittest.main()
