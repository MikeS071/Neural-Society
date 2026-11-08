import unittest

from stats import mean, median, mode, percentile, variance


class TestStats(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

    def test_mean_empty(self):
        with self.assertRaises(ValueError):
            mean([])

    def test_median_odd(self):
        self.assertEqual(median([5, 1, 3]), 3)

    def test_median_even(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_variance(self):
        self.assertAlmostEqual(variance([2, 4, 4, 4, 5, 5, 7, 9]), 4.0)

    def test_mode_tie(self):
        self.assertEqual(mode([3, 1, 3, 1, 2]), 1)

    def test_percentile(self):
        xs = [15, 20, 35, 40, 50]
        self.assertEqual(percentile(xs, 30), 20)
        self.assertEqual(percentile(xs, 100), 50)


if __name__ == "__main__":
    unittest.main()
