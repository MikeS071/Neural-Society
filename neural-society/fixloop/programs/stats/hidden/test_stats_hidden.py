import unittest

from stats import mean, median, mode, percentile, variance


class TestStatsHidden(unittest.TestCase):
    def test_mean_single(self):
        self.assertEqual(mean([7]), 7)

    def test_median_single(self):
        self.assertEqual(median([9]), 9)

    def test_median_even_large(self):
        self.assertEqual(median([10, 40, 20, 30, 50, 60]), 35)

    def test_variance_constant(self):
        self.assertEqual(variance([3, 3, 3]), 0)

    def test_mode_single_winner(self):
        self.assertEqual(mode([4, 4, 4, 1, 2]), 4)

    def test_percentile_edges(self):
        xs = [3, 6, 7, 8, 8, 10, 13, 15, 16, 20]
        self.assertEqual(percentile(xs, 0), 3)
        self.assertEqual(percentile(xs, 25), 7)
        self.assertEqual(percentile(xs, 50), 8)
        self.assertEqual(percentile(xs, 75), 15)

    def test_percentile_range(self):
        with self.assertRaises(ValueError):
            percentile([1, 2], 101)


if __name__ == "__main__":
    unittest.main()
