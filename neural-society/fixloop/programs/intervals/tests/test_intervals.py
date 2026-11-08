import unittest

from intervals import merge_intervals, overlaps, total_length


class TestIntervals(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(merge_intervals([(6, 8), (1, 3), (2, 4)]), [(1, 4), (6, 8)])

    def test_merge_contained(self):
        self.assertEqual(merge_intervals([(1, 10), (2, 3)]), [(1, 10)])

    def test_total_length(self):
        self.assertEqual(total_length([(1, 3), (10, 10)]), 4)

    def test_overlaps(self):
        self.assertTrue(overlaps((1, 5), (5, 9)))
        self.assertFalse(overlaps((1, 4), (6, 9)))


if __name__ == "__main__":
    unittest.main()
