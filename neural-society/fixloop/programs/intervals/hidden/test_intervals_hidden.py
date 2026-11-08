import unittest

from intervals import merge_intervals, overlaps, total_length


class TestIntervalsHidden(unittest.TestCase):
    def test_merge_empty(self):
        self.assertEqual(merge_intervals([]), [])

    def test_merge_touching(self):
        self.assertEqual(merge_intervals([(1, 2), (3, 4)]), [(1, 4)])

    def test_merge_gap(self):
        self.assertEqual(merge_intervals([(1, 2), (4, 5)]), [(1, 2), (4, 5)])

    def test_total_length_overlap(self):
        self.assertEqual(total_length([(1, 5), (3, 7), (20, 21)]), 9)

    def test_overlaps_contained(self):
        self.assertTrue(overlaps((1, 10), (4, 5)))
        self.assertTrue(overlaps((4, 5), (1, 10)))


if __name__ == "__main__":
    unittest.main()
