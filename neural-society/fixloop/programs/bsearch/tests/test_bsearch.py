import unittest

from bsearch import binary_search, count_in_range, insert_sorted, lower_bound


class TestBsearch(unittest.TestCase):
    xs = [1, 3, 3, 5, 8, 13]

    def test_lower_bound(self):
        self.assertEqual(lower_bound(self.xs, 3), 1)
        self.assertEqual(lower_bound(self.xs, 4), 3)

    def test_binary_search(self):
        self.assertEqual(binary_search(self.xs, 8), 4)
        self.assertEqual(binary_search(self.xs, 7), -1)

    def test_insert(self):
        self.assertEqual(insert_sorted([1, 4, 9], 5), [1, 4, 5, 9])

    def test_count_in_range(self):
        self.assertEqual(count_in_range(self.xs, 3, 8), 4)


if __name__ == "__main__":
    unittest.main()
