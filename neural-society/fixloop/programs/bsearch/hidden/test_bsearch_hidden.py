import unittest

from bsearch import binary_search, count_in_range, insert_sorted, lower_bound


class TestBsearchHidden(unittest.TestCase):
    def test_edges(self):
        self.assertEqual(lower_bound([], 5), 0)
        self.assertEqual(lower_bound([1, 2], 0), 0)
        self.assertEqual(lower_bound([1, 2], 9), 2)
        self.assertEqual(binary_search([], 1), -1)
        self.assertEqual(binary_search([2], 2), 0)
        self.assertEqual(binary_search([1, 2, 3], 3), 2)

    def test_insert_ends(self):
        self.assertEqual(insert_sorted([], 1), [1])
        self.assertEqual(insert_sorted([2, 3], 1), [1, 2, 3])
        self.assertEqual(insert_sorted([2, 3], 4), [2, 3, 4])

    def test_insert_does_not_mutate(self):
        xs = [1, 2]
        insert_sorted(xs, 0)
        self.assertEqual(xs, [1, 2])

    def test_count_brute_force(self):
        xs = [0, 1, 1, 2, 4, 4, 4, 7, 9]
        for lo in range(-1, 11):
            for hi in range(lo, 11):
                self.assertEqual(count_in_range(xs, lo, hi), sum(lo <= v <= hi for v in xs))


if __name__ == "__main__":
    unittest.main()
