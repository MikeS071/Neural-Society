import unittest

from graph import build, components, shortest_path


class TestGraphHidden(unittest.TestCase):
    def test_same_node(self):
        self.assertEqual(shortest_path(build([(1, 2)]), 1, 1), [1])

    def test_line(self):
        g = build([(i, i + 1) for i in range(10)])
        self.assertEqual(shortest_path(g, 0, 10), list(range(11)))
        self.assertEqual(shortest_path(g, 10, 0), list(range(10, -1, -1)))

    def test_missing_node(self):
        self.assertIsNone(shortest_path(build([(1, 2)]), 1, 99))

    def test_components_many(self):
        g = build([(5, 6), (1, 2), (3, 3), (9, 8), (8, 7)])
        self.assertEqual(components(g), [[1, 2], [3], [5, 6], [7, 8, 9]])


if __name__ == "__main__":
    unittest.main()
