import unittest

from graph import build, components, shortest_path

EDGES = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 4), (7, 8)]


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.g = build(EDGES)

    def test_shortest(self):
        self.assertEqual(shortest_path(self.g, 1, 4), [1, 5, 4])

    def test_unreachable(self):
        self.assertIsNone(shortest_path(self.g, 1, 8))

    def test_components(self):
        self.assertEqual(components(self.g), [[1, 2, 3, 4, 5], [7, 8]])


if __name__ == "__main__":
    unittest.main()
