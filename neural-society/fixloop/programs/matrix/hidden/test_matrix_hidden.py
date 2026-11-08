import unittest

from matrix import determinant, identity, matmul, transpose


class TestMatrixHidden(unittest.TestCase):
    def test_transpose_square(self):
        self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])

    def test_matmul_rect(self):
        self.assertEqual(matmul([[1, 2, 3]], [[1], [2], [3]]), [[14]])
        self.assertEqual(matmul([[1], [2]], [[3, 4]]), [[3, 4], [6, 8]])

    def test_det_identity(self):
        for n in range(1, 5):
            self.assertEqual(determinant(identity(n)), 1)

    def test_det_4x4(self):
        m = [[1, 0, 2, -1], [3, 0, 0, 5], [2, 1, 4, -3], [1, 0, 5, 0]]
        self.assertEqual(determinant(m), 30)

    def test_det_nonsquare(self):
        with self.assertRaises(ValueError):
            determinant([[1, 2, 3], [4, 5, 6]])


if __name__ == "__main__":
    unittest.main()
