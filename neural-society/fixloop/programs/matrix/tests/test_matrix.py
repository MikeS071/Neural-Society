import unittest

from matrix import determinant, identity, matmul, transpose


class TestMatrix(unittest.TestCase):
    def test_transpose(self):
        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])

    def test_matmul(self):
        self.assertEqual(matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]])

    def test_matmul_identity(self):
        m = [[2, 0, 1], [1, 3, 2]]
        self.assertEqual(matmul(m, identity(3)), m)

    def test_matmul_mismatch(self):
        with self.assertRaises(ValueError):
            matmul([[1, 2]], [[1, 2]])

    def test_determinant(self):
        self.assertEqual(determinant([[1, 2], [3, 4]]), -2)
        self.assertEqual(determinant([[2, 0, 1], [1, 3, 2], [1, 1, 2]]), 6)


if __name__ == "__main__":
    unittest.main()
