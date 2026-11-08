"""Dense matrices as lists of row lists."""


def shape(m):
    return (len(m), len(m[0]) if m else 0)


def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def transpose(m):
    rows, cols = shape(m)
    return [[m[r][c] for r in range(rows)] for c in range(cols)]


def matmul(a, b):
    ar, ac = shape(a)
    br, bc = shape(b)
    if ac != br:
        raise ValueError(f"shape mismatch {ar}x{ac} @ {br}x{bc}")
    return [[sum(a[i][k] * b[k][j] for k in range(ac)) for j in range(bc)] for i in range(ar)]


def minor(m, row, col):
    return [r[:col] + r[col + 1:] for i, r in enumerate(m) if i != row]


def determinant(m):
    n, c = shape(m)
    if n != c:
        raise ValueError("determinant of non-square matrix")
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return sum((-1) ** j * m[0][j] * determinant(minor(m, 0, j)) for j in range(n))
