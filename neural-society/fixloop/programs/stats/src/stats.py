"""Descriptive statistics on lists of numbers."""


def mean(xs):
    if not xs:
        raise ValueError("mean of empty list")
    return sum(xs) / len(xs)


def median(xs):
    if not xs:
        raise ValueError("median of empty list")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def variance(xs):
    """Population variance."""
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def mode(xs):
    """Most common value; ties go to the smallest value."""
    if not xs:
        raise ValueError("mode of empty list")
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    return min(x for x, c in counts.items() if c == best)


def percentile(xs, p):
    """Nearest-rank percentile, p in [0, 100]."""
    if not 0 <= p <= 100:
        raise ValueError("p out of range")
    s = sorted(xs)
    if p == 0:
        return s[0]
    rank = -(-p * len(s) // 100)  # ceil(p * n / 100)
    return s[rank - 1]
