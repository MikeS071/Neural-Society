"""Binary search over sorted lists."""


def lower_bound(xs, target):
    """First index i with xs[i] >= target (len(xs) if none)."""
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_search(xs, target):
    """Index of target in xs, or -1."""
    i = lower_bound(xs, target)
    if i < len(xs) and xs[i] == target:
        return i
    return -1


def insert_sorted(xs, value):
    """Return a new sorted list with value inserted."""
    i = lower_bound(xs, value)
    return xs[:i] + [value] + xs[i:]


def count_in_range(xs, lo, hi):
    """Number of elements v with lo <= v <= hi."""
    return lower_bound(xs, hi + 1) - lower_bound(xs, lo)
