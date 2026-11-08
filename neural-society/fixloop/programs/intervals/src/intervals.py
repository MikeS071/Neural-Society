"""Closed integer intervals as (start, end) tuples with start <= end."""


def merge_intervals(intervals):
    """Merge overlapping or touching intervals; returns a sorted list."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_length(intervals):
    """Number of integers covered by the union of the intervals."""
    return sum(end - start + 1 for start, end in merge_intervals(intervals))


def overlaps(a, b):
    """True if the two closed intervals share at least one integer."""
    return a[0] <= b[1] and b[0] <= a[1]
