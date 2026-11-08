"""Roman numerals for 1..3999."""

_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]
_SYMBOL = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def to_roman(n):
    if not 1 <= n <= 3999:
        raise ValueError("out of range")
    out = []
    for value, sym in _VALUES:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def from_roman(s):
    if not s or any(c not in _SYMBOL for c in s):
        raise ValueError("invalid numeral")
    total = 0
    for i, c in enumerate(s):
        v = _SYMBOL[c]
        if i + 1 < len(s) and _SYMBOL[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total
