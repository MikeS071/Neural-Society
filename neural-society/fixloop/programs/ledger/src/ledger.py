"""A ledger of money transactions, kept in integer cents."""


def parse_amount(text):
    """'12.34' -> 1234, '-0.5' -> -50, '7' -> 700. At most two decimal places."""
    text = text.strip()
    sign = -1 if text.startswith("-") else 1
    text = text.lstrip("+-")
    whole, _, frac = text.partition(".")
    if len(frac) > 2 or not (whole or frac):
        raise ValueError(f"bad amount {text!r}")
    cents = int(whole or "0") * 100 + int(frac.ljust(2, "0") or "0")
    return sign * cents


def format_amount(cents):
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"


def parse_line(line):
    """'2024-01-05,deposit,12.50' -> ('2024-01-05', 'deposit', 1250)."""
    date, kind, amount = [p.strip() for p in line.split(",")]
    if kind not in ("deposit", "withdrawal"):
        raise ValueError(f"unknown kind {kind!r}")
    return date, kind, parse_amount(amount)


def balance(lines):
    total = 0
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        _, kind, cents = parse_line(line)
        total += cents if kind == "deposit" else -cents
    return total


def statement(lines):
    return f"Balance: {format_amount(balance(lines))}"
