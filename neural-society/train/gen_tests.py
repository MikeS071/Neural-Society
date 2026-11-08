"""Generate benchmark expressions with ground truth from Python big ints.

Writes two TSV files (columns: n_digits, n_ops, expr, answer) to ../data/:
  tests.tsv         sweep: every operand has exactly n_digits digits (1-30), 1-5 operators
                    drawn uniformly from + - *, standard precedence (* binds tighter).
  tests_3digit.tsv  baseline-comparable: one operator, operands uniform in 0..999,
                    i.e. the monolithic baseline's own input distribution.

Usage: python3 gen_tests.py [per_cell] [seed]
"""
import os, random, sys

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
LENGTHS = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
N_OPS = [1, 2, 3, 4, 5]


def operand(rng, n):
    if n == 1:
        return str(rng.randrange(10))
    return str(rng.randrange(1, 10)) + "".join(str(rng.randrange(10)) for _ in range(n - 1))


def expression(rng, n_digits, n_ops):
    parts = [operand(rng, n_digits)]
    for _ in range(n_ops):
        parts += [rng.choice("+-*"), operand(rng, n_digits)]
    return "".join(parts)


def truth(expr):
    # Only digits and + - * by construction; Python ints are arbitrary precision.
    assert set(expr) <= set("0123456789+-*")
    return eval(expr)


if __name__ == "__main__":
    per_cell = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    rng = random.Random(seed)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tests.tsv"), "w") as f:
        f.write("# n_digits\tn_ops\texpr\tanswer\n")
        for n in LENGTHS:
            for k in N_OPS:
                for _ in range(per_cell):
                    e = expression(rng, n, k)
                    f.write(f"{n}\t{k}\t{e}\t{truth(e)}\n")
    with open(os.path.join(OUT, "tests_3digit.tsv"), "w") as f:
        f.write("# n_digits\tn_ops\texpr\tanswer\n")
        for _ in range(3000):
            a, b, op = rng.randrange(1000), rng.randrange(1000), rng.choice("+-*")
            e = f"{a}{op}{b}"
            f.write(f"3\t1\t{e}\t{truth(e)}\n")
    print(f"wrote {len(LENGTHS) * len(N_OPS) * per_cell} sweep + 3000 baseline-distribution tests to {OUT}")
