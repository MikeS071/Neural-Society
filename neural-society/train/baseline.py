"""Monolithic baseline: one MLP that maps `a op b` straight to the answer.

Input  : a (3 fixed-width digits, one-hot) + op (+,-,*) + b (3 fixed-width digits)
         = 30 + 3 + 30 = 63 features. Operands are 0..999.
Output : 6 result-digit heads (10-way each, most significant first) for |a op b|
         (999*999 = 998001 needs 6 digits) + 1 sign head (2-way: 0 = >=0, 1 = <0).

Trained on a random subset of the 3 * 1000 * 1000 = 3M possible problems and evaluated
on a disjoint held-out set. Three parameter budgets: ~equal to all four specialists
combined (31,493 params), 10x and 100x.

Structural limitation (part of the result): the input layer is 3 digits wide, so this
model cannot even be *asked* about a 4-digit operand or a 2-operator expression.

Usage: python3 baseline.py [--quick]
"""
import json, os, sys, time
import numpy as np
from nn import MLP, one_hot

OUT = os.path.join(os.path.dirname(__file__), "..", "weights")
SPECIALIST_PARAMS = 6932 + 6932 + 11930 + 5699  # add + sub + mul + cmp = 31,493
N_DIG, N_OUT_DIG = 3, 6
OPS = "+-*"
FIELDS = [10] * N_DIG + [3] + [10] * N_DIG
HEADS = [(10 * i, 10 * i + 10) for i in range(N_OUT_DIG)] + [(60, 62)]
N_IN, N_OUTPUTS = sum(FIELDS), 62


def digits(x, n):
    """(m,) ints -> (m, n) digits, most significant first."""
    return np.stack([(x // 10 ** (n - 1 - i)) % 10 for i in range(n)], 1)


def encode(a, op, b):
    return one_hot(np.concatenate([digits(a, N_DIG), op[:, None], digits(b, N_DIG)], 1), FIELDS)


def targets(a, op, b):
    r = np.where(op == 0, a + b, np.where(op == 1, a - b, a * b))
    return list(digits(np.abs(r), N_OUT_DIG).T) + [(r < 0).astype(np.int64)], r


def decode(logits):
    d = np.stack([logits[:, s:e].argmax(1) for s, e in HEADS[:-1]], 1)
    mag = (d * 10 ** np.arange(N_OUT_DIG - 1, -1, -1)).sum(1)
    sign = logits[:, 60:62].argmax(1)
    return np.where(sign == 1, -mag, mag)


def width_for(budget):
    """Two hidden layers of width w: params = w^2 + (N_IN + N_OUTPUTS + 2) w + N_OUTPUTS."""
    c = N_IN + N_OUTPUTS + 2
    return int(round((-c + np.sqrt(c * c + 4 * (budget - N_OUTPUTS))) / 2))


def sample_problems(rng, n):
    """Distinct (a, op, b) triples, as flat indices into the 3M problem space."""
    idx = rng.choice(3 * 1000 * 1000, size=n, replace=False)
    return idx // 1_000_000, (idx // 1000) % 1000, idx % 1000  # op, a, b


def run(budget_mult, n_train, n_test, steps, batch, seed=0):
    rng = np.random.default_rng(seed)
    op, a, b = sample_problems(rng, n_train + n_test)  # disjoint by construction
    tr, te = slice(0, n_train), slice(n_train, None)
    Xtr, Xte = encode(a[tr], op[tr], b[tr]), encode(a[te], op[te], b[te])
    ytr, _ = targets(a[tr], op[tr], b[tr])
    _, rte = targets(a[te], op[te], b[te])

    w = width_for(SPECIALIST_PARAMS * budget_mult)
    net = MLP([N_IN, w, w, N_OUTPUTS], seed=seed)
    t0 = time.time()
    lr0 = 2e-3
    for s in range(steps):
        lr = lr0 * 0.5 * (1 + np.cos(np.pi * s / steps))  # cosine decay
        bi = rng.integers(0, n_train, batch)
        loss = net.step(Xtr[bi], HEADS, [y[bi] for y in ytr], lr=lr)
        if s % max(1, steps // 10) == 0:
            print(f"  [{budget_mult}x w={w}] step {s} loss {loss:.4f}", flush=True)

    def acc(X, r, o):
        pred = np.concatenate([decode(net.logits(X[i:i + 8192])) for i in range(0, len(X), 8192)])
        out = {"all": float((pred == r).mean())}
        for k, name in enumerate(OPS):
            out[name] = float((pred[o == k] == r[o == k]).mean())
        return out

    _, rtr = targets(a[tr][:n_test], op[tr][:n_test], b[tr][:n_test])
    res = dict(budget=f"{budget_mult}x", params=net.n_params(), hidden=[w, w],
               n_train=n_train, n_test=n_test, steps=steps, batch=batch,
               train_s=round(time.time() - t0, 1),
               train_acc=acc(Xtr[:n_test], rtr, op[tr][:n_test]),
               heldout_acc=acc(Xte, rte, op[te]))
    net.save(os.path.join(OUT, f"baseline_{budget_mult}x.bin"))
    return res


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    report = []
    # Same data and step budget for all three sizes so the only variable is capacity.
    cfg = dict(n_train=300_000, n_test=30_000, steps=600 if quick else 8000, batch=512)
    for mult in (1, 10, 100):
        r = run(mult, **cfg)
        report.append(r)
        print(json.dumps(r), flush=True)
    json.dump(report, open(os.path.join(OUT, "baseline_report.json"), "w"), indent=1)
