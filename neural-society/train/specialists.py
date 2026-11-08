"""Train the specialist "neurons". Each learns ONE atomic digit operation.

Because each op has a tiny, finite input space, we can train on and verify against
EVERY possible input: a specialist is either provably 100% correct or we know exactly
which cases it gets wrong. That is the main payoff of fine-grained decomposition.

  add : (a, b, carry_in 0-1)   -> a+b+c        (20 classes)
  sub : (a, b, borrow_in 0-1)  -> a-b-borrow+10 (20 classes)
  mul : (a, b, carry_in 0-8)   -> a*b+c        (90 classes)
  cmp : (a, b)                 -> <, =, >      (3 classes)

Also trains 'mul_weak': a deliberately under-sized multiply specialist, used to show
how one imperfect neuron's errors compound through a long computation.
"""
import itertools, json, os, sys, time
import numpy as np
from nn import MLP, one_hot

OUT = os.path.join(os.path.dirname(__file__), "..", "weights")
os.makedirs(OUT, exist_ok=True)

SPECS = {
    "add": dict(fields=[10, 10, 2], classes=20, f=lambda a, b, c: a + b + c),
    "sub": dict(fields=[10, 10, 2], classes=20, f=lambda a, b, c: a - b - c + 10),
    "mul": dict(fields=[10, 10, 9], classes=90, f=lambda a, b, c: a * b + c),
    "cmp": dict(fields=[10, 10], classes=3, f=lambda a, b: int(a > b) - int(a < b) + 1),
}


def dataset(spec):
    vals = np.array(list(itertools.product(*[range(s) for s in spec["fields"]])))
    y = np.array([spec["f"](*v) for v in vals])
    return one_hot(vals, spec["fields"]), y


def train(name, spec, hidden, steps, seed=0, lr=3e-3):
    X, y = dataset(spec)
    net = MLP([X.shape[1], *hidden, spec["classes"]], seed=seed)
    t0 = time.time()
    for s in range(steps):
        loss = net.step(X, [(0, spec["classes"])], [y], lr=lr)
    acc = (net.logits(X).argmax(1) == y).mean()
    return net, dict(name=name, params=net.n_params(), cases=len(y), exhaustive_acc=float(acc),
                     loss=float(loss), train_s=round(time.time() - t0, 2), hidden=hidden)


def train_into_band(name, spec, hidden, lo, hi, seed=0, lr=1e-2, chunk=25, max_steps=3000):
    """Train until exhaustive accuracy first reaches [lo, hi]. Records the wrong cases."""
    X, y = dataset(spec)
    net = MLP([X.shape[1], *hidden, spec["classes"]], seed=seed)
    t0 = time.time()
    steps = 0
    while steps < max_steps:
        for _ in range(chunk):
            loss = net.step(X, [(0, spec["classes"])], [y], lr=lr)
        steps += chunk
        acc = (net.logits(X).argmax(1) == y).mean()
        if acc >= lo:
            break
    if not (lo <= acc <= hi):
        raise RuntimeError(f"{name}: accuracy {acc:.4f} overshot band [{lo}, {hi}]; lower chunk")
    vals = np.array(list(itertools.product(*[range(s) for s in spec["fields"]])))
    wrong = vals[net.logits(X).argmax(1) != y].tolist()
    return net, dict(name=name, params=net.n_params(), cases=len(y), exhaustive_acc=float(acc),
                     loss=float(loss), train_s=round(time.time() - t0, 2), hidden=hidden,
                     steps=steps, wrong_cases=wrong)


if __name__ == "__main__":
    report = []
    for name, spec in SPECS.items():
        net, r = train(name, spec, hidden=[64, 64], steps=1500)
        net.save(os.path.join(OUT, f"{name}.bin"))
        report.append(r)
        print(json.dumps(r))
    # Deliberately weak multiplier for the error-compounding experiment.
    # Target: 95-99% exhaustive accuracy. Hidden [12] at lr 1e-2 crosses that band
    # around 500-700 steps (250 steps -> 73%, 900 -> 99.8%), so train in chunks and
    # stop as soon as accuracy lands in the band. Deterministic given the seed.
    net, r = train_into_band("mul_weak", SPECS["mul"], hidden=[12], lo=0.96, hi=0.99,
                             seed=1, lr=1e-2)
    net.save(os.path.join(OUT, "mul_weak.bin"))
    report.append(r)
    print(json.dumps(r))
    json.dump(report, open(os.path.join(OUT, "specialists_report.json"), "w"), indent=1)
