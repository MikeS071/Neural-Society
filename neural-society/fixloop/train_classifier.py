"""Train the failure-classifier specialist.

1. Generate AST mutants of every reference program (ns_fix/mutate.py).
2. Run the real round-one specialists (compiler, linter, test runner) on each mutant
   that breaks the visible suite, and extract the classifier's feature vector.
3. Train a tiny MLP (train/nn.py, NumPy) and export it in the NSOC format.

Reports leave-one-program-out accuracy (generalisation to unseen programs) and accuracy
on the hand-written evaluation tasks, which are never used for training.

Usage: python3 train_classifier.py [--per-class N] [--reuse]
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from nn import MLP  # noqa: E402

from ns_fix.blackboard import Blackboard  # noqa: E402
from ns_fix.mutate import CLASSES, TASK_CATEGORY_TO_CLASS, mutants  # noqa: E402
from ns_fix.specialists import N_FEATURES, Compiler, Linter, TestRunner, features  # noqa: E402
from ns_fix.tasks import Task, Workspace, load_tasks, read_text, write_text  # noqa: E402

OUT = os.path.join(HERE, "weights")
DATA = os.path.join(OUT, "classifier_data.json")


def observe(ws):
    bb = Blackboard()
    reports = {s.name: s(ws, bb) for s in (Compiler(), Linter(), TestRunner())}
    return reports


def mutant_example(program, cls, desc, source):
    ws = Workspace.create(Task(f"mut-{program}", program, []))
    try:
        rel = f"src/{program}.py"
        write_text(ws.safe(rel), source)
        r = observe(ws)
        if r["tests"].ok:  # equivalent mutant: the suite does not notice it
            return None
        x = features(r["tests"].data, r["compiler"].data, r["linter"].data)
        return {"program": program, "label": cls, "desc": desc, "x": x}
    finally:
        ws.cleanup()


def build_dataset(per_class):
    rng = random.Random(0)
    jobs = []
    for program in sorted(os.listdir(os.path.join(HERE, "programs"))):
        src = read_text(os.path.join(HERE, "programs", program, "src", f"{program}.py"))
        for cls, desc, text in mutants(src, rng, per_class):
            jobs.append((program, cls, desc, text))
    print(f"{len(jobs)} mutants; running specialists on each...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        rows = [r for r in pool.map(lambda j: mutant_example(*j), jobs) if r]
    print(f"{len(rows)} mutants break the visible suite ({time.time() - t0:.0f}s)", flush=True)
    return rows


def task_examples():
    rows = []
    for t in load_tasks():
        ws = Workspace.create(t)
        try:
            r = observe(ws)
            # first bug's class = the one the visible failure is about
            label = TASK_CATEGORY_TO_CLASS[t.bugs[0].category]
            rows.append(
                {"task": t.id, "label": label, "x": features(r["tests"].data, r["compiler"].data, r["linter"].data)}
            )
        finally:
            ws.cleanup()
    return rows


def fit(X, y, steps=3000, hidden=(32,), seed=0):
    net = MLP([X.shape[1], *hidden, len(CLASSES)], seed=seed)
    # class-balanced sampling weights via repetition
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = 1.0 / np.maximum(counts[y], 1)
    p = w / w.sum()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.choice(len(y), size=min(256, len(y)), p=p)
        net.step(X[idx], [(0, len(CLASSES))], [y[idx]], lr=3e-3, wd=1e-4)
    return net


def accuracy(net, X, y):
    return float((net.logits(X).argmax(1) == y).mean()) if len(y) else float("nan")


if __name__ == "__main__":
    per_class = int(sys.argv[sys.argv.index("--per-class") + 1]) if "--per-class" in sys.argv else 25
    os.makedirs(OUT, exist_ok=True)
    if "--reuse" in sys.argv and os.path.exists(DATA):
        rows = json.loads(read_text(DATA))
    else:
        rows = build_dataset(per_class)
        write_text(DATA, json.dumps(rows))
    X = np.array([r["x"] for r in rows], np.float32)
    assert X.shape[1] == N_FEATURES, X.shape
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    progs = np.array([r["program"] for r in rows])
    print("class counts:", dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES)).tolist())))

    # leave-one-program-out
    lopo = []
    for p in sorted(set(progs)):
        tr, te = progs != p, progs == p
        lopo.append((p, accuracy(fit(X[tr], y[tr]), X[te], y[te]), int(te.sum())))
    lopo_acc = sum(a * n for _, a, n in lopo) / sum(n for *_, n in lopo)
    print("leave-one-program-out:", {p: round(a, 2) for p, a, _ in lopo}, f"overall {lopo_acc:.3f}")

    net = fit(X, y)
    net.save(os.path.join(OUT, "classifier.bin"))
    train_acc = accuracy(net, X, y)

    tasks = task_examples()
    Xt = np.array([r["x"] for r in tasks], np.float32)
    yt = np.array([CLASSES.index(r["label"]) for r in tasks])
    probs = net.logits(Xt)
    probs = np.exp(probs - probs.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    pred = probs.argmax(1)
    task_rows = [
        {
            "task": r["task"],
            "label": r["label"],
            "pred": CLASSES[k],
            "conf": round(float(probs[i, k]), 3),
            "top2": CLASSES[k] == r["label"] or CLASSES[int(np.argsort(-probs[i])[1])] == r["label"],
        }
        for i, (r, k) in enumerate(zip(tasks, pred))
    ]
    task_acc = float((pred == yt).mean())
    top2 = float(np.mean([r["top2"] for r in task_rows]))
    # majority-class and exception-rule baselines on the tasks
    majority = float((yt == np.bincount(y).argmax()).mean())
    conf_correct = [r["conf"] for r in task_rows if r["pred"] == r["label"]]
    conf_wrong = [r["conf"] for r in task_rows if r["pred"] != r["label"]]
    report = {
        "n_train": len(rows),
        "params": net.n_params(),
        "features": N_FEATURES,
        "train_acc": round(train_acc, 3),
        "lopo_acc": round(lopo_acc, 3),
        "lopo": {p: {"acc": round(a, 3), "n": n} for p, a, n in lopo},
        "task_acc": round(task_acc, 3),
        "task_top2": round(top2, 3),
        "task_majority_baseline": round(majority, 3),
        "task_mean_conf_correct": round(float(np.mean(conf_correct)), 3) if conf_correct else None,
        "task_mean_conf_wrong": round(float(np.mean(conf_wrong)), 3) if conf_wrong else None,
        "tasks": task_rows,
    }
    write_text(os.path.join(OUT, "classifier_report.json"), json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("tasks", "lopo")}, indent=1))
    for r in task_rows:
        print(f"  {r['task']:28s} {r['label']:14s} -> {r['pred']:14s} ({r['conf']:.2f})")
