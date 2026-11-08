"""Fast specialists. Each reads the workspace (and possibly earlier blackboard entries)
and writes one structured Report: `run(ws, bb) -> Report(data, confidence)`, the same
`forward(inputs) -> (output, confidence)` contract as the arithmetic specialists.
"""

import ast
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass

from .mutate import CLASSES

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "classifier.bin")


@dataclass
class Report:
    specialist: str
    ok: bool
    data: dict
    confidence: float = 1.0
    ms: float = 0.0


class Specialist:
    name = "base"
    needs = ()  # blackboard entry kinds this specialist reads

    def run(self, ws, bb):
        raise NotImplementedError

    def __call__(self, ws, bb):
        t = time.perf_counter()
        r = self.run(ws, bb)
        r.ms = (time.perf_counter() - t) * 1e3
        return r


# --------------------------------------------------------------------- compiler
class Compiler(Specialist):
    """Byte-compiles every source file in-process (no code is executed)."""

    name = "compiler"

    def run(self, ws, bb):
        errors = []
        files = ws.files("src")
        for rel, text in files.items():
            try:
                compile(text, rel, "exec")
            except SyntaxError as e:
                errors.append({"file": rel, "line": e.lineno, "msg": e.msg, "code": (e.text or "").rstrip()})
        return Report(self.name, not errors, {"files": len(files), "errors": errors})


# --------------------------------------------------------------------- linter
class Linter(Specialist):
    """ruff (pyflakes + pycodestyle errors + bugbear subset) on src/. Optional."""

    name = "linter"
    RULES = "F,E9,B006,B008,B015,B018,PLE"

    def run(self, ws, bb):
        ruff = shutil.which("ruff")
        if not ruff:
            return Report(self.name, True, {"available": False, "findings": []}, confidence=0.0)
        p = subprocess.run(
            [ruff, "check", "--no-cache", "--output-format", "json", "--select", self.RULES, "src"],
            cwd=ws.path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        try:
            raw = json.loads(p.stdout or "[]")
        except json.JSONDecodeError:
            raw = []
        findings = [
            {
                "file": os.path.relpath(f["filename"], ws.path),
                "line": f["location"]["row"],
                "code": f.get("code") or "E999",
                "msg": f["message"],
            }
            for f in raw
        ]
        # unused imports in tiny modules are noise for this task
        findings = [f for f in findings if f["code"] not in ("F401",)]
        return Report(self.name, not findings, {"available": True, "findings": findings})


# --------------------------------------------------------------------- test runner
_FRAME_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+) in (?P<func>\S+)')


class TestRunner(Specialist):
    """Runs the visible suite through the JSON runner; on a hang, reports where it hung."""

    name = "tests"

    def run(self, ws, bb):
        r = ws.run_structured("tests")
        tests = r["tests"]
        failing = []
        for t in tests:
            if t["status"] in ("fail", "error"):
                src_frames = [f for f in t.get("frames", []) if f["file"].startswith("src/")]
                test_frames = [f for f in t.get("frames", []) if f["file"].startswith("tests/")]
                msg = t.get("message", "")
                if t["test"].startswith("unittest.loader._FailedTest"):
                    msg = msg.strip().splitlines()[-1] if msg.strip() else msg
                failing.append(
                    {
                        "test": t["test"].split(".")[-1],
                        "id": t["test"],
                        "kind": t["status"],
                        "exc_type": t.get("exc_type", ""),
                        "message": msg[:400],
                        "src_frames": src_frames[-3:],
                        "test_line": test_frames[-1] if test_frames else None,
                    }
                )
        hang = None
        if r.get("timeout"):
            frames = []
            root = ws.path + os.sep
            for m in _FRAME_RE.finditer(r.get("stack", "")):
                f = m.group("file")
                if f.startswith(root):
                    frames.append({"file": f[len(root) :], "line": int(m.group("line")), "func": m.group("func")})
            hang = {"frames": frames}  # most recent call first
        data = {
            "total": len(tests),
            "failing": failing,
            "passed": sum(t["status"] == "pass" for t in tests),
            "timeout": bool(r.get("timeout")),
            "hang": hang,
        }
        ok = not r.get("timeout") and bool(tests) and not failing
        return Report(self.name, ok, data)


# --------------------------------------------------------------------- localizer
def _functions(source):
    """name -> (start, end) line span for every function / method (qualified Class.method)."""
    out = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + child.name] = (child.lineno, child.end_lineno)
                visit(child, prefix + child.name + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")

    visit(tree)
    return out


class Localizer(Specialist):
    """Ranks suspect functions: deepest source frame of each failure / the hang stack,
    or for pure assertion failures, the source functions named on the failing test line."""

    name = "localizer"
    needs = ("tests", "compiler")

    def run(self, ws, bb):
        tests, comp = bb.latest("tests"), bb.latest("compiler")
        votes = {}

        def vote(file, func, w=1.0):
            votes[(file, func)] = votes.get((file, func), 0.0) + w

        files = ws.files("src")
        spans = {rel: _functions(text) for rel, text in files.items()}

        def func_at(rel, line):
            best = None
            for name, (a, b) in spans.get(rel, {}).items():
                if a <= line <= b and (best is None or a >= spans[rel][best][0]):
                    best = name
            return best

        for e in comp.data["errors"] if comp else []:
            vote(e["file"], func_at(e["file"], e["line"] or 0) or "<module>", 3.0)
        if tests:
            if tests.data["hang"]:
                for depth, f in enumerate(x for x in tests.data["hang"]["frames"] if x["file"].startswith("src/")):
                    vote(f["file"], func_at(f["file"], f["line"]) or f["func"], 2.0 / (depth + 1))
            for t in tests.data["failing"]:
                if t["src_frames"]:
                    f = t["src_frames"][-1]
                    vote(f["file"], func_at(f["file"], f["line"]) or f["func"], 1.0)
                    for f in t["src_frames"][:-1]:
                        vote(f["file"], func_at(f["file"], f["line"]) or f["func"], 0.25)
                elif t["test_line"]:
                    called = set(re.findall(r"([A-Za-z_]\w*)\s*\(", t["test_line"].get("code", "")))
                    called |= set(re.findall(r"\.([A-Za-z_]\w*)\s*\(", t["test_line"].get("code", "")))
                    hits = [(rel, n) for rel, fs in spans.items() for n in fs if n.split(".")[-1] in called]
                    for rel, n in hits:
                        vote(rel, n, 1.0 / len(hits))
        if not votes:
            return Report(self.name, False, {"suspects": []}, confidence=0.0)
        total = sum(votes.values())
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
        suspects = []
        for (rel, func), w in ranked:
            lines = files.get(rel, "").splitlines()
            a, b = spans.get(rel, {}).get(func, (1, min(len(lines), 15)))
            snippet = "\n".join(f"{i:4d} | {lines[i - 1]}" for i in range(a, min(b, len(lines)) + 1))
            suspects.append(
                {"file": rel, "func": func, "lines": [a, b], "share": round(w / total, 2), "source": snippet}
            )
        return Report(self.name, True, {"suspects": suspects}, confidence=suspects[0]["share"])


# --------------------------------------------------------------------- failure classifier
EXC_VOCAB = [
    "AssertionError",
    "TypeError",
    "NameError",
    "IndexError",
    "KeyError",
    "ValueError",
    "AttributeError",
    "ZeroDivisionError",
    "RecursionError",
    "UnboundLocalError",
    "ImportError",
    "StopIteration",
    "Other",
]
N_HASH = 32
_NUM = r"-?\d+(?:\.\d+)?"


def _h(token):
    h = 2166136261
    for c in token.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return h % N_HASH


def features(tests, compiler, linter):
    """Fixed-length feature vector from the three first-round specialists' reports."""
    x = []
    failing = tests["failing"] if tests else []
    total = max(1, tests["total"] if tests else 1)
    counts = [0.0] * len(EXC_VOCAB)
    for t in failing:
        e = t["exc_type"] if t["exc_type"] in EXC_VOCAB else "Other"
        counts[EXC_VOCAB.index(e)] += 1
    n = max(1, len(failing))
    x += [c / n for c in counts]
    x += [float(bool(tests and tests["timeout"])), float(bool(compiler and compiler["errors"])), len(failing) / total]
    codes = [f["code"] for f in (linter or {}).get("findings", [])]
    x += [
        float(any(c == "F821" for c in codes)),
        float(any(c.startswith("F") and c != "F821" for c in codes)),
        float(any(c.startswith("E9") for c in codes)),
    ]
    # assertion-message shape
    flags = dict(
        diff1=0,
        sign=0,
        zero=0,
        bigger=0,
        smaller=0,
        listlen1=0,
        lists=0,
        boolean=0,
        none=0,
        strings=0,
        has_src_frame=0,
        raised_expected=0,
        float_vs_int=0,
    )
    toks = [0.0] * N_HASH
    for t in failing:
        msg = t["message"]
        if t["src_frames"]:
            flags["has_src_frame"] = 1
            code = t["src_frames"][-1].get("code", "")
            for tok in re.findall(r"[A-Za-z_]+|[<>=!]=?|//|\*\*|[-+*/%\[\]]", code):
                toks[_h("c:" + tok)] += 1
        m = re.match(rf"^({_NUM}) != ({_NUM})", msg)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            flags["diff1"] |= abs(a - b) == 1
            flags["sign"] |= a == -b and a != 0
            flags["zero"] |= a == 0 or b == 0
            flags["bigger"] |= a > b
            flags["smaller"] |= a < b
            flags["float_vs_int"] |= ("." in m.group(1)) != ("." in m.group(2))
        if "differ" in msg:
            flags["lists"] = 1
            la = re.search(r"First list contains (\d+) additional", msg) or re.search(
                r"Second list contains (\d+) additional", msg
            )
            flags["listlen1"] |= bool(la and la.group(1) == "1")
        flags["boolean"] |= "True" in msg or "False" in msg
        flags["none"] |= "None" in msg
        flags["strings"] |= msg.startswith("'")
        flags["raised_expected"] |= "not raised" in msg
        for tok in re.findall(r"[A-Za-z_]+", msg)[:20]:
            toks[_h("m:" + tok.lower())] += 1
    x += [float(v) for v in flags.values()]
    s = sum(toks) or 1.0
    x += [v / s for v in toks]
    return x


N_FEATURES = len(EXC_VOCAB) + 3 + 3 + 13 + N_HASH


def load_mlp(path=WEIGHTS):
    """Read an NSOC file (same format as the arithmetic runtime) into nested lists."""
    with open(path, "rb") as f:
        b = f.read()
    assert b[:4] == b"NSOC", "bad magic"
    (n,), pos, layers = struct.unpack_from("<I", b, 4), 8, []
    for _ in range(n):
        i, o = struct.unpack_from("<II", b, pos)
        pos += 8
        w = struct.unpack_from(f"<{i * o}f", b, pos)
        pos += 4 * i * o
        bias = struct.unpack_from(f"<{o}f", b, pos)
        pos += 4 * o
        layers.append(([w[r * i : (r + 1) * i] for r in range(o)], bias))
    return layers


def mlp_forward(layers, x):
    h = x
    for k, (w, b) in enumerate(layers):
        h = [bj + sum(wi * hi for wi, hi in zip(row, h)) for row, bj in zip(w, b)]
        if k < len(layers) - 1:
            h = [max(0.0, v) for v in h]
    m = max(h)
    e = [math.exp(v - m) for v in h]
    s = sum(e)
    return [v / s for v in e]


class FailureClassifier(Specialist):
    """Tiny learned MLP: failure signature -> likely bug class, with softmax confidence."""

    name = "classifier"
    needs = ("tests", "compiler", "linter")

    def __init__(self, path=WEIGHTS):
        self.layers = load_mlp(path) if os.path.exists(path) else None

    def run(self, ws, bb):
        if self.layers is None:
            return Report(self.name, False, {"available": False}, confidence=0.0)
        t, c, lint = bb.latest("tests"), bb.latest("compiler"), bb.latest("linter")
        x = features(t.data if t else None, c.data if c else None, lint.data if lint else None)
        p = mlp_forward(self.layers, x)
        order = sorted(range(len(p)), key=lambda i: -p[i])
        top = [{"class": CLASSES[i], "p": round(p[i], 3)} for i in order[:3]]
        return Report(self.name, True, {"available": True, "top": top}, confidence=p[order[0]])


ROUND_ONE = (Compiler, Linter, TestRunner)
ROUND_TWO = (Localizer, FailureClassifier)
