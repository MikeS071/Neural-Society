"""Task suite: reference programs + injected bugs, materialised into temp workspaces."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRAMS = os.path.join(ROOT, "programs")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner_child.py")
TEST_TIMEOUT = 5.0
# Edits often keep a file's size and land within the same second as the previous run,
# which would let Python reuse a stale .pyc. Never write bytecode in workspaces.
_ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


@dataclass
class Bug:
    file: str
    old: str
    new: str
    category: str


@dataclass
class Task:
    id: str
    program: str
    bugs: list
    latent: bool = False

    @property
    def categories(self):
        return sorted({b.category for b in self.bugs})


def load_tasks(path=None):
    path = path or os.path.join(ROOT, "tasks.json")
    raw = json.loads(read_text(path))["tasks"]
    return [Task(t["id"], t["program"], [Bug(**b) for b in t["bugs"]], t.get("latent", False)) for t in raw]


def read_text(path):
    with open(path) as f:
        return f.read()


def write_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def replace_once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        raise ValueError(f"{where}: expected exactly one match, found {n}")
    return text.replace(old, new)


@dataclass
class Workspace:
    """A writable copy of a buggy program. Hidden tests live outside it."""

    task: Task
    path: str
    hidden: str
    edits_applied: int = 0
    history: list = field(default_factory=list)

    @classmethod
    def create(cls, task, base_dir=None):
        path = tempfile.mkdtemp(prefix=f"nsfix-{task.id}-", dir=base_dir)
        prog = os.path.join(PROGRAMS, task.program)
        for sub in ("src", "tests"):
            shutil.copytree(os.path.join(prog, sub), os.path.join(path, sub))
        for b in task.bugs:
            fp = os.path.join(path, b.file)
            write_text(fp, replace_once(read_text(fp), b.old, b.new, f"{task.id}:{b.file}"))
        return cls(task, path, os.path.join(prog, "hidden"))

    def cleanup(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def files(self, sub="src"):
        out = {}
        for dirpath, _, names in os.walk(os.path.join(self.path, sub)):
            for n in sorted(names):
                if n.endswith(".py"):
                    full = os.path.join(dirpath, n)
                    out[os.path.relpath(full, self.path)] = read_text(full)
        return dict(sorted(out.items()))

    def read(self, rel):
        return read_text(self.safe(rel))

    def safe(self, rel):
        full = os.path.normpath(os.path.join(self.path, rel))
        if not full.startswith(self.path + os.sep):
            raise ValueError(f"path escapes workspace: {rel}")
        return full

    def apply_edits(self, edits):
        """Apply search/replace edits to src/ files. Returns a list of error strings
        (empty on success). All-or-nothing: on any error nothing is written."""
        staged, errors = {}, []
        for i, e in enumerate(edits):
            rel = e.get("file", "")
            if not rel.startswith("src/"):
                errors.append(f"edit {i}: only files under src/ may be edited (got {rel!r})")
                continue
            try:
                full = self.safe(rel)
            except ValueError as ex:
                errors.append(f"edit {i}: {ex}")
                continue
            text = staged.get(rel, read_text(full) if os.path.exists(full) else None)
            if "content" in e:
                staged[rel] = e["content"]
                continue
            if text is None:
                errors.append(f"edit {i}: {rel} does not exist")
                continue
            old, new = e.get("old", ""), e.get("new", "")
            n = text.count(old) if old else 0
            if n != 1:
                errors.append(f"edit {i}: 'old' text must match exactly once in {rel}, matched {n} times")
                continue
            staged[rel] = text.replace(old, new)
        if errors:
            return errors
        for rel, text in staged.items():
            write_text(self.safe(rel), text)
        self.edits_applied += len(edits)
        return []

    def run_structured(self, which="tests", timeout=TEST_TIMEOUT):
        """Run a suite with the JSON runner. which = 'tests' or 'hidden'."""
        test_dir = os.path.join(self.path, "tests") if which == "tests" else self.hidden
        try:
            p = subprocess.run(
                [sys.executable, RUNNER, test_dir, "src", str(timeout)],
                cwd=self.path,
                capture_output=True,
                text=True,
                timeout=timeout + 5,
                env=_ENV,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"timeout": True, "tests": [], "stack": ""}
        if p.returncode == 0 and p.stdout.strip().startswith("{"):
            out = json.loads(p.stdout.strip().splitlines()[-1])
            out["timeout"] = False
            return out
        # faulthandler fired (hang) or the runner itself crashed
        return {"timeout": "Timeout" in p.stderr or "Thread" in p.stderr, "tests": [], "stack": p.stderr[-4000:]}

    def run_raw(self, timeout=TEST_TIMEOUT):
        """What a developer sees: `python -m unittest` output, verbatim."""
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        env = dict(_ENV, PYTHONPATH=os.path.join(self.path, "src"))
        try:
            p = subprocess.run(
                cmd, cwd=self.path, capture_output=True, text=True, timeout=timeout, env=env, check=False
            )
            return p.returncode == 0, (p.stdout + p.stderr)
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            partial += (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return False, partial + f"\n[command timed out after {timeout:.0f} s]"

    def passes(self, which="tests"):
        r = self.run_structured(which)
        return (not r["timeout"]) and bool(r["tests"]) and all(t["status"] == "pass" for t in r["tests"])
