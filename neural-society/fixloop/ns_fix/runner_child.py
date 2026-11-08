"""Run a unittest suite and print one JSON document describing every test.

Executed in a subprocess with cwd = workspace: python runner_child.py TEST_DIR SRC_DIR TIMEOUT
If the suite hangs, faulthandler dumps every thread's stack and exits, so even a
timeout says *where* the code was stuck.
"""

import faulthandler
import io
import json
import os
import sys
import traceback
import unittest


def frames(tb, root):
    out = []
    for f in traceback.extract_tb(tb):
        path = os.path.relpath(f.filename, root)
        if path.startswith("..") or "unittest" in f.filename.split(os.sep)[-2:]:
            continue
        out.append({"file": path, "line": f.lineno, "func": f.name, "code": (f.line or "").strip()})
    return out


class Collector(unittest.TestResult):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.records = []

    def _rec(self, test, status, err=None):
        r = {"test": test.id(), "status": status}
        if err is not None:
            etype, evalue, tb = err
            r["exc_type"] = etype.__name__
            r["message"] = str(evalue)[:600]
            r["frames"] = frames(tb, self.root)
        self.records.append(r)

    def addSuccess(self, test):
        self._rec(test, "pass")

    def addFailure(self, test, err):
        self._rec(test, "fail", err)

    def addError(self, test, err):
        self._rec(test, "error", err)

    def addSkip(self, test, reason):
        self._rec(test, "skip")


def main():
    test_dir, src_dir, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
    root = os.getcwd()
    faulthandler.dump_traceback_later(timeout, exit=True)
    sys.path.insert(0, os.path.abspath(src_dir))
    sys.setrecursionlimit(2000)
    loader = unittest.TestLoader()
    collector = Collector(root)
    captured = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = captured
    try:
        suite = loader.discover(test_dir, top_level_dir=test_dir)
        suite.run(collector)
    finally:
        sys.stdout = real_stdout
    # discovery errors show up as failed "_FailedTest" cases with ImportError/SyntaxError
    print(json.dumps({"tests": collector.records}))


if __name__ == "__main__":
    main()
