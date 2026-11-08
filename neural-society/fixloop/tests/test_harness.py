"""Harness tests. Offline: no API key needed. Run from fixloop/: python3 -m unittest discover -s tests"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ns_fix.agents import solve  # noqa: E402
from ns_fix.blackboard import Blackboard  # noqa: E402
from ns_fix.llm import ScriptedPlanner, oracle_planner, parse_edits  # noqa: E402
from ns_fix.mutate import CLASSES  # noqa: E402
from ns_fix.specialists import (  # noqa: E402
    N_FEATURES,
    Compiler,
    FailureClassifier,
    Linter,
    Localizer,
    TestRunner,
    features,
)
from ns_fix.tasks import Task, Workspace, load_tasks  # noqa: E402

TASKS = {t.id: t for t in load_tasks()}


class TestTasks(unittest.TestCase):
    def test_reference_programs_pass(self):
        for p in sorted({t.program for t in TASKS.values()}):
            ws = Workspace.create(Task("ref", p, []))
            try:
                self.assertTrue(ws.passes("tests"), p)
                self.assertTrue(ws.passes("hidden"), p)
            finally:
                ws.cleanup()

    def test_every_task_breaks_visible_suite(self):
        for t in TASKS.values():
            ws = Workspace.create(t)
            try:
                self.assertFalse(ws.passes("tests"), t.id)
            finally:
                ws.cleanup()


class TestEdits(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace.create(TASKS["textutil-palindrome-name"])

    def tearDown(self):
        self.ws.cleanup()

    def test_parse(self):
        e, err = parse_edits('blah\n```json\n{"edits": [{"file": "src/a.py", "old": "x", "new": "y"}]}\n```')
        self.assertIsNone(err)
        self.assertEqual(e[0]["new"], "y")
        e, err = parse_edits("no json here")
        self.assertEqual(e, [])
        self.assertIsNotNone(err)
        e, err = parse_edits('{"edits": [{"file": "src/a.py", "content": "{}"}]}')
        self.assertIsNone(err)

    def test_refuses_tests_and_escapes(self):
        self.assertTrue(self.ws.apply_edits([{"file": "tests/test_textutil.py", "old": "a", "new": "b"}]))
        self.assertTrue(self.ws.apply_edits([{"file": "src/../../etc/x.py", "content": ""}]))

    def test_all_or_nothing(self):
        before = self.ws.files("src")
        errs = self.ws.apply_edits(
            [
                {"file": "src/textutil.py", "old": "char[::-1]", "new": "chars[::-1]"},
                {"file": "src/textutil.py", "old": "does not exist", "new": ""},
            ]
        )
        self.assertEqual(len(errs), 1)
        self.assertEqual(self.ws.files("src"), before)

    def test_ambiguous_old_rejected(self):
        self.assertTrue(self.ws.apply_edits([{"file": "src/textutil.py", "old": "return", "new": "return"}]))


class TestSpecialists(unittest.TestCase):
    def observe(self, task_id):
        ws = Workspace.create(TASKS[task_id])
        bb = Blackboard()
        for s in (Compiler(), Linter(), TestRunner(), Localizer(), FailureClassifier()):
            bb.post(s.name, s(ws, bb))
        ws.cleanup()
        return bb

    def test_syntax_error(self):
        bb = self.observe("matrix-syntax")
        self.assertFalse(bb.latest("compiler").ok)
        self.assertEqual(bb.latest("compiler").data["errors"][0]["file"], "src/matrix.py")

    def test_hang_is_localized(self):
        bb = self.observe("bsearch-hang")
        self.assertTrue(bb.latest("tests").data["timeout"])
        self.assertEqual(bb.latest("localizer").data["suspects"][0]["func"], "lower_bound")

    def test_helper_bug_localized_from_traceback(self):
        bb = self.observe("textutil-palindrome-name")
        self.assertEqual(bb.latest("localizer").data["suspects"][0]["func"], "is_palindrome")

    def test_assertion_localized_from_test_line(self):
        bb = self.observe("roman-table")
        funcs = [s["func"] for s in bb.latest("localizer").data["suspects"]]
        self.assertIn("to_roman", funcs)

    def test_classifier(self):
        bb = self.observe("matrix-syntax")
        c = bb.latest("classifier")
        self.assertTrue(c.data["available"])
        self.assertIn(c.data["top"][0]["class"], CLASSES)
        self.assertEqual(c.data["top"][0]["class"], "syntax")
        self.assertEqual(len(features(None, None, None)), N_FEATURES)


class TestLoop(unittest.TestCase):
    def test_oracle_solves_in_one_call(self):
        for tid in ("bsearch-hang", "graph-visited", "ledger-helper-pad", "inventory-latent-delete"):
            for arm in ("solo", "society"):
                r = solve(TASKS[tid], arm, oracle_planner(TASKS[tid]))
                self.assertTrue(r["success"], (tid, arm))
                self.assertEqual(r["iterations"], 1)

    def test_useless_planner_exhausts_budget(self):
        r = solve(TASKS["stats-median-even"], "society", ScriptedPlanner(["I give up."]), max_iters=3)
        self.assertFalse(r["success"])
        self.assertEqual(r["iterations"], 3)
        self.assertEqual(r["parse_failures"], 3)

    def test_fixing_only_visible_bug_fails_latent_task(self):
        t = TASKS["inventory-latent-delete"]
        first = t.bugs[0]
        edit = {"file": first.file, "old": first.new, "new": first.old}
        reply = "```json\n" + json.dumps({"edits": [edit]}) + "\n```"
        r = solve(t, "solo", ScriptedPlanner([reply]), max_iters=1)
        self.assertTrue(r["visible_pass"])
        self.assertFalse(r["hidden_pass"])


if __name__ == "__main__":
    unittest.main()
