"""Harness tests. Offline: no API key needed. Run from fixloop/: python3 -m unittest discover -s tests"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ns_fix.agents import solve  # noqa: E402
from ns_fix.blackboard import Blackboard  # noqa: E402
from ns_fix.llm import OllamaPlanner, OpenAICompatPlanner, ScriptedPlanner, oracle_planner, parse_edits  # noqa: E402
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

    def test_syntax_error_localizer_shows_error_line(self):
        # Unparsable file: the snippet must surround the compiler's line, not lines 1-15.
        bb = self.observe("matrix-syntax")
        line = bb.latest("compiler").data["errors"][0]["line"]
        top = bb.latest("localizer").data["suspects"][0]
        self.assertLessEqual(top["lines"][0], line)
        self.assertGreaterEqual(top["lines"][1], line)
        self.assertIn(f"{line:4d} | ", top["source"])

    def test_classifier(self):
        bb = self.observe("matrix-syntax")
        c = bb.latest("classifier")
        self.assertTrue(c.data["available"])
        self.assertIn(c.data["top"][0]["class"], CLASSES)
        self.assertEqual(c.data["top"][0]["class"], "syntax")
        self.assertEqual(len(features(None, None, None)), N_FEATURES)


class FakeOpenAIServer:
    """Minimal OpenAI-compatible endpoint that answers like the oracle planner."""

    def __init__(self, task):
        import http.server
        import threading

        oracle = oracle_planner(task)
        seen = self.requests = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                seen.append(body)
                text = (
                    '<think>maybe ```json\n{"edits": []}\n``` no...</think>\n' + oracle("", body["messages"][1:]).text
                )
                out = {
                    "choices": [{"message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 123, "completion_tokens": 45},
                }
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class FakeOllamaServer:
    """Minimal Ollama /api/chat: thinks (separate `thinking` field) unless told think=false."""

    def __init__(self, task):
        import http.server
        import threading

        oracle = oracle_planner(task)
        seen = self.requests = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                seen.append(body)
                out = {"message": {"role": "assistant", "content": ""}, "load_duration": 2_000_000_000}
                if body["messages"]:
                    out["message"]["content"] = oracle("", body["messages"][1:]).text
                    if body.get("think", True):
                        out["message"]["thinking"] = "let me think " * 50
                    out.update(prompt_eval_count=200, eval_count=30)
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TestLoop(unittest.TestCase):
    def test_ollama_native_planner(self):
        t = TASKS["roman-repeat"]
        srv = FakeOllamaServer(t)
        try:
            off = OllamaPlanner("m", srv.url + "/v1", think=False)  # a trailing /v1 is tolerated
            off.warmup()
            r_off = solve(t, "solo", off)
            r_on = solve(t, "solo", OllamaPlanner("m", srv.url, think=True))
        finally:
            srv.close()
        warm, req = srv.requests[0], srv.requests[1]
        self.assertEqual(warm["messages"], [])
        self.assertIs(req["think"], False)
        self.assertEqual(req["options"]["num_ctx"], 16384)
        self.assertFalse(req["stream"])
        self.assertTrue(r_off["success"])
        self.assertEqual((r_off["input_tokens"], r_off["output_tokens"], r_off["thinking_chars"]), (200, 30, 0))
        self.assertEqual(r_off["load_s"], 2.0)
        self.assertGreater(r_on["thinking_chars"], 0)

    def test_local_openai_compatible_planner(self):
        t = TASKS["roman-repeat"]
        srv = FakeOpenAIServer(t)
        try:
            r = solve(t, "society", OpenAICompatPlanner("local-model", srv.url, think=False), keep=True)
        finally:
            srv.close()
        self.assertTrue(r["success"])
        self.assertEqual((r["input_tokens"], r["output_tokens"]), (123, 45))
        req = srv.requests[0]
        self.assertEqual(req["model"], "local-model")
        self.assertEqual(req["messages"][0]["role"], "system")
        self.assertEqual(req["chat_template_kwargs"], {"enable_thinking": False})
        self.assertFalse(req["think"])
        self.assertNotIn("<think>", r["transcript"][1]["content"])  # reasoning stripped from history

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

    def test_progress_ignores_tests_that_did_not_run(self):
        # Replays the first move of the live matrix-syntax run: the import error is fixed but
        # minor() is still wrong, so test_determinant fails. It never ran in round 1 (the module
        # did not import), so it must not be reported as a regression.
        edit = {
            "file": "src/matrix.py",
            "old": "    return [r[:col] + r[col + 1: for i, r in enumerate(m) if i != row]",
            "new": "    return [[r[:col] + r[col+1:] for i, r in enumerate(m) if i != row]]",
        }
        reply = "```json\n" + json.dumps({"edits": [edit]}) + "\n```"
        r = solve(TASKS["matrix-syntax"], "society", ScriptedPlanner([reply]), max_iters=2, keep=True)
        round2 = r["transcript"][2]["content"]
        self.assertIn("test_determinant", round2)
        self.assertNotIn("REGRESSIONS", round2)
        self.assertNotIn("newly passing", round2)
        # The same edit again changes nothing: still no spurious progress lines.
        round3 = r["transcript"][4]["content"]
        self.assertNotIn("REGRESSIONS", round3)

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
