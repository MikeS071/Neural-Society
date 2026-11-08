"""The two arms of the experiment.

solo    : the large model alone. Each turn it sees the source, the tests and the raw
          `python -m unittest -v` output, exactly what a developer would paste.
society : the same model as planner, plus fast specialists (compiler, linter, test
          runner, localizer, failure classifier) that write structured findings to a
          blackboard. Each turn it sees the source and the rendered blackboard.

Same model, same edit format, same iteration budget, same success criterion
(visible suite passes, then hidden tests are run for the score).
"""

import time

from .blackboard import Blackboard, new_pool, run_specialists
from .llm import Usage, parse_edits
from .specialists import ROUND_ONE, ROUND_TWO
from .tasks import Workspace

MAX_FIELD = 8000

SYSTEM = """You are fixing bugs in a small Python project. Source code is in src/, unit tests in tests/.
The test suite currently fails. Edit files under src/ so that all tests pass. Never edit tests.
Fix the underlying bug: the code must be correct in general, not just for the tested inputs
(hidden tests will check this). A project may contain more than one bug.

Reply with brief reasoning (a few sentences), then exactly one JSON block:
```json
{"edits": [{"file": "src/name.py", "old": "exact text to replace", "new": "replacement text"}]}
```
Rules for edits: "old" must be copied exactly from the current file (including indentation) and
must occur exactly once in it; keep edits small. To rewrite a whole file instead, use
{"file": "src/name.py", "content": "<entire new file>"}."""


def _clip(s, n=MAX_FIELD):
    if len(s) <= n:
        return s
    return s[: n // 2] + f"\n... [{len(s) - n} characters omitted] ...\n" + s[-n // 2 :]


def _files_block(files):
    return "\n\n".join(f"### {rel}\n```python\n{text.rstrip()}\n```" for rel, text in files.items())


def _render_blackboard(bb, prev_failing):
    """Compact, structured view of this round's specialist reports for the planner."""
    lines = [f"## Blackboard, round {bb.round}"]
    comp, lint, tests = bb.latest("compiler"), bb.latest("linter"), bb.latest("tests")
    loc, cls = bb.latest("localizer"), bb.latest("classifier")
    if comp:
        if comp.ok:
            lines.append(f"- compiler: OK ({comp.data['files']} files)")
        for e in comp.data["errors"]:
            lines.append(f"- compiler: SyntaxError in {e['file']}:{e['line']}: {e['msg']}\n    {e['code']}")
    if lint and lint.data.get("available"):
        fs = lint.data["findings"]
        lines.append(f"- linter: {'clean' if not fs else str(len(fs)) + ' finding(s)'}")
        for f in fs[:8]:
            lines.append(f"    {f['file']}:{f['line']} {f['code']} {f['msg']}")
    if tests:
        d = tests.data
        if d["timeout"]:
            lines.append("- tests: TIMED OUT (likely an infinite loop). Stack at timeout, innermost first:")
            for f in (d["hang"] or {}).get("frames", [])[:6]:
                lines.append(f"    {f['file']}:{f['line']} in {f['func']}")
        else:
            lines.append(f"- tests: {len(d['failing'])} of {d['total']} failing")
        now = {t["test"] for t in d["failing"]}
        if prev_failing is not None:
            fixed, broke = prev_failing - now, now - prev_failing
            if fixed:
                lines.append(f"    newly passing: {', '.join(sorted(fixed))}")
            if broke:
                lines.append(f"    REGRESSIONS (passed last round, fail now): {', '.join(sorted(broke))}")
        for t in d["failing"][:10]:
            lines.append(f"  * {t['test']} [{t['exc_type'] or t['kind']}] {t['message'][:300]}")
            if t["test_line"]:
                lines.append(f"      test line {t['test_line']['line']}: {t['test_line']['code']}")
            for f in t["src_frames"][-2:]:
                lines.append(f"      at {f['file']}:{f['line']} in {f['func']}: {f['code']}")
    if loc and loc.data.get("suspects"):
        lines.append(f"- localizer (confidence {loc.confidence:.2f}): most likely location(s)")
        for s in loc.data["suspects"]:
            lines.append(f"    {s['file']} :: {s['func']} (lines {s['lines'][0]}-{s['lines'][1]}, share {s['share']})")
        lines.append("```\n" + loc.data["suspects"][0]["source"] + "\n```")
    if cls and cls.data.get("available") and tests and not tests.ok:
        top = ", ".join(f"{c['class']} {c['p']:.2f}" for c in cls.data["top"][:2])
        hint = "treat as a weak hint" if cls.confidence < 0.7 else "fairly confident"
        lines.append(f"- failure classifier (learned, {hint}): likely bug class: {top}")
    return "\n".join(lines)


def solve(task, arm, planner, max_iters=5, pool=None, keep=False):
    """Run one arm on one task. Returns a result dict."""
    assert arm in ("solo", "society")
    ws = Workspace.create(task)
    usage = Usage()
    tool_ms = 0.0
    messages = []
    own_pool = pool is None
    pool = pool or new_pool()
    bb = Blackboard()
    specialists = [cls() for cls in ROUND_ONE + ROUND_TWO]
    prev_failing = None

    def observe():
        nonlocal tool_ms, prev_failing
        t = time.perf_counter()
        if arm == "solo":
            ok, raw = ws.run_raw()
            tool_ms += (time.perf_counter() - t) * 1e3
            return ok, "## Output of `python -m unittest discover -s tests -v`\n```\n" + _clip(raw.strip()) + "\n```"
        bb.round += 1
        tool_ms += run_specialists(specialists, ws, bb, pool)
        tests = bb.latest("tests")
        text = _clip(_render_blackboard(bb, prev_failing))
        prev_failing = {x["test"] for x in tests.data["failing"]}
        return tests.ok, text

    t_start = time.perf_counter()
    try:
        ok, obs = observe()
        intro = "# Task\nThe test suite fails. Fix the bugs in src/.\n\n# Source files\n" + _files_block(
            ws.files("src")
        )
        if arm == "solo":
            intro += "\n\n# Test files\n" + _files_block(ws.files("tests"))
        messages.append({"role": "user", "content": intro + "\n\n" + obs})
        iters, parse_failures, apply_failures = 0, 0, 0
        while not ok and iters < max_iters:
            iters += 1
            before = ws.files("src")
            reply = planner(SYSTEM, messages)
            usage.add(reply.usage)
            messages.append({"role": "assistant", "content": reply.text or "(empty reply)"})
            edits, err = parse_edits(reply.text)
            errors = [err] if err else (ws.apply_edits(edits) if edits else ["your reply contained no edits"])
            parse_failures += bool(err)
            apply_failures += bool(errors and not err)
            ok, obs = observe()
            if ok:
                break
            after = ws.files("src")
            changed = {k: v for k, v in after.items() if before.get(k) != v}
            fb = ["# Result of your edits"]
            fb.append("Edits were NOT applied:\n" + "\n".join(f"- {e}" for e in errors) if errors else "Edits applied.")
            if changed:
                fb.append("## Current content of changed files\n" + _files_block(changed))
            fb.append(obs)
            messages.append({"role": "user", "content": "\n\n".join(fb)})
        hidden_ok = ws.passes("hidden") if ok else False
        return {
            "task": task.id,
            "arm": arm,
            "visible_pass": ok,
            "hidden_pass": hidden_ok,
            "success": ok and hidden_ok,
            "iterations": iters,
            "llm_calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "llm_s": round(usage.seconds, 2),
            "tool_ms": round(tool_ms, 1),
            "wall_s": round(time.perf_counter() - t_start, 2),
            "parse_failures": parse_failures,
            "apply_failures": apply_failures,
            "latent": task.latent,
            "categories": task.categories,
            "transcript": messages if keep else None,
        }
    finally:
        ws.cleanup()
        if own_pool:
            pool.shutdown()
