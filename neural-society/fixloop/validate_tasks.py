"""Check every reference program passes, and every task's bugs break the visible suite."""

import sys

from ns_fix.tasks import Task, Workspace, load_tasks

bad = 0
programs = sorted({t.program for t in load_tasks()})
for p in programs:
    ws = Workspace.create(Task(f"ref-{p}", p, []))
    ok = ws.passes("tests") and ws.passes("hidden")
    print(f"reference {p:10s} {'ok' if ok else 'FAILS'}")
    bad += not ok
    ws.cleanup()
for t in load_tasks():
    ws = Workspace.create(t)
    vis, hid = ws.passes("tests"), ws.passes("hidden")
    ok = not vis
    print(
        f"task {t.id:28s} visible={'pass' if vis else 'FAIL'} hidden={'pass' if hid else 'FAIL'} {'ok' if ok else '<-- bug not caught by visible tests'}"
    )
    bad += not ok
    ws.cleanup()
sys.exit(1 if bad else 0)
