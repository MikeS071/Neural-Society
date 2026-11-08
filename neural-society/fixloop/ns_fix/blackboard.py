"""Blackboard: typed, append-only entries written by specialists and the planner,
plus a scheduler that runs each specialist as soon as the entries it needs exist,
in parallel (the "bus" for this milestone: coarse messages, milliseconds of work each).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class Entry:
    round: int
    kind: str  # specialist name, or "plan" / "apply"
    report: object  # specialists.Report, or a dict for planner entries


class Blackboard:
    def __init__(self):
        self.entries = []
        self.round = 0

    def post(self, kind, report):
        self.entries.append(Entry(self.round, kind, report))

    def latest(self, kind, this_round=True):
        for e in reversed(self.entries):
            if e.kind == kind and (not this_round or e.round == self.round):
                return e.report
        return None

    def history(self):
        return [e for e in self.entries if e.kind in ("plan", "apply", "tests")]


def run_specialists(specialists, ws, bb, pool, gate=None):
    """Run specialists in dependency waves. `gate(bb)` may stop early after a wave
    (e.g. skip the slow test run when the compiler already found a syntax error).
    Returns wall-clock ms."""
    t0 = time.perf_counter()
    pending = list(specialists)
    done = set()
    while pending:
        ready = [s for s in pending if all(n in done for n in s.needs)]
        if not ready:
            # dependency on a specialist that was skipped: run the rest anyway
            ready = pending
        futures = {pool.submit(s, ws, bb): s for s in ready}
        for f, s in futures.items():
            bb.post(s.name, f.result())
            done.add(s.name)
            pending.remove(s)
        if gate and gate(bb):
            break
    return (time.perf_counter() - t0) * 1e3


def new_pool(n=4):
    return ThreadPoolExecutor(max_workers=n)
