"""Run the fix-the-failing-test benchmark: solo vs society, same planner model.

Usage:
  python3 run_bench.py --model claude-sonnet-5-5 [--arms solo,society] [--repeats 1]
                       [--max-iters 5] [--tasks id1,id2] [--workers 4] [--planner anthropic|local|oracle]

  --planner anthropic  (default) Claude via the Messages API; needs ANTHROPIC_API_KEY.
  --planner local      any OpenAI-compatible server (Ollama, llama-server, LM Studio, vLLM), e.g.
                       --planner local --model qwen2.5-coder:7b --base-url http://localhost:11434/v1 --workers 1
  --planner oracle     reverts the injected bugs; only checks the harness end to end.
Writes results/<model>.json (every run, with transcripts) and results/<model>.md.
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from ns_fix.agents import solve
from ns_fix.blackboard import new_pool
from ns_fix.llm import AnthropicPlanner, OpenAICompatPlanner, oracle_planner
from ns_fix.tasks import load_tasks, write_text

HERE = os.path.dirname(os.path.abspath(__file__))


def summarize(runs, arms):
    def agg(rs):
        n = len(rs)
        succ = [r for r in rs if r["success"]]
        return {
            "runs": n,
            "success": sum(r["success"] for r in rs) / n,
            "visible_pass": sum(r["visible_pass"] for r in rs) / n,
            "iters_mean": statistics.mean(r["iterations"] for r in rs),
            "iters_success": statistics.mean(r["iterations"] for r in succ) if succ else float("nan"),
            "in_tok": statistics.mean(r["input_tokens"] for r in rs),
            "out_tok": statistics.mean(r["output_tokens"] for r in rs),
            "tok_per_success": sum(r["input_tokens"] + r["output_tokens"] for r in rs) / max(1, len(succ)),
            "llm_s": statistics.mean(r["llm_s"] for r in rs),
            "tool_s": statistics.mean(r["tool_ms"] for r in rs) / 1e3,
            "apply_fail": sum(r["apply_failures"] for r in rs),
            "parse_fail": sum(r["parse_failures"] for r in rs),
        }

    return {a: agg([r for r in runs if r["arm"] == a]) for a in arms}


def markdown(model, runs, arms, args):
    s = summarize(runs, arms)
    out = [
        f"# Fix-the-failing-test: `{model}`\n",
        f"{len({r['task'] for r in runs})} tasks x {args.repeats} repeat(s), max {args.max_iters} planner calls per run. "
        "Success = visible suite passes **and** hidden tests pass.\n",
        "| arm | success | visible pass | planner calls (all / successful) | input tok | output tok | tokens per success | LLM s | tool s | bad edits | unparseable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in arms:
        x = s[a]
        out.append(
            f"| {a} | {x['success']:.1%} | {x['visible_pass']:.1%} | {x['iters_mean']:.2f} / {x['iters_success']:.2f} | "
            f"{x['in_tok']:.0f} | {x['out_tok']:.0f} | {x['tok_per_success']:.0f} | {x['llm_s']:.1f} | "
            f"{x['tool_s']:.2f} | {x['apply_fail']} | {x['parse_fail']} |"
        )
    out += [
        "\n## Per task (successes / runs, mean planner calls)\n",
        "| task | categories | " + " | ".join(arms) + " |",
        "|---|---|" + "---|" * len(arms),
    ]
    for tid in sorted({r["task"] for r in runs}):
        cells = []
        for a in arms:
            rs = [r for r in runs if r["task"] == tid and r["arm"] == a]
            cells.append(
                f"{sum(r['success'] for r in rs)}/{len(rs)}, {statistics.mean(r['iterations'] for r in rs):.1f}"
                if rs
                else "—"
            )
        cats = ",".join(next(r["categories"] for r in runs if r["task"] == tid))
        latent = " (latent)" if next(r["latent"] for r in runs if r["task"] == tid) else ""
        out.append(f"| {tid}{latent} | {cats} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--arms", default="solo,society")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--planner", default="anthropic", choices=["anthropic", "local", "oracle"])
    ap.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="for --planner local: OpenAI-compatible endpoint (Ollama default; llama-server: http://localhost:8080/v1)",
    )
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    arms = args.arms.split(",")
    tasks = load_tasks()
    if args.tasks:
        keep = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.id in keep]
    jobs = [(t, a, k) for k in range(args.repeats) for t in tasks for a in arms]
    spec_pool = new_pool(8)

    def run(job):
        t, a, k = job
        if args.planner == "oracle":
            planner = oracle_planner(t)
        elif args.planner == "local":
            planner = OpenAICompatPlanner(args.model, args.base_url)
        else:
            planner = AnthropicPlanner(args.model)
        try:
            r = solve(t, a, planner, max_iters=args.max_iters, pool=spec_pool, keep=True)
        except Exception as e:  # keep the sweep going; record the failure
            r = {"task": t.id, "arm": a, "error": repr(e)}
        r["repeat"] = k
        flag = "ok " if r.get("success") else "-- "
        print(
            f"{flag}{t.id:28s} {a:8s} iters={r.get('iterations')} tok={r.get('input_tokens')}/{r.get('output_tokens')}"
            + (f" ERROR {r['error']}" if "error" in r else ""),
            flush=True,
        )
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(run, jobs))
    spec_pool.shutdown()
    errors = [r for r in runs if "error" in r]
    runs = [r for r in runs if "error" not in r]
    name = "oracle" if args.planner == "oracle" else args.model.replace("/", "_").replace(":", "-")
    os.makedirs(args.out, exist_ok=True)
    write_text(
        os.path.join(args.out, f"{name}.json"),
        json.dumps({"model": name, "args": vars(args), "errors": errors, "runs": runs}, indent=1),
    )
    md = markdown(name, runs, arms, args)
    if errors:
        md += f"\n{len(errors)} run(s) errored and are excluded: {sorted({e['error'][:120] for e in errors})}\n"
    write_text(os.path.join(args.out, f"{name}.md"), md)
    print(md)
    print(f"{len(runs)} runs in {time.time() - t0:.0f}s; wrote {args.out}/{name}.md")


if __name__ == "__main__":
    main()
