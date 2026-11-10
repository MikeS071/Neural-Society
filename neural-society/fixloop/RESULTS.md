# Milestone 2 results: fix the failing test, on a local 4B model

Raw results: [`results/`](results/). Every run has its full transcript in the `.json`
files, and the `.md` files have the summary and per-task tables.

| | |
|---|---|
| planner | `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` (4-bit) via Ollama 0.40.16, native `/api/chat`, temperature 0, 16K context, 3,000-token reply cap (12,000 with thinking on) |
| machine | AMD Ryzen 9 5900HS (8 cores / 16 threads), 38 GB RAM, Pop!_OS, CPU only. The desktop session (browser, editor, Claude app) stayed open, so timings carry some noise. |
| tools | Python 3.12.3, ruff 0.17.4 (so the linter specialist was live in every society run) |
| suite | 31 injected-bug tasks x 2 arms, budget of 5 planner calls, success = visible **and** hidden tests pass |

Main result file: [`results/hf.co_unsloth_Qwen3.5-4B-GGUF-Q4_K_M-think-off.md`](results/hf.co_unsloth_Qwen3.5-4B-GGUF-Q4_K_M-think-off.md).

## Summary

Unless a row says otherwise, numbers are for thinking off.

| question | answer for this model |
|---|---|
| Does the society solve more tasks? | **No. It solves fewer:** 24/31 vs 28/31 for solo. Society alone solved 1 task and solo alone solved 5. With thinking on the gap almost closes (26 vs 27). |
| Is it cheaper when it works? | **Yes, modestly.** On the 23 tasks both arms solved, society used 32% fewer tokens (1,802 vs 2,653), fewer calls (1.09 vs 1.26) and 12% less LLM time (12.5 vs 14.2 s). |
| Which specialists visibly helped? | The **hang stack** (`graph-visited`, which only society solved) and, after a harness fix, the **compiler + localizer** on `matrix-syntax`. |
| What hurt? | Three things. Twice, society made an extra edit that contradicts the docstring, kept it once the visible tests passed, and failed the hidden tests. Twice, its replies ran to the token cap without an edit. Once, the localizer pointed at the validator that raised rather than the buggy caller. |
| Did the weak classifier mislead? | No visible harm. Its top-1 hint was right on 13/31 tasks, but no planner reply ever mentions it. |
| Can a specialist round replace reasoning time? | **Not here.** Thinking-off society (24/31, 38.5 s per run) is cheaper than thinking-on solo (27/31, 117 s) but solves fewer tasks. Plain thinking-off solo beat both (28/31, 24.6 s). Thinking helped society, not solo, and every thinking-on failure was a reasoning trace that hit the 12K-token cap. |

At this scale, structuring the tool output did not make a small model better at fixing
bugs. It made easy fixes cheaper and made a few hard ones fail. The rest of this file
backs that up task by task.

## Harness bugs found and fixed during the run

The first full thinking-off run (kept as
[`results/full1-prefix_qwen3.5-4b_think-off.*`](results/full1-prefix_qwen3.5-4b_think-off.md))
exposed two bugs in the society blackboard. Both handed the planner wrong facts on
`matrix-syntax`:

1. **The localizer ignored the error line in unparsable files.** With a `SyntaxError`
   the source cannot be parsed into functions, so the suspect fell back to
   `<module> (lines 1-15)`. It showed the planner unrelated code at "confidence 1.00",
   while the compiler said line 26. It now shows a window around the reported line.
2. **Progress tracking invented regressions.** "Passed last round" was computed as "not
   in last round's failing list". While the module failed to import, the only test was
   the import itself. After the import was fixed, `test_determinant` (which had never
   run) was reported as a **REGRESSION**. The planner then broke the file again chasing
   it, and the next round reported the same test as "newly passing". Progress
   now compares against the tests that actually passed or failed last round.

Both are fixed in commit `5ccf0a2` with regression tests (`tests/test_harness.py`;
both new tests fail on the old code). **Both arms were rerun.** The fix changes nothing
the solo arm sees, so the rerun doubles as a repeat for solo. Tasks, prompts and the
success criterion are unchanged. The only outcome that changed was `matrix-syntax`
for society: it went from failing after 5 calls to solved in 1.

Earlier, a cloud session fixed a third bug before these runs: `--think off` was ignored
through Ollama's OpenAI-compatible endpoint
([`results/smoke1_*`](results/smoke1_qwen3.5-4b_openai-endpoint_thinking-leaked.md)).
The native planner is confirmed working here:
[`results/smoke2_*`](results/smoke2_qwen3.5-4b_native-ollama_think-off.md) and both full
thinking-off runs record **0 hidden reasoning characters** and 0 s model load inside
timed calls.

## 1. Per arm, thinking off (after the fix)

| arm | success | visible pass | planner calls (all / successful) | input tok | output tok | tokens per success | LLM s / run | s / call | tool s / run | runs with unparseable replies | runs with rejected edits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| solo | **90.3%** (28) | 93.5% | 1.48 / 1.25 | 3,652 | 1,223 | **5,397** | **24.6** | 16.6 | 1.26 | 3 | 1 |
| society | 77.4% (24) | 90.3% | 1.55 / 1.12 | 4,622 | 1,870 | 8,386 | 38.5 | 24.9 | **0.62** | 5 | 1 |

- Generation speed was the same in both arms, about 50 output tokens/s. Prompt sizes
  are small, so a call costs mostly its output. Median call time was 5–6 s; the means
  are pulled up by replies that run to the 3,000-token cap (about 60 s each).
- The whole 62-run pass took 33 minutes of wall time. That is far quicker than the
  handover's "hours" estimate, because thinking was off and most tasks took one call.
- Specialists are cheap: society's 5 specialists cost about the same per round as
  solo's single test run (0.1–0.3 s; a hang costs 5.1 s in both arms). Society's lower
  mean tool time comes almost entirely from `graph-visited`, where solo hit the 5 s hang
  on all 5 rounds (30.5 s), while society solved it in 2 calls (10.2 s).
- The pre-fix run gave almost the same numbers (solo 28/31, society 23/31). The only
  society difference was `matrix-syntax`.

### On the 23 tasks both arms solved

| arm | calls | tokens | LLM s (mean / median) | solved in one call |
|---|---:|---:|---:|---:|
| solo | 1.26 | 2,653 | 14.2 / 5.7 | 19 |
| society | **1.09** | **1,802** | **12.5 / 5.4** | **21** |

When society works, it is the cheaper arm. It sends a smaller prompt (no test files,
1,165 vs 1,950 input tokens) and needs fewer second attempts. It was faster on 13 of
the 23 tasks.

## 2. Paired per-task comparison

`OK` = solved, `vis` = visible tests pass but hidden fail, `--` = visible still failing
after 5 calls. The number is planner calls.

| task | solo | society | |
|---|---|---|---|
| stats-two-bugs | OK 2 | vis 2 | **solo only** |
| intervals-contained | OK 1 | vis 2 | **solo only** |
| inventory-total-items | OK 1 | -- 5 | **solo only** |
| dates-year-wrap | OK 1 | -- 5 | **solo only** |
| ledger-helper-pad | OK 1 | -- 5 | **solo only** |
| graph-visited | -- 5 | OK 2 | **society only** |
| inventory-latent-delete (latent) | vis 1 | vis 1 | both fail |
| ledger-latent-plus (latent) | -- 5 | vis 1 | both fail |
| the other 23 | OK | OK | both |

Both thinking-off runs gave the same paired outcomes, apart from the fixed
`matrix-syntax`.

### Why society failed where solo succeeded

**A wrong extra edit kept after the visible tests passed (2 tasks).** Neither arm sees
the hidden tests, and both see the same source and docstrings. In both tasks, society
made a change that the docstring contradicts. It then got the visible suite green and
the loop stopped, so the bad edit was never revisited.
- `intervals-contained`: the bug drops `max(...)` when merging a contained interval.
  Society's first edit changed `start <= end + 1` to `start <= end`, so touching
  intervals no longer merge, although the docstring says "Merge overlapping or touching
  intervals". Its second edit fixed the real bug. The visible tests passed, but the
  first edit was still there, and the hidden `test_merge_touching` failed. Solo edited
  only the merge line.
- `stats-two-bugs`: society fixed the variance bug, then rewrote the percentile index as
  floor plus a clamp to the list length instead of fixing `s[rank]` back to
  `s[rank - 1]`. That passes the two visible cases but is not the "nearest-rank"
  percentile the docstring names, so the hidden edge cases failed. Solo's second
  attempt got it right.

Whether seeing the full test file would have stopped these edits is unclear: the
visible tests do not pin down touching intervals or those percentile edges either. It
looks more like a different first guess plus a loop that stops at green. Solo's first
guesses happened to be narrower.

**Runaway replies (2 tasks).** On `inventory-total-items` and `ledger-helper-pad`,
every society reply argued with itself ("Wait, maybe...") until it hit the 3,000-token
cap, then ended without the JSON edit. At temperature 0 the next reply was nearly
identical, so all 5 calls were wasted (about 5 minutes per run). Solo produced a short,
correct fix on the first call for both. The pattern is not unique to society: solo did
the same on `graph-visited`. But it is more frequent there. 5 of 31 society runs had a
runaway first reply against 2 for solo, and society's first replies average 3,100
characters against 1,900. One plausible reason is that without the test file the model
has to reconstruct how the code is used (for example, how `Inventory` stores items)
and does so by long speculation. That is a hypothesis; this data cannot test it.

**Localizer pointed at the validator (1 task).** In `dates-year-wrap`, `add_days` sets
`month = 0` on year wrap, and `days_in_month` then raises "bad month". The localizer
weights the innermost frame, so it ranked `days_in_month` first (share 0.8) and showed
its source, not that of `add_days`. The planner did find the right fix in its later
replies, but wrote `old` text with the wrong indentation (12 spaces, where the source has
8). The edit was rejected 4 times running with "matched 0 times". The localizer's
snippet showed the wrong function, so there was no line-numbered view of `add_days` to
copy from. Solo copied the right lines from the full source on its first try.

### Why society succeeded where solo failed

**`graph-visited` (hang).** The BFS never marks visited nodes. Solo's raw output just
says `test_shortest ... [command timed out after 5 s]`. Every solo reply speculated
about path reconstruction until the token cap and never emitted an edit. Society's
blackboard gives the hang stack (`src/graph.py:26 in shortest_path`) and the function's
source, so its second reply produced a working edit.

## 3. Which specialists visibly helped

| specialist | evidence |
|---|---|
| test runner, hang stack | Decisive on `graph-visited`, which only society solved. On `bsearch-hang` both arms solved it in one call: the infinite loop is easy to see in a 6-line function. |
| compiler + localizer | After the fix, `matrix-syntax` was solved in 1 call. Before the fix, the misleading localizer snippet and fake regression made it fail. Solo solved it in 1–2 calls from the raw traceback, so the gain is small. |
| localizer on helper bugs | Mixed. On `textutil-palindrome-name` (helper bug) both arms needed one call. On `ledger-helper-pad` the traceback has no source frame for the helper (the wrong value is returned, nothing raises). The localizer fell back to the functions named on the test line (`balance`, `statement`), not `parse_amount`, and society never found the bug. On `dates-year-wrap` it pointed at the raising validator instead of the caller. |
| linter | No visible effect. It was clean on every task except the syntax error, where it duplicated the compiler. |
| failure classifier | See below. |

### Did the weak classifier mislead?

Its round-1 top hint was right on **13/31** tasks (42%, as in the README). Wrong hints
included "arithmetic 1.00" on `matrix-det-sign` (truth: value) and "condition 0.87" on
`textutil-truncate`. Society solved both in one call anyway. No society transcript
reasons from the bug class: the planner works from the failing test, the trace and the
code. The confident correct hints ("wrong_name 1.00" on `inventory-total-items`,
"syntax 0.98" on `matrix-syntax`) did not help either; the first of those ended in a
runaway reply. At this size the classifier is noise the planner ignores. It neither
helps nor hurts, and it costs a line of prompt.

## 4. Latent tasks

Neither arm solved either latent task, in either run.

- `inventory-latent-delete`: both arms fixed the visible bug in one call and stopped.
  The visible suite passed, so the loop never ran again. Nothing showed them the second
  bug (`remove` leaving a zero-quantity entry), and they did not look for it.
- `ledger-latent-plus`: society fixed the visible bug (comment lines start with `#`,
  not `//`) in one call and stopped the same way, leaving the cents formatting bug
  (`{cents % 100}` without `:02d`). Solo never got the visible tests to pass. In the
  first run it sent malformed JSON (a stray `}`) 5 times; in the second, `old` text that
  matched nothing, also 5 times. Both times every reply was word for word the same.

That is the expected result for a fix loop that stops when the visible tests pass.
Neither arm "proactively" audits code it was not told was broken. Catching latent bugs
would need a specialist that looks beyond the failing tests (for example, a mutation or
property-test specialist), not a better rendering of the same failures.

## 5. Thinking on vs thinking off

Result file: [`results/hf.co_unsloth_Qwen3.5-4B-GGUF-Q4_K_M-think-on.md`](results/hf.co_unsloth_Qwen3.5-4B-GGUF-Q4_K_M-think-on.md).
The pass took 2 h 52 min, against 33 min with thinking off.

| configuration | success | visible pass | calls / run | tokens per success | LLM s / run | s / call | total LLM time |
|---|---:|---:|---:|---:|---:|---:|---:|
| solo, thinking off | **28/31** | 29 | 1.48 | **5,397** | **24.6** | 16.6 | **12.7 min** |
| society, thinking off | 24/31 | 28 | 1.55 | 8,386 | 38.5 | 24.9 | 19.9 min |
| solo, thinking on | 27/31 | 29 | 1.39 | 9,069 | 117.1 | 84.4 | 60.5 min |
| society, thinking on | 26/31 | 27 | 1.77 | 16,322 | 213.1 | 120.1 | 110.1 min |

**Can a specialist round replace reasoning time?** The handover's question was
thinking-on solo vs thinking-off society. Thinking-off society is 3x cheaper in LLM time
(38.5 vs 117 s per run) but solves 3 fewer tasks (24 vs 27). The more striking result
is that **for this model, reasoning bought nothing for solo**. Thinking-off solo is the
best configuration on every metric: most tasks solved, fewest tokens, lowest time.
Neither thinking nor the specialists improved on the plain loop.

**Thinking did help society**, from 24 to 26 tasks. Society lost five tasks to solo with
thinking off. With thinking on it solved four of them (`stats-two-bugs`,
`intervals-contained`, `inventory-total-items`, `dates-year-wrap`). The fifth,
`ledger-helper-pad`, stayed unsolved, and solo lost it too. With reasoning, the arms end
up close: they differ on one task only.

| task | solo on | society on | note |
|---|---|---|---|
| graph-visited | OK 2 | -- 5 | reversed from thinking off, where only society solved it |
| roman-repeat | -- 5 | -- 5 | solved by both arms with thinking off |
| ledger-helper-pad | -- 5 | -- 5 | solo solved it with thinking off |
| ledger-latent-plus | vis 1 | -- 5 | latent; neither solves it |
| inventory-latent-delete | vis 1 | vis 1 | latent; neither solves it |

**Every thinking-on failure has the same cause: the reasoning trace runs to the 12,000-
token cap, so the visible reply is empty.** For example, `ledger-helper-pad` produced
180–250K characters of reasoning per run and five empty replies in both arms. A reply
that ends empty costs about 4 minutes. A run like that costs 15–20 minutes and yields
nothing. Society ran into the cap more often (25 empty or unparseable replies against
13), and its runs reasoned twice as much in total (1.05M against 0.53M characters). The
richer blackboard gave the model more to think about, not less. Several runs did
recover: after an empty reply, the next call often produced a short, correct edit
(`stats-two-bugs` solo, `bsearch-upper` and `matrix-det-sign` society).

Hidden reasoning was recorded per run (`thinking_chars`), so this pass also confirms
that the think switch works both ways through the native planner: 0 characters with
thinking off, 0.5–1M with it on.

## 6. Determinism

The planner runs at temperature 0, but the solo arm is not reproducible. Its first
reply was word for word identical between the two thinking-off runs in only 5/31
tasks, against 26/31 for society. The likely cause is that raw `unittest` output
contains timings (`Ran 5 tests in 0.002s`), so the solo prompt differs slightly between
runs. Despite that, solo's per-task success was identical in both runs, while
iteration counts moved on 8 tasks (for example `dates-day-of-year` went from 1 to 4
calls). Treat iteration and token differences of a few tenths as noise. Success
outcomes look more stable, but this is two repeats, not a measurement of variance.

## Caveats

- **31 small tasks, single run per configuration.** A 4-task difference (28 vs 24) is
  suggestive, not significant. One repeat with a harness fix in between is not
  `--repeats 3`.
- **One small model.** A 4B model at 4-bit has its own failure modes (runaway replies,
  copy errors in `old` text). A stronger planner may use the blackboard differently,
  and the 1.7B and 9B options in the README were not run.
- **Arm design confounds two things.** Society gets structured findings *and* loses the
  test files. Its longer first replies may come from the missing test files. An
  ablation (society plus test files) is needed before blaming or crediting the
  blackboard itself.
- **The classifier** was trained on mutants of the same 10 programs, though never on
  these bugs.
- **CPU timings** depend on this machine (Ryzen 9 5900HS, 38 GB RAM) with a desktop
  session running. Use ratios, not absolute seconds.

## What to try next

1. **Society + test files** (an ablation): does it keep the token savings and stop the
   runaway replies?
2. **Ask for a check before stopping**: once the visible suite passes, show the planner
   its full diff once and ask whether every change was needed. That targets the two
   "kept a wrong extra edit" failures. It applies to both arms.
3. **Localizer improvements**: rank the caller above a function that only validates and
   raises, and when there is no source frame, follow the call graph from the test line
   (`balance` -> `parse_line` -> `parse_amount`).
4. **Runaway-reply guard** for both arms: when a reply has no edit, say so and ask for
   the JSON only; or lower the per-reply cap. This harness change would apply to both
   arms and would need a full rerun.
   With thinking on, the same problem shows up as an empty reply after a reasoning trace
   that hits the cap. A thinking budget, or one retry with thinking off after an empty
   reply, would likely recover most of the thinking-on failures.
5. `--repeats 3`, and the 1.7B and 9B models.
