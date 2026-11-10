# Fix-the-failing-test: `hf.co_unsloth_Qwen3.5-4B-GGUF-Q4_K_M-think-off`

2 tasks x 1 repeat(s), max 5 planner calls per run. Success = visible suite passes **and** hidden tests pass.

| arm | success | visible pass | planner calls (all / successful) | input tok | output tok | tokens per success | LLM s | tool s | hidden reasoning chars | bad edits | unparseable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| solo | 100.0% | 100.0% | 1.00 / 1.00 | 943 | 476 | 1420 | 8.5 | 2.59 | 0 | 0 | 0 |
| society | 100.0% | 100.0% | 1.00 / 1.00 | 834 | 465 | 1299 | 8.4 | 2.59 | 0 | 0 | 0 |

## Per task (successes / runs, mean planner calls)

| task | categories | solo | society |
|---|---|---|---|
| bsearch-hang | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
| stats-median-even | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
