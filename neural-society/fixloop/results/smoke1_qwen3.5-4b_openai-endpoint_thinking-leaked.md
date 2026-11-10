# Fix-the-failing-test: `hf.co_unsloth_Qwen3.5-4B-GGUF-Q4_K_M-think-off`

2 tasks x 1 repeat(s), max 5 planner calls per run. Success = visible suite passes **and** hidden tests pass.

| arm | success | visible pass | planner calls (all / successful) | input tok | output tok | tokens per success | LLM s | tool s | bad edits | unparseable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| solo | 100.0% | 100.0% | 1.50 / 1.50 | 1359 | 2710 | 4068 | 87.9 | 5.08 | 0 | 1 |
| society | 100.0% | 100.0% | 1.00 / 1.00 | 826 | 1055 | 1881 | 16.3 | 2.61 | 0 | 0 |

## Per task (successes / runs, mean planner calls)

| task | categories | solo | society |
|---|---|---|---|
| bsearch-hang | off_by_one | 1/1, 2.0 | 1/1, 1.0 |
| stats-median-even | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
