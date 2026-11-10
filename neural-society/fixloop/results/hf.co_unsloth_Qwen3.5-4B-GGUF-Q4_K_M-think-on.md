# Fix-the-failing-test: `hf.co_unsloth_Qwen3.5-4B-GGUF-Q4_K_M-think-on`

31 tasks x 1 repeat(s), max 5 planner calls per run. Success = visible suite passes **and** hidden tests pass.

| arm | success | visible pass | planner calls (all / successful) | input tok | output tok | tokens per success | LLM s | tool s | hidden reasoning chars | bad edits | unparseable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| solo | 87.1% | 93.5% | 1.39 / 1.15 | 2180 | 5719 | 9069 | 117.1 | 0.61 | 16949 | 0 | 13 |
| society | 83.9% | 87.1% | 1.77 / 1.31 | 3068 | 10621 | 16322 | 213.1 | 1.28 | 33979 | 0 | 25 |

## Per task (successes / runs, mean planner calls)

| task | categories | solo | society |
|---|---|---|---|
| bsearch-count | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
| bsearch-hang | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
| bsearch-upper | condition | 1/1, 1.0 | 1/1, 2.0 |
| dates-day-of-year | off_by_one | 1/1, 1.0 | 1/1, 3.0 |
| dates-leap-logic | condition | 1/1, 1.0 | 1/1, 1.0 |
| dates-year-wrap | constant | 1/1, 3.0 | 1/1, 4.0 |
| graph-components-sort | missing_logic | 1/1, 1.0 | 1/1, 1.0 |
| graph-path-order | missing_logic | 1/1, 1.0 | 1/1, 1.0 |
| graph-visited | missing_logic | 1/1, 2.0 | 0/1, 5.0 |
| intervals-contained | missing_logic | 1/1, 1.0 | 1/1, 1.0 |
| intervals-length | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
| intervals-overlap-touch | condition | 1/1, 1.0 | 1/1, 1.0 |
| inventory-add-existing | arithmetic | 1/1, 1.0 | 1/1, 1.0 |
| inventory-latent-delete (latent) | condition | 0/1, 1.0 | 0/1, 1.0 |
| inventory-total-items | wrong_name | 1/1, 1.0 | 1/1, 1.0 |
| ledger-format-div | arithmetic | 1/1, 1.0 | 1/1, 1.0 |
| ledger-helper-pad | wrong_name | 0/1, 5.0 | 0/1, 5.0 |
| ledger-latent-plus (latent) | constant | 0/1, 1.0 | 0/1, 5.0 |
| ledger-two-bugs | arithmetic,condition | 1/1, 1.0 | 1/1, 2.0 |
| matrix-det-sign | off_by_one | 1/1, 1.0 | 1/1, 2.0 |
| matrix-syntax | syntax | 1/1, 1.0 | 1/1, 1.0 |
| matrix-transpose-index | wrong_name | 1/1, 1.0 | 1/1, 1.0 |
| roman-repeat | missing_logic | 0/1, 5.0 | 0/1, 5.0 |
| roman-subtractive | condition | 1/1, 1.0 | 1/1, 1.0 |
| roman-table | constant | 1/1, 1.0 | 1/1, 1.0 |
| stats-median-even | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
| stats-mode-tie | wrong_name | 1/1, 1.0 | 1/1, 1.0 |
| stats-two-bugs | off_by_one | 1/1, 2.0 | 1/1, 1.0 |
| textutil-palindrome-name | wrong_name | 1/1, 1.0 | 1/1, 1.0 |
| textutil-title-first | condition | 1/1, 1.0 | 1/1, 1.0 |
| textutil-truncate | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
