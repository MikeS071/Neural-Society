# Fix-the-failing-test: `hf.co_unsloth_Qwen3.5-4B-GGUF-Q4_K_M-think-off`

31 tasks x 1 repeat(s), max 5 planner calls per run. Success = visible suite passes **and** hidden tests pass.

| arm | success | visible pass | planner calls (all / successful) | input tok | output tok | tokens per success | LLM s | tool s | hidden reasoning chars | bad edits | unparseable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| solo | 90.3% | 93.5% | 1.48 / 1.25 | 3652 | 1223 | 5397 | 24.6 | 1.26 | 0 | 5 | 7 |
| society | 77.4% | 90.3% | 1.55 / 1.12 | 4622 | 1870 | 8386 | 38.5 | 0.62 | 0 | 4 | 13 |

## Per task (successes / runs, mean planner calls)

| task | categories | solo | society |
|---|---|---|---|
| bsearch-count | off_by_one | 1/1, 2.0 | 1/1, 1.0 |
| bsearch-hang | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
| bsearch-upper | condition | 1/1, 1.0 | 1/1, 2.0 |
| dates-day-of-year | off_by_one | 1/1, 4.0 | 1/1, 1.0 |
| dates-leap-logic | condition | 1/1, 1.0 | 1/1, 1.0 |
| dates-year-wrap | constant | 1/1, 1.0 | 0/1, 5.0 |
| graph-components-sort | missing_logic | 1/1, 1.0 | 1/1, 1.0 |
| graph-path-order | missing_logic | 1/1, 1.0 | 1/1, 1.0 |
| graph-visited | missing_logic | 0/1, 5.0 | 1/1, 2.0 |
| intervals-contained | missing_logic | 1/1, 1.0 | 0/1, 2.0 |
| intervals-length | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
| intervals-overlap-touch | condition | 1/1, 1.0 | 1/1, 1.0 |
| inventory-add-existing | arithmetic | 1/1, 1.0 | 1/1, 1.0 |
| inventory-latent-delete (latent) | condition | 0/1, 1.0 | 0/1, 1.0 |
| inventory-total-items | wrong_name | 1/1, 1.0 | 0/1, 5.0 |
| ledger-format-div | arithmetic | 1/1, 1.0 | 1/1, 1.0 |
| ledger-helper-pad | wrong_name | 1/1, 1.0 | 0/1, 5.0 |
| ledger-latent-plus (latent) | constant | 0/1, 5.0 | 0/1, 1.0 |
| ledger-two-bugs | arithmetic,condition | 1/1, 1.0 | 1/1, 2.0 |
| matrix-det-sign | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
| matrix-syntax | syntax | 1/1, 2.0 | 1/1, 1.0 |
| matrix-transpose-index | wrong_name | 1/1, 1.0 | 1/1, 1.0 |
| roman-repeat | missing_logic | 1/1, 2.0 | 1/1, 1.0 |
| roman-subtractive | condition | 1/1, 1.0 | 1/1, 1.0 |
| roman-table | constant | 1/1, 1.0 | 1/1, 1.0 |
| stats-median-even | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
| stats-mode-tie | wrong_name | 1/1, 1.0 | 1/1, 1.0 |
| stats-two-bugs | off_by_one | 1/1, 2.0 | 0/1, 2.0 |
| textutil-palindrome-name | wrong_name | 1/1, 1.0 | 1/1, 1.0 |
| textutil-title-first | condition | 1/1, 1.0 | 1/1, 1.0 |
| textutil-truncate | off_by_one | 1/1, 1.0 | 1/1, 1.0 |
