# Handover: run milestone 2 locally and write up the results

For a Claude Code session running on the owner's own machine (Pop!_OS, CPU-only) in a
clone of `MikeS071/Neural-Society`, on branch `claude/neural-society-arithmetic-xlo644`.

## Where things stand

- **Milestone 1 (arithmetic)** is finished and merged to `main`. See
  `neural-society/RESULTS.md`.
- **Milestone 2 ("fix the failing test")**: the harness is built and tested offline in
  `neural-society/fixloop/` (read its `README.md` first). It compares two arms on 31
  injected-bug tasks, each with the same planner model, edit format, budget of 5 planner
  calls and success test (visible tests and hidden tests both pass):
  - **solo**: the model sees source, tests and raw `unittest` output.
  - **society**: the model sees source plus a blackboard. Compiler, ruff linter and a
    structured, hang-aware test runner run in parallel, then a localizer and a learned
    failure classifier.
- The cloud session could not run a model: there was no API key, and it had no access to
  this machine. So the planner is a **local CPU model via Ollama**. The pick is
  `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` (Apache-2.0, 2.7 GB, hybrid thinking). The
  README lists alternatives:
  - `hf.co/ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M`: a strong 9B reasoning model, slow on
    CPU.
  - `hf.co/LiquidAI/LFM3-1.7B-GGUF:Q4_K_M`: the "how small can it go" option.

## What the first smoke run showed

The run is in `results/smoke1_qwen3.5-4b_openai-endpoint_thinking-leaked.*`:
- 2 tasks, both arms 4/4 successful; society needed fewer calls, tokens and seconds.
  Too small a sample to mean anything.
- **Bug found:** through Ollama's OpenAI-compatible endpoint, `--think off` was ignored.
  Each call had 1–4K hidden reasoning tokens, and one solo reply came back empty after
  the reasoning used up the token limit. The first call also included model load time.
- **Fixed** in commit `24d77a5`. `--planner ollama` uses the native `/api/chat` with
  `think` and `num_ctx` set per request, warms the model up before timing, and records
  `thinking_chars` and `load_s` per run. `run_local.sh` now uses it.

## Your tasks

1. **Prerequisites.** Python 3.10+ and Ollama (`curl -fsSL https://ollama.com/install.sh | sh`).
   Optional: `pip install ruff`, so the linter specialist is available.
2. `git pull`, then `cd neural-society/fixloop`.
3. `python3 -m unittest discover -s tests`: the offline harness tests should all pass.
4. `bash run_local.sh smoke`. Check in `results/*.md` that **hidden reasoning chars is
   ~0** with thinking off, and note the seconds per call.
5. `bash run_local.sh full`: 31 tasks x 2 arms with thinking off. Expect hours on CPU.
   Keep the machine otherwise idle, because timings are part of the result.
6. If time allows, run in this order:
   - `bash run_local.sh all`, which adds a thinking-on pass. Compare **thinking-on solo
     vs thinking-off society**: can a specialist round replace reasoning time?
   - `--repeats 3`, because local outputs are not deterministic.
   - The 1.7B and 9B models.
7. **Commit `results/` after each pass** (`git add results && git commit && git push`),
   so nothing is lost if a later run fails.
8. **Write `neural-society/fixloop/RESULTS.md`** in the style of
   `neural-society/RESULTS.md`, with an honest read of the numbers:
   - Success (visible and hidden), planner calls, input and output tokens, tokens per
     success, LLM seconds and tool seconds, per arm.
   - **Paired per-task comparison**: tasks only one arm solved, and why. Read the
     transcripts in `results/*.json`.
   - Which specialists visibly helped. Examples: the localizer on helper-function bugs
     (`ledger-helper-pad`), the hang stack on `bsearch-hang` and `graph-visited`, the
     compiler on `matrix-syntax`. Did the weak classifier hint (42% accurate on these
     tasks) mislead the planner?
   - The latent tasks (`inventory-latent-delete`, `ledger-latent-plus`), where only the
     hidden tests reveal the second bug.
   - Caveats: 31 small tasks; one model family; the classifier was trained on mutants
     of the same 10 programs; CPU timings depend on this machine (record CPU model and
     RAM).
9. Update the roadmap item "Milestone 2 results" in the top-level `README.md` and link
   the new results.

## Ground rules

- Don't change the tasks, the prompts or the success criterion mid-experiment. If you
  find a harness bug, fix it, say so in RESULTS.md, and rerun **both** arms.
- Commit to `claude/neural-society-arithmetic-xlo644`. Don't push to `main`, and don't
  open or merge a PR unless the owner asks.
- Don't paste API keys or tokens into files or commits. No key is needed for the
  local path.
