# Neural Society: handover for Claude Code

## Goal
Test whether many tiny, single-skill "specialist" models, coordinated through a
shared blackboard memory and a message bus, can compose into a system that does
harder tasks. First testbed: exact arithmetic on expressions (+, -, *), where
correctness is easy to check against Python big ints.

Hypothesis to test: decomposing into atomic skills gives (a) exhaustively
verifiable specialists, (b) length generalization that a monolithic model of equal
size lacks, and (c) very low latency when the runtime is in a systems language.

## Architecture
- **Python (training only):** trains specialists and the baseline, exports weights to
  a flat binary file.
- **Rust (runtime):** model inference, blackboard, message bus, router/planner,
  benchmarks. No Python at runtime.
- The specialist interface is model-agnostic: `forward(inputs) -> (output, confidence)`.
  Learned models and exact-code functions implement the same trait, so any
  specialist can be swapped for code (the ablation).

## Done so far (files included)
- `train/nn.py`: NumPy MLP with manual backprop, Adam and binary export.
  Weight format (little-endian): `b"NSOC"`, `u32 n_layers`, then per layer
  `u32 in, u32 out, f32[out*in] W (row-major), f32[out] b`. Hidden layers use
  ReLU; the last layer gives logits. Inputs are concatenated one-hot fields.
- `train/specialists.py`: trains 4 specialists on their **entire** input space.
  All reached 100% exhaustive accuracy:

  | name | inputs (one-hot fields) | output classes | params |
  |------|-------------------------|----------------|--------|
  | add | a 0-9, b 0-9, carry 0-1 | a+b+c (20) | 6,932 |
  | sub | a 0-9, b 0-9, borrow 0-1 | a-b-borrow+10 (20) | 6,932 |
  | mul | a 0-9, b 0-9, carry 0-8 | a*b+c (90) | 11,930 |
  | cmp | a 0-9, b 0-9 | <,=,> (3) | 5,699 |

- `mul_weak`: was meant to be a deliberately imperfect multiplier for the
  error-compounding experiment. Hidden [12] with 250 steps gave 73% (too broken).
  Hidden [20] or more with 400+ steps hit 100% (too good). **Still to tune:** aim for
  95–99%, e.g. hidden [12–16] with fewer steps, or train on a random 80% subset of
  cases.

## Remaining work

### 1. Monolithic baseline (`train/baseline.py`)
- One MLP that takes `a op b` directly: fixed-width one-hot digits for operands of
  up to 3 digits each, plus the op. Output is one 10-way softmax head per result
  digit, plus a sign head (`MLP.step` already supports multiple heads).
- Train on a random split; report held-out accuracy.
- Run three parameter budgets: equal to all specialists combined (~31K), 10x and
  100x.
- Note: it structurally cannot take longer inputs. That is part of the result.

### 2. Rust runtime (`runtime/`, cargo, std only)
- `model.rs`: load `.bin` files, run the MLP forward pass, softmax max-prob as
  confidence.
- `bus.rs`: fixed-size `Message { id, from, to, op, args: [u8; 4], result: u16, conf: f32 }`.
  One inbox per specialist (`std::sync::mpsc`, or a ring buffer later).
- `blackboard.rs`: preallocated arena of typed slots (operand digit vectors,
  intermediate results, step status).
- `specialists.rs`: a trait with learned and exact implementations; each
  specialist runs on its own thread, pulling from its inbox.
- `router.rs`: an exact-code parser (shunting-yard) turns the expression into a
  plan. Big-number long addition, subtraction and multiplication are orchestrated
  digit by digit by sending messages to specialists. Sign handling uses the cmp
  specialist. Log any reply with confidence below a threshold.
- Also add a direct-call mode (no threads or channels) to measure pure
  orchestration overhead.

### 3. Benchmarks
- `train/gen_tests.py`: random expressions with ground truth from Python big ints,
  saved to a file Rust reads. Sweep operand length (1–30 digits) and operator count
  (1–5).
- Measure:
  - accuracy by size: learned specialists vs exact-code specialists vs `mul_weak`
  - latency per expression, messages per expression, throughput
  - threaded bus vs direct-call mode
- Baseline comparison: in-distribution 3-digit accuracy, plus a note that it can't
  run on longer inputs at all.
- Write the results to `RESULTS.md`, including an honest read on where the
  approach wins and where it doesn't.

### 4. Next milestone (after arithmetic)
"Fix the failing test" loop: a large model as planner, with fast specialists
(compiler, tests, linter, failure classifier) writing structured results to the
blackboard. Compare against the large model alone on success rate, iterations and
tokens.

## Environment notes
- The original sandbox had no PyTorch and no network, which is why training is
  NumPy-only. If PyTorch is available, swapping the MLPs for tiny transformers
  (llama2.c-style) is a good follow-up. Keep the same export-to-binary approach.
- Run training with `cd train && python3 specialists.py` (takes about 8 s on CPU).
