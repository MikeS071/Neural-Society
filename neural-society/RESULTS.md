# Neural Society: arithmetic results

All numbers come from one full run of `cargo run --release -- bench` (raw output in
[`results/bench.md`](results/bench.md)). The machine was a 4-vCPU cloud container,
so treat the absolute latencies as indicative only; the ratios between modes are
the useful part.

Test set: 2,000 random expressions. Operands have exactly 1, 2, 3, 5, 8, 10, 15, 20,
25 or 30 digits, with 1 to 5 operators drawn from `+ - *` (40 per cell). Ground truth
comes from Python big ints. A second set of 3,000 single-operator problems with
operands from 0 to 999 matches the baseline's own input distribution.

## Summary

| hypothesis | verdict |
|---|---|
| (a) specialists are exhaustively verifiable | **Yes, trivially.** All four are 100% correct on every possible input, and the Rust forward pass confirms it. The calibration of the imperfect one can also be measured exactly. |
| (b) length generalization that an equal-size monolith lacks | **Yes, but the credit belongs mostly to the hand-written router.** See the caveats below. |
| (c) very low latency in a systems language | **Only without the bus.** Direct calls cost 1.3 µs per learned digit operation. A thread-per-specialist bus adds 0.7–30 µs per message, which is 20–3,000x the compute it carries. |

## 1. Verifiability

| specialist | params | cases | accuracy | min conf (correct) | max conf (wrong) |
|---|---:|---:|---:|---:|---:|
| add | 6,932 | 200 | 100% | 0.9995 | — |
| sub | 6,932 | 200 | 100% | 0.9996 | — |
| mul | 11,930 | 900 | 100% | 0.9982 | — |
| cmp | 5,699 | 100 | 100% | 0.9999 | — |
| mul_weak | 1,530 | 900 | 96.1% (35 wrong) | 0.326 | 0.637 |

`mul_weak` is now trained until it first lands in the 96–99% band (hidden layer [12],
525 steps), which makes it deterministic. Its 35 wrong cases are listed in
`weights/specialists_report.json`.

Because the input space is tiny, confidence can be calibrated exactly:

| threshold | wrong answers flagged | correct answers flagged |
|---:|---:|---:|
| 0.5 | 24/35 | 46/865 |
| 0.6 | 32/35 | 123/865 |
| **0.7** | **35/35** | **245/865** |
| 0.9 | 35/35 | 602/865 |

All 35 errors have confidence below 0.637, so a threshold of 0.7 catches every one of
them. The cost is re-checking 28% of the correct answers.

## 2. Accuracy and length generalization

| digits | exact code | learned | learned + mul_weak | learned + mul_weak, escalate conf<0.7 |
|---:|---:|---:|---:|---:|
| 1 | 100% | 100% | 98.5% | 100% |
| 3 | 100% | 100% | 85.0% | 100% |
| 5 | 100% | 100% | 69.0% | 100% |
| 10 | 100% | 100% | 48.5% | 100% |
| 20 | 100% | 100% | 42.0% | 100% |
| 30 | 100% | 100% | 34.0% | 100% |
| **all 2,000** | **100%** | **100%** | **60.9%** | **100%** |

The learned society gets every expression right, from 1 to 30 digits and 1 to 5
operators. Answers run to roughly 150 digits. Nothing in the design caps the length.

**Error compounding.** Inside real runs, `mul_weak` gets 2.66% of its multiply calls
wrong. That is lower than its 3.9% exhaustive error rate, because carries are not
uniformly distributed. One bad digit anywhere ruins the answer. Accuracy therefore
decays with the number of multiply calls: 98% at around 1 call per expression, 34%
at around 1,200. The full digits-by-operators grid is in `results/bench.md`.
Sixty-eight percent of 30-digit single-operator expressions still come out right,
because about two-thirds of them are additions or subtractions, which never call the
multiplier.

**Escalation.** If any reply under confidence 0.7 is re-asked to the exact-code
specialist, accuracy returns to 100% on every test. 13% of all calls (28% of multiply
calls) are escalated. In the bus design this is the blackboard's low-confidence log
acting as a supervisor.

### Monolithic baseline (same 3,000 problems)

The baseline is one MLP for `a op b` with 3-digit operands: 6 digit heads plus a sign
head. It was trained on 300K random problems (10% of the 3M problem space) for 8,000
steps at batch size 512.

| model | params | held-out accuracy | + | − | × | latency | share of sweep it can accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 1x | 31,562 | 48.9% | 84.5% | 64.2% | 0.3% | 3.5 µs | 6.0% |
| baseline 10x | 314,690 | 65.2% | 99.6% | 97.8% | 1.5% | 37 µs | 6.0% |
| baseline 100x | 3,148,430 | 66.8% | 99.9% | 99.5% | 4.3% | 498 µs | 6.0% |
| society, learned | 31,493 | **100%** | 100% | 100% | 100% | 9.0 µs | 100% |
| society, exact code | 0 | 100% | 100% | 100% | 100% | 0.2 µs | 100% |

- Addition and subtraction become learnable once the baseline is 10x the society's
  size. Multiplication does not, even at 100x. At 100x, multiplication accuracy on
  training data is 12.9% but on held-out data only 3.2%, so the model is memorizing
  rather than learning an algorithm. The baseline's own training-time evaluation
  (`weights/baseline_report.json`, a different random split) agrees within about 1
  point.
- The baseline cannot take a 4-digit operand or a second operator, because its input
  layer has no place for them. Only 6% of the sweep can be fed to it at all.
- Caveat: all three sizes got the same data and step budget, so the 100x model is
  probably undertrained relative to its capacity. A longer run might push addition
  and subtraction to 100%. Nothing seen here suggests multiplication would follow.

## 3. Latency, messages, throughput

Times are per expression, one expression at a time. A message is one
request/reply pair with a specialist.

| digits | msgs / expr | exact direct | learned direct | learned bus (blocking) | learned bus (spin) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.3 µs | 5.3 µs | 156 µs | 21 µs |
| 3 | 30 | 0.5 µs | 36 µs | 1.04 ms | 64 µs |
| 10 | 314 | 2.8 µs | 396 µs | 10.7 ms | 676 µs |
| 30 | 2,423 | 17.9 µs | 3.09 ms | 73.2 ms | 4.85 ms |

| mode | ns / message | expr / s (whole sweep, 1 router) |
|---|---:|---:|
| exact, direct call | 8 | 192,000 |
| learned, direct call | 1,309 | 1,200 |
| exact, threaded bus, spin-then-block | 669 | 2,350 |
| learned, threaded bus, spin-then-block | 1,993 | 790 |
| exact, threaded bus, blocking `mpsc` | 28,400 | 55 |
| learned, threaded bus, blocking `mpsc` | 30,600 | 51 |

What the numbers say:

- **Compute.** A learned digit operation is a 64x64 MLP, about 10K multiply-adds,
  and costs 1.3 µs. That is roughly 160x the cost of the same operation in exact
  code.
- **Bus overhead.** With blocking receives, the bus adds about 29 µs per message:
  thread park and wake, two futex syscalls per hop. Spinning before blocking cuts
  this to about 0.7 µs. It is still pure overhead, because every add and subtract
  waits on the previous carry, and each multiply row waits on the one before it. One
  router never has more than one message in flight, so the threads buy no parallelism.
- **Throughput.** Throughput scales across independent expressions, not inside one
  expression. With 4 router threads, each owning a private society (direct mode),
  throughput is 3.5–3.75x that of one router. With 4 routers sharing one bus, the
  speedup is 3.3x when blocking, but only 1.05–1.34x when spinning: 8 spinning threads
  on 4 cores starve each other.
- **Message counts.** These grow as length × length for multiplication: a
  30-digit expression averages 2,400 messages. The blackboard arena is preallocated
  (64K digits) and `reset()` keeps its capacity, so evaluation does not allocate in
  steady state.

## Honest read

**Where the approach wins**

1. **Correctness you can prove.** Every specialist's whole input space is checked,
   so "100% correct" is a fact, not an estimate. Confidence calibration on the weak
   specialist is also exact, and it shows that the softmax confidence of a small MLP
   is good enough to flag 100% of its own errors at a 28% false-alarm rate. The
   society can supervise itself: a low-confidence log plus escalation turned 61%
   accuracy back into 100%.
2. **Unbounded length at fixed size.** 31K parameters handle 30-digit, 5-operator
   expressions, which no size of the monolithic baseline can even accept. Within the
   baseline's own 3-digit range, the society is 100% correct where a 100x-larger
   monolith gets 67% (and 4% on multiplication).
3. **Swappability.** One trait covers learned and exact specialists, so the
   ablations here are just config strings (`learned,mul=weak`, `exact,add=learned`).

**Where it doesn't, or where the comparison flatters it**

1. **The router is doing the hard part.** Long multiplication, carry propagation,
   sign handling and operator precedence are all hand-written exact code. The
   specialists are learned lookup tables with at most 900 entries. The length
   generalization belongs to the algorithm, not to learning. The baseline has to
   discover that algorithm from examples, so this is not a like-for-like test of
   learning. The real open question is whether a *learned* planner can produce
   these plans, and this prototype does not test it.
2. **Learning adds nothing at this granularity except cost.** An exhaustively
   verified specialist over 900 inputs is a lookup table; compiled to one, it would
   be about 160x faster with identical behaviour. Learned specialists only pay off
   when the skill cannot be written as code or enumerated, which will be true for the
   next milestone but is not true here.
3. **Digit-level messages are far too fine-grained for a bus.** Even a spinning
   channel costs 80x the compute of an exact-code operation, and the carry
   dependencies leave nothing to parallelise. A bus pays off when each message
   carries a lot of work, such as a compile or a test run, or when many independent
   requests are in flight. The design should batch (send a whole row or column) or
   keep cheap specialists in-process and put only expensive ones behind a bus.
4. **Errors compound multiplicatively.** A 97%-accurate specialist drops end-to-end
   accuracy to around 35% after 1,000 calls. The escalation that fixed it here
   needed a confidence threshold calibrated on the full input space, which real-world
   specialists will not have.

## Next milestone (not started)

The next milestone is the "fix the failing test" loop: a large model as planner,
with fast specialists (compiler, test runner, linter, failure classifier) writing
structured results to the blackboard, compared against the large model alone on
success rate, iterations and tokens. The arithmetic results point to a few design
choices:

- Put the large model in the router's seat, since planning is where the hard part
  was here. Make specialists coarse-grained, so each message carries milliseconds or
  more of work and bus overhead stops mattering.
- Keep the `forward -> (output, confidence)` contract, the low-confidence log and
  escalation to a stronger or exact tool. That combination was the most useful
  mechanism in this prototype.
- Run independent checks (lint, type-check, tests) concurrently over the bus. That
  is where it can actually parallelize, unlike the carry chains here.
