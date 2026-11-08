# Neural Society: arithmetic prototype

Can many tiny single-skill "specialist" models, coordinated through a shared blackboard
and a message bus, compose into a system that does harder tasks? First testbed: exact
arithmetic on `+ - *` expressions of arbitrary length, checked against Python big ints.

The results and an honest read on them are in [`RESULTS.md`](RESULTS.md). The original
brief is [`HANDOVER.md`](HANDOVER.md).

```
train/            Python + NumPy, training only
  nn.py           MLP, manual backprop, Adam, NSOC binary export
  specialists.py  add / sub / mul / cmp, trained on and verified against every input;
                  mul_weak, tuned into the 96-99% band for the error-compounding test
  baseline.py     monolithic `a op b` MLP at 1x / 10x / 100x the specialists' params
  gen_tests.py    benchmark expressions + ground truth -> data/*.tsv
weights/          exported .bin models + training reports (JSON)
data/             benchmark expressions
runtime/          Rust, std only, no Python at runtime
  src/model.rs        NSOC loader, MLP forward pass, softmax confidence
  src/specialists.rs  Specialist trait; Learned (MLP) and Exact (code) impls
  src/bus.rs          fixed-size Message, threaded mpsc bus, direct-call backend
  src/blackboard.rs   preallocated append-only arena of typed digit slots
  src/router.rs       shunting-yard planner + digit-by-digit long arithmetic
  src/baseline.rs     the monolithic baseline, run from Rust
  src/main.rs         `ns` CLI: eval / verify / bench
results/bench.md  raw benchmark output
```

## Reproduce

```bash
cd train
python3 specialists.py      # ~8 s; writes weights/{add,sub,mul,cmp,mul_weak}.bin
python3 baseline.py         # ~15 min on 4 cores (--quick for a 1-min smoke run)
python3 gen_tests.py        # data/tests.tsv, data/tests_3digit.tsv

cd ../runtime
cargo test --release
cargo run --release -- verify                       # exhaustive check through Rust
cargo run --release -- eval "123456789*987654321-5"
cargo run --release -- eval --society learned,mul=weak --escalate "999*999*999"
cargo run --release -- eval --bus --spin "2-3-4*5*6+7"
cargo run --release -- bench                        # writes ../results/bench.md
```

Society configs swap any specialist for code or for another model:
`learned`, `exact`, `learned,mul=weak`, `exact,add=learned`, and so on.
`--threshold T` logs replies with confidence below T on the blackboard.
`--escalate` also replaces them with the exact-code answer.
