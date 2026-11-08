# Neural Society benchmark

2000 sweep expressions (40 per (digits, ops) cell), 3000 baseline-distribution expressions. Host: 4 hardware threads. Release build.

### Exhaustive specialist verification (Rust forward pass)

| specialist | params | cases | correct | accuracy | min conf (correct) | max conf (wrong) |
|---|---:|---:|---:|---:|---:|---:|
| add | 6932 | 200 | 200 | 100.00% | 0.9995 | — |
| sub | 6932 | 200 | 200 | 100.00% | 0.9996 | — |
| mul | 11930 | 900 | 900 | 100.00% | 0.9982 | — |
| cmp | 5699 | 100 | 100 | 100.00% | 0.9999 | — |
| mul_weak | 1530 | 900 | 865 | 96.11% | 0.3258 | 0.6369 |

### `mul_weak` calibration: does low confidence flag the wrong answers? (all 900 inputs)

| threshold | wrong answers flagged | correct answers flagged (wasted escalations) |
|---:|---:|---:|
| 0.5 | 24/35 | 46/865 |
| 0.6 | 32/35 | 123/865 |
| 0.7 | 35/35 | 245/865 |
| 0.8 | 35/35 | 382/865 |
| 0.9 | 35/35 | 602/865 |
| 0.95 | 35/35 | 691/865 |
| 0.99 | 35/35 | 761/865 |
| 0.999 | 35/35 | 811/865 |

## Accuracy by operand length (direct mode; all operator counts pooled)

| digits | exact | learned | learned + mul_weak | learned + mul_weak, escalate conf<0.7 | mul calls / expr |
|---:|---:|---:|---:|---:|---:|
| 1 | 100.0% | 100.0% | 98.5% | 100.0% | 1 |
| 2 | 100.0% | 100.0% | 91.5% | 100.0% | 5 |
| 3 | 100.0% | 100.0% | 85.0% | 100.0% | 11 |
| 5 | 100.0% | 100.0% | 69.0% | 100.0% | 30 |
| 8 | 100.0% | 100.0% | 61.5% | 100.0% | 68 |
| 10 | 100.0% | 100.0% | 48.5% | 100.0% | 144 |
| 15 | 100.0% | 100.0% | 44.0% | 100.0% | 288 |
| 20 | 100.0% | 100.0% | 42.0% | 100.0% | 475 |
| 25 | 100.0% | 100.0% | 35.0% | 100.0% | 836 |
| 30 | 100.0% | 100.0% | 34.0% | 100.0% | 1172 |
| **all** | **100.0%** | **100.0%** | **60.9%** | **100.0%** | |

## Accuracy by operator count (all lengths pooled)

| ops | exact | learned | learned + mul_weak | learned + mul_weak, escalate conf<0.7 |
|---:|---:|---:|---:|---:|
| 1 | 100.0% | 100.0% | 83.5% | 100.0% |
| 2 | 100.0% | 100.0% | 66.0% | 100.0% |
| 3 | 100.0% | 100.0% | 60.2% | 100.0% |
| 4 | 100.0% | 100.0% | 51.2% | 100.0% |
| 5 | 100.0% | 100.0% | 43.5% | 100.0% |

## Digit-level calls: in-situ error rates

| society | calls | wrong calls | wrong add/sub/mul/cmp | escalated to exact |
|---|---:|---:|---:|---:|
| exact | 1274707 | 0 (0.000%) | 0/0/0/0 (mul: 0.000% of 606016) | 0 (0.00%) |
| learned | 1274707 | 0 (0.000%) | 0/0/0/0 (mul: 0.000% of 606016) | 0 (0.00%) |
| learned + mul_weak | 1274775 | 16118 (1.264%) | 0/0/16118/0 (mul: 2.660% of 606047) | 0 (0.00%) |
| learned + mul_weak, escalate conf<0.7 | 1274707 | 15999 (1.255%) | 0/0/15999/0 (mul: 2.640% of 606016) | 167154 (13.11%) |

("wrong calls" are counted before escalation, i.e. what the specialist itself answered.)

## Error compounding: `learned + mul_weak` accuracy grid (rows: digits, cols: operators)

| digits | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 100% | 100% | 100% | 100% | 92% |
| 2 | 100% | 92% | 85% | 90% | 90% |
| 3 | 98% | 85% | 85% | 85% | 72% |
| 5 | 90% | 75% | 72% | 65% | 42% |
| 8 | 80% | 65% | 58% | 58% | 48% |
| 10 | 82% | 60% | 40% | 32% | 28% |
| 15 | 80% | 55% | 52% | 18% | 15% |
| 20 | 78% | 50% | 38% | 20% | 25% |
| 25 | 60% | 42% | 35% | 28% | 10% |
| 30 | 68% | 35% | 38% | 18% | 12% |

## Latency per expression (single client, one expression at a time)

| digits | msgs / expr | exact direct | learned direct | exact bus | learned bus | exact bus-spin | learned bus-spin | overhead / msg (bus, bus-spin) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.3 µs | 5.3 µs | 139.2 µs | 156.0 µs | 13.7 µs | 21.0 µs | 29.7 µs, 2.9 µs |
| 2 | 15 | 0.4 µs | 17.1 µs | 430.8 µs | 513.1 µs | 16.9 µs | 32.7 µs | 29.0 µs, 1.1 µs |
| 3 | 30 | 0.5 µs | 35.9 µs | 881.8 µs | 1.04 ms | 28.2 µs | 64.2 µs | 29.6 µs, 0.9 µs |
| 5 | 73 | 1.0 µs | 89.1 µs | 2.11 ms | 2.40 ms | 55.8 µs | 139.7 µs | 28.9 µs, 0.7 µs |
| 8 | 158 | 1.6 µs | 196.0 µs | 4.72 ms | 5.43 ms | 109.4 µs | 323.3 µs | 29.9 µs, 0.7 µs |
| 10 | 314 | 2.8 µs | 396.4 µs | 9.65 ms | 10.70 ms | 213.3 µs | 675.8 µs | 30.7 µs, 0.7 µs |
| 15 | 618 | 5.3 µs | 1.03 ms | 18.11 ms | 19.56 ms | 417.0 µs | 1.29 ms | 29.3 µs, 0.7 µs |
| 20 | 1003 | 8.2 µs | 1.27 ms | 28.68 ms | 29.87 ms | 688.4 µs | 1.94 ms | 28.6 µs, 0.7 µs |
| 25 | 1736 | 13.4 µs | 2.21 ms | 49.35 ms | 52.02 ms | 1.12 ms | 3.36 ms | 28.4 µs, 0.6 µs |
| 30 | 2423 | 17.9 µs | 3.09 ms | 67.02 ms | 73.15 ms | 1.60 ms | 4.85 ms | 27.7 µs, 0.7 µs |

| mode | mean | p50 | p99 | total msgs | ns / msg | expr / s |
|---|---:|---:|---:|---:|---:|---:|
| exact bus | 18.11 ms | 1.75 ms | 193.98 ms | 1274707 | 28416 | 55 |
| exact bus-spin | 426.1 µs | 54.0 µs | 4.59 ms | 1274707 | 669 | 2346 |
| exact direct | 5.1 µs | 0.9 µs | 49.4 µs | 1274707 | 8 | 192376 |
| learned bus | 19.48 ms | 1.85 ms | 197.94 ms | 1274707 | 30572 | 51 |
| learned bus-spin | 1.27 ms | 124.2 µs | 14.06 ms | 1274707 | 1993 | 787 |
| learned direct | 834.0 µs | 65.3 µs | 9.50 ms | 1274707 | 1309 | 1198 |

## Throughput with 4 concurrent routers (whole sweep, split across routers)

| society | mode | passes over sweep | wall time | expr / s | msgs / s | vs 1 router |
|---|---|---:|---:|---:|---:|---:|
| exact | direct (one society per router) | 50 | 0.14 s | 720819 | 459.42M | 3.75x |
| exact | bus, blocking | 1 | 11.14 s | 180 | 0.11M | 3.25x |
| exact | bus, spin | 1 | 0.81 s | 2474 | 1.58M | 1.05x |
| learned | direct (one society per router) | 2 | 0.96 s | 4178 | 2.66M | 3.49x |
| learned | bus, blocking | 1 | 11.79 s | 170 | 0.11M | 3.30x |
| learned | bus, spin | 1 | 1.90 s | 1052 | 0.67M | 1.34x |

## Monolithic baseline vs society on the baseline's own distribution (a op b, 0..999)

| model | params | accuracy | + | - | * | mean latency | can run sweep inputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 1x | 31562 | 48.9% | 84.5% | 64.2% | 0.3% | 3.5 µs | 6.0% |
| baseline 10x | 314690 | 65.2% | 99.6% | 97.8% | 1.5% | 37.1 µs | 6.0% |
| baseline 100x | 3148430 | 66.8% | 99.9% | 99.5% | 4.3% | 497.6 µs | 6.0% |
| society (learned) | 31493 | 100.0% | 100.0% | 100.0% | 100.0% | 9.0 µs | 100% |
| society (exact code) | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 0.2 µs | 100% |
| society (learned + mul_weak) | 31493 | 95.4% | 100.0% | 100.0% | 86.6% | 6.6 µs | 100% |

The baseline's input layer is three digits wide and has one operator slot, so only 6.0% of the sweep (1-3 digit operands with a single operator) can even be fed to it.
