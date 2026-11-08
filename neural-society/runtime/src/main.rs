//! `ns` — CLI for the Neural Society runtime.
//!
//!   ns eval [--society CFG] [--bus [--spin]] [--threshold T] [--escalate] [--trace] EXPR
//!   ns verify                       exhaustive check of every specialist + calibration
//!   ns bench [--quick] [--out FILE] full benchmark suite, markdown report
//!
//! Society configs: `exact`, `learned`, `learned,mul=weak`, `exact,add=learned`, ...
//! Paths default to ../weights and ../data next to this crate (override with
//! --weights DIR / --data DIR).

use neural_society::baseline::Baseline;
use neural_society::bus::{Backend, Bus, Direct};
use neural_society::router::Router;
use neural_society::specialists::{build_society, Exact, Learned, Op, Specialist};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::{env, fs, process};

struct Args {
    pos: Vec<String>,
    flags: BTreeMap<String, String>,
}

impl Args {
    fn parse() -> Args {
        let mut pos = Vec::new();
        let mut flags = BTreeMap::new();
        let mut it = env::args().skip(1).peekable();
        while let Some(a) = it.next() {
            if let Some(k) = a.strip_prefix("--") {
                let takes_value = matches!(k, "society" | "threshold" | "weights" | "data" | "out" | "reps");
                let v = if takes_value { it.next().unwrap_or_default() } else { "1".into() };
                flags.insert(k.to_string(), v);
            } else {
                pos.push(a);
            }
        }
        Args { pos, flags }
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(|s| s.as_str())
    }
    fn has(&self, k: &str) -> bool {
        self.flags.contains_key(k)
    }
    fn dir(&self, k: &str, default: &str) -> PathBuf {
        self.get(k).map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(default))
    }
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    process::exit(1)
}

fn main() {
    let args = Args::parse();
    match args.pos.first().map(|s| s.as_str()) {
        Some("eval") => cmd_eval(&args),
        Some("verify") => print!("{}", verify(&args.dir("weights", "weights"))),
        Some("bench") => cmd_bench(&args),
        _ => {
            eprintln!("usage: ns eval [--society CFG] [--bus [--spin]] [--threshold T] [--escalate] [--trace] EXPR\n       ns verify\n       ns bench [--quick] [--out FILE]");
            process::exit(2)
        }
    }
}

// ---------------------------------------------------------------- eval

fn cmd_eval(args: &Args) {
    let expr = args.pos.get(1).unwrap_or_else(|| die("missing expression"));
    let cfg = args.get("society").unwrap_or("learned");
    let thr: f32 = args.get("threshold").map(|t| t.parse().unwrap_or_else(|_| die("bad threshold"))).unwrap_or(0.9);
    let specs = build_society(cfg, &args.dir("weights", "weights")).unwrap_or_else(|e| die(e));
    let report = |out: &str, calls: u64, esc: u64, dt: Duration, low: &[neural_society::blackboard::LowConf], slots: usize, arena: usize| {
        println!("{out}");
        eprintln!("society={cfg} messages={calls} escalations={esc} slots={slots} arena_digits={arena} time={:.1}us", dt.as_secs_f64() * 1e6);
        if args.has("trace") || !low.is_empty() {
            for l in low.iter().take(20) {
                eprintln!("  low-confidence: call #{} {} {:?} -> {} (conf {:.3}){}", l.call, l.op.name(), &l.args[..3], l.result, l.conf, if l.escalated { " [escalated to exact]" } else { "" });
            }
            if low.len() > 20 {
                eprintln!("  ... {} more", low.len() - 20);
            }
        }
    };
    let mut out = String::new();
    if args.has("bus") {
        let mut bus = Bus::spawn(specs, 1, args.has("spin"));
        let mut r = Router::new(bus.client(0)).with_threshold(thr, args.has("escalate"));
        let t = Instant::now();
        r.eval(expr, &mut out).unwrap_or_else(|e| die(e));
        report(&out, r.calls, r.escalations, t.elapsed(), &r.bb.low_conf, r.bb.n_slots(), r.bb.arena_bytes());
        drop(r);
        bus.shutdown();
    } else {
        let mut r = Router::new(Direct::new(specs)).with_threshold(thr, args.has("escalate"));
        let t = Instant::now();
        r.eval(expr, &mut out).unwrap_or_else(|e| die(e));
        report(&out, r.calls, r.escalations, t.elapsed(), &r.bb.low_conf, r.bb.n_slots(), r.bb.arena_bytes());
    }
}

// ---------------------------------------------------------------- verify

const SPEC_FILES: [(&str, Op); 5] = [("add", Op::Add), ("sub", Op::Sub), ("mul", Op::Mul), ("cmp", Op::Cmp), ("mul_weak", Op::Mul)];
const THRESHOLDS: [f32; 8] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999];

/// Exhaustively run every learned specialist through the Rust forward pass.
fn verify(weights: &Path) -> String {
    let mut md = String::new();
    writeln!(md, "### Exhaustive specialist verification (Rust forward pass)\n").unwrap();
    writeln!(md, "| specialist | params | cases | correct | accuracy | min conf (correct) | max conf (wrong) |").unwrap();
    writeln!(md, "|---|---:|---:|---:|---:|---:|---:|").unwrap();
    let mut weak_confs: Vec<(f32, bool)> = Vec::new();
    for (file, op) in SPEC_FILES {
        let mut s = Learned::load(op, weights.join(format!("{file}.bin")), file).unwrap_or_else(|e| die(e));
        let inputs = op.all_inputs();
        let (mut ok, mut min_ok, mut max_bad) = (0, f32::INFINITY, f32::NAN);
        for a in &inputs {
            let (r, c) = s.forward(*a);
            let good = r == op.exact(*a);
            if good {
                ok += 1;
                min_ok = min_ok.min(c);
            } else {
                max_bad = if max_bad.is_nan() { c } else { max_bad.max(c) };
            }
            if file == "mul_weak" {
                weak_confs.push((c, good));
            }
        }
        let fmt = |x: f32| if x.is_nan() { "—".into() } else { format!("{x:.4}") };
        writeln!(md, "| {file} | {} | {} | {ok} | {:.2}% | {} | {} |", s.n_params(), inputs.len(), 100.0 * ok as f64 / inputs.len() as f64, fmt(min_ok), fmt(max_bad)).unwrap();
    }
    let n_bad = weak_confs.iter().filter(|c| !c.1).count();
    writeln!(md, "\n### `mul_weak` calibration: does low confidence flag the wrong answers? (all 900 inputs)\n").unwrap();
    writeln!(md, "| threshold | wrong answers flagged | correct answers flagged (wasted escalations) |").unwrap();
    writeln!(md, "|---:|---:|---:|").unwrap();
    for t in THRESHOLDS {
        let caught = weak_confs.iter().filter(|c| !c.1 && c.0 < t).count();
        let wasted = weak_confs.iter().filter(|c| c.1 && c.0 < t).count();
        writeln!(md, "| {t} | {caught}/{n_bad} | {wasted}/{} |", weak_confs.len() - n_bad).unwrap();
    }
    md
}

// ---------------------------------------------------------------- bench

#[derive(Clone)]
struct Test {
    n: usize,
    k: usize,
    expr: String,
    ans: String,
}

fn load_tests(path: &Path) -> Vec<Test> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| die(format!("{}: {e} (run train/gen_tests.py)", path.display())));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Test { n: f[0].parse().unwrap(), k: f[1].parse().unwrap(), expr: f[2].into(), ans: f[3].into() }
        })
        .collect()
}

/// Wraps a backend and checks every reply against exact code (only used for accuracy runs).
struct Audit<B: Backend> {
    inner: B,
    wrong: u64,
    wrong_by_op: [u64; 4],
    calls_by_op: [u64; 4],
}

impl<B: Backend> Backend for Audit<B> {
    fn call(&mut self, op: Op, args: [u8; 4]) -> (u16, f32) {
        let (r, c) = self.inner.call(op, args);
        self.calls_by_op[op as usize] += 1;
        if r != op.exact(args) {
            self.wrong += 1;
            self.wrong_by_op[op as usize] += 1;
        }
        (r, c)
    }
    fn messages(&self) -> u64 {
        self.inner.messages()
    }
}

#[derive(Default, Clone)]
struct Run {
    correct: Vec<bool>,
    micros: Vec<f64>,
    msgs: Vec<u64>,
    total: Duration,
}

/// Evaluate every test once (after a warm-up pass) and record per-expression stats.
fn run_router<B: Backend>(r: &mut Router<B>, tests: &[Test], warmup: bool) -> Run {
    let mut out = String::with_capacity(1024);
    if warmup {
        for t in tests.iter().step_by(7) {
            out.clear();
            r.eval(&t.expr, &mut out).unwrap();
        }
    }
    let mut run = Run::default();
    let t_all = Instant::now();
    for t in tests {
        out.clear();
        let m0 = r.backend.messages();
        let t0 = Instant::now();
        r.eval(&t.expr, &mut out).unwrap();
        run.micros.push(t0.elapsed().as_secs_f64() * 1e6);
        run.msgs.push(r.backend.messages() - m0);
        run.correct.push(out == t.ans);
    }
    run.total = t_all.elapsed();
    run
}

fn pct(v: &[bool]) -> f64 {
    100.0 * v.iter().filter(|&&b| b).count() as f64 / v.len().max(1) as f64
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * q).round() as usize]
}
fn us(x: f64) -> String {
    if x >= 1000.0 {
        format!("{:.2} ms", x / 1000.0)
    } else {
        format!("{x:.1} µs")
    }
}
fn group<'a>(tests: &'a [Test], key: impl Fn(&Test) -> usize) -> BTreeMap<usize, Vec<usize>> {
    let mut m: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, t) in tests.iter().enumerate() {
        m.entry(key(t)).or_default().push(i);
    }
    m
}
fn pick<T: Clone>(v: &[T], idx: &[usize]) -> Vec<T> {
    idx.iter().map(|&i| v[i].clone()).collect()
}

fn cmd_bench(args: &Args) {
    let weights = args.dir("weights", "weights");
    let data = args.dir("data", "data");
    let quick = args.has("quick");
    let mut tests = load_tests(&data.join("tests.tsv"));
    let tests3 = load_tests(&data.join("tests_3digit.tsv"));
    if quick {
        tests = tests.into_iter().step_by(4).collect();
    }
    let society = |cfg: &str| build_society(cfg, &weights).unwrap_or_else(|e| die(e));
    let mut md = String::new();
    let host_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    writeln!(md, "# Neural Society benchmark\n").unwrap();
    writeln!(md, "{} sweep expressions ({} per (digits, ops) cell), {} baseline-distribution expressions. Host: {host_threads} hardware threads. Release build.\n", tests.len(), tests.len() / group(&tests, |t| t.n * 100 + t.k).len(), tests3.len()).unwrap();
    md += &verify(&weights);
    eprintln!("verify done");

    // ---- accuracy sweep (direct mode, audited)
    let thr: f32 = args.get("threshold").map(|t| t.parse().unwrap()).unwrap_or(0.7);
    let configs: Vec<(String, &str, bool)> = vec![
        ("exact".into(), "exact", false),
        ("learned".into(), "learned", false),
        ("learned + mul_weak".into(), "learned,mul=weak", false),
        (format!("learned + mul_weak, escalate conf<{thr}"), "learned,mul=weak", true),
    ];
    let mut acc_runs = Vec::new();
    for (label, cfg, esc) in &configs {
        let backend = Audit { inner: Direct::new(society(cfg)), wrong: 0, wrong_by_op: [0; 4], calls_by_op: [0; 4] };
        let mut r = Router::new(backend).with_threshold(thr, *esc);
        let run = run_router(&mut r, &tests, false);
        acc_runs.push((label.clone(), run, r.backend.wrong, r.backend.wrong_by_op, r.backend.calls_by_op, r.escalations, r.calls));
        eprintln!("accuracy {label} done");
    }
    writeln!(md, "\n## Accuracy by operand length (direct mode; all operator counts pooled)\n").unwrap();
    write!(md, "| digits |").unwrap();
    for (l, ..) in &acc_runs {
        write!(md, " {l} |").unwrap();
    }
    writeln!(md, " mul calls / expr |").unwrap();
    writeln!(md, "|---:|{}---:|", "---:|".repeat(acc_runs.len())).unwrap();
    for (n, idx) in group(&tests, |t| t.n) {
        write!(md, "| {n} |").unwrap();
        for (_, run, ..) in &acc_runs {
            write!(md, " {:.1}% |", pct(&pick(&run.correct, &idx))).unwrap();
        }
        let mul_calls: f64 = idx.iter().map(|&i| mul_calls_for(&tests[i].expr) as f64).sum::<f64>() / idx.len() as f64;
        writeln!(md, " {mul_calls:.0} |").unwrap();
    }
    write!(md, "| **all** |").unwrap();
    for (_, run, ..) in &acc_runs {
        write!(md, " **{:.1}%** |", pct(&run.correct)).unwrap();
    }
    writeln!(md, " |").unwrap();

    writeln!(md, "\n## Accuracy by operator count (all lengths pooled)\n").unwrap();
    write!(md, "| ops |").unwrap();
    for (l, ..) in &acc_runs {
        write!(md, " {l} |").unwrap();
    }
    writeln!(md, "\n|---:|{}", "---:|".repeat(acc_runs.len())).unwrap();
    for (k, idx) in group(&tests, |t| t.k) {
        write!(md, "| {k} |").unwrap();
        for (_, run, ..) in &acc_runs {
            write!(md, " {:.1}% |", pct(&pick(&run.correct, &idx))).unwrap();
        }
        writeln!(md).unwrap();
    }

    writeln!(md, "\n## Digit-level calls: in-situ error rates\n").unwrap();
    writeln!(md, "| society | calls | wrong calls | wrong add/sub/mul/cmp | escalated to exact |").unwrap();
    writeln!(md, "|---|---:|---:|---:|---:|").unwrap();
    for (label, _, wrong, wbo, cbo, esc, calls) in &acc_runs {
        writeln!(md, "| {label} | {calls} | {wrong} ({:.3}%) | {}/{}/{}/{} (mul: {:.3}% of {}) | {esc} ({:.2}%) |", 100.0 * *wrong as f64 / *calls as f64, wbo[0], wbo[1], wbo[2], wbo[3], 100.0 * wbo[2] as f64 / cbo[2].max(1) as f64, cbo[2], 100.0 * *esc as f64 / *calls as f64).unwrap();
    }
    writeln!(md, "\n(\"wrong calls\" are counted before escalation, i.e. what the specialist itself answered.)").unwrap();

    // full grid for the weak multiplier (error compounding)
    let weak = &acc_runs[2].1;
    writeln!(md, "\n## Error compounding: `learned + mul_weak` accuracy grid (rows: digits, cols: operators)\n").unwrap();
    writeln!(md, "| digits | 1 | 2 | 3 | 4 | 5 |\n|---:|---:|---:|---:|---:|---:|").unwrap();
    let cells = group(&tests, |t| t.n * 100 + t.k);
    for (n, _) in group(&tests, |t| t.n) {
        write!(md, "| {n} |").unwrap();
        for k in 1..=5 {
            match cells.get(&(n * 100 + k)) {
                Some(idx) => write!(md, " {:.0}% |", pct(&pick(&weak.correct, idx))).unwrap(),
                None => write!(md, " — |").unwrap(),
            }
        }
        writeln!(md).unwrap();
    }

    // ---- latency: direct vs bus, exact vs learned
    writeln!(md, "\n## Latency per expression (single client, one expression at a time)\n").unwrap();
    writeln!(md, "| digits | msgs / expr | exact direct | learned direct | exact bus | learned bus | exact bus-spin | learned bus-spin | overhead / msg (bus, bus-spin) |").unwrap();
    writeln!(md, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    let lat_tests = &tests;
    let mut lat: BTreeMap<&str, Run> = BTreeMap::new();
    for (name, cfg, bus, spin) in [
        ("exact direct", "exact", false, false),
        ("learned direct", "learned", false, false),
        ("exact bus", "exact", true, false),
        ("learned bus", "learned", true, false),
        ("exact bus-spin", "exact", true, true),
        ("learned bus-spin", "learned", true, true),
    ] {
        let run = if bus {
            let mut b = Bus::spawn(society(cfg), 1, spin);
            let mut r = Router::new(b.client(0));
            let run = run_router(&mut r, lat_tests, true);
            drop(r);
            b.shutdown();
            run
        } else {
            let mut r = Router::new(Direct::new(society(cfg)));
            run_router(&mut r, lat_tests, true)
        };
        eprintln!("latency {name} done ({:.1}s)", run.total.as_secs_f64());
        lat.insert(name, run);
    }
    for (n, idx) in group(lat_tests, |t| t.n) {
        let m = |name: &str| mean(&pick(&lat[name].micros, &idx));
        let msgs = mean(&pick(&lat["exact direct"].msgs, &idx).iter().map(|&x| x as f64).collect::<Vec<_>>());
        writeln!(md, "| {n} | {msgs:.0} | {} | {} | {} | {} | {} | {} | {:.1} µs, {:.1} µs |", us(m("exact direct")), us(m("learned direct")), us(m("exact bus")), us(m("learned bus")), us(m("exact bus-spin")), us(m("learned bus-spin")), (m("exact bus") - m("exact direct")) / msgs, (m("exact bus-spin") - m("exact direct")) / msgs).unwrap();
    }
    writeln!(md, "\n| mode | mean | p50 | p99 | total msgs | ns / msg | expr / s |\n|---|---:|---:|---:|---:|---:|---:|").unwrap();
    for (name, run) in &lat {
        let msgs: u64 = run.msgs.iter().sum();
        writeln!(md, "| {name} | {} | {} | {} | {msgs} | {:.0} | {:.0} |", us(mean(&run.micros)), us(quantile(&run.micros, 0.5)), us(quantile(&run.micros, 0.99)), run.total.as_secs_f64() * 1e9 / msgs as f64, run.micros.len() as f64 / run.total.as_secs_f64()).unwrap();
    }

    // ---- throughput with concurrent routers
    let n_clients = host_threads.clamp(2, 8);
    writeln!(md, "\n## Throughput with {n_clients} concurrent routers (whole sweep, split across routers)\n").unwrap();
    writeln!(md, "| society | mode | passes over sweep | wall time | expr / s | msgs / s | vs 1 router |\n|---|---|---:|---:|---:|---:|---:|").unwrap();
    let total_msgs: u64 = lat["exact direct"].msgs.iter().sum();
    for cfg in ["exact", "learned"] {
        for (mode, bus, spin) in [("direct (one society per router)", false, false), ("bus, blocking", true, false), ("bus, spin", true, true)] {
            let reps = match (cfg, bus) { ("exact", false) => 50, (_, false) => 2, _ => 1 };
            let dt = throughput(lat_tests, n_clients, bus, spin, reps, &|| society(cfg));
            let single = lat[format!("{cfg} {}", if !bus { "direct" } else if spin { "bus-spin" } else { "bus" }).as_str()].total.as_secs_f64();
            let eps = (reps * lat_tests.len()) as f64 / dt.as_secs_f64();
            writeln!(md, "| {cfg} | {mode} | {reps} | {:.2} s | {eps:.0} | {:.2}M | {:.2}x |", dt.as_secs_f64(), (reps as u64 * total_msgs) as f64 / dt.as_secs_f64() / 1e6, eps / (lat_tests.len() as f64 / single)).unwrap();
            eprintln!("throughput {cfg} {mode} done");
        }
    }

    // ---- baseline comparison on its own distribution
    writeln!(md, "\n## Monolithic baseline vs society on the baseline's own distribution (a op b, 0..999)\n").unwrap();
    writeln!(md, "| model | params | accuracy | + | - | * | mean latency | can run sweep inputs |\n|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    let by_op = |run: &Run| -> [f64; 3] {
        let mut o = [0.0; 3];
        for (j, c) in ['+', '-', '*'].iter().enumerate() {
            let idx: Vec<usize> = tests3.iter().enumerate().filter(|(_, t)| t.expr.contains(*c)).map(|(i, _)| i).collect();
            o[j] = pct(&pick(&run.correct, &idx));
        }
        o
    };
    let coverage = 100.0 * tests.iter().filter(|t| Baseline::accepts(&t.expr).is_some()).count() as f64 / tests.len() as f64;
    for mult in [1, 10, 100] {
        let path = weights.join(format!("baseline_{mult}x.bin"));
        let Ok(mut b) = Baseline::load(&path) else {
            writeln!(md, "| baseline {mult}x | — | (missing {}) | | | | | |", path.display()).unwrap();
            continue;
        };
        let mut run = Run::default();
        let mut out = String::new();
        for t in &tests3 {
            out.clear();
            let t0 = Instant::now();
            assert!(b.predict(&t.expr, &mut out));
            run.micros.push(t0.elapsed().as_secs_f64() * 1e6);
            run.correct.push(out == t.ans);
        }
        let o = by_op(&run);
        writeln!(md, "| baseline {mult}x | {} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {} | {coverage:.1}% |", b.n_params(), pct(&run.correct), o[0], o[1], o[2], us(mean(&run.micros))).unwrap();
    }
    let learned_params: usize = ["add", "sub", "mul", "cmp"].iter().zip(Op::ALL).map(|(f, op)| Learned::load(op, weights.join(format!("{f}.bin")), f).unwrap().n_params()).sum();
    for (label, cfg) in [("society (learned)", "learned"), ("society (exact code)", "exact"), ("society (learned + mul_weak)", "learned,mul=weak")] {
        let mut r = Router::new(Direct::new(society(cfg)));
        let run = run_router(&mut r, &tests3, true);
        let o = by_op(&run);
        let params = if cfg == "exact" { "0".to_string() } else { learned_params.to_string() };
        writeln!(md, "| {label} | {params} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {} | 100% |", pct(&run.correct), o[0], o[1], o[2], us(mean(&run.micros))).unwrap();
    }
    writeln!(md, "\nThe baseline's input layer is three digits wide and has one operator slot, so only {coverage:.1}% of the sweep (1-3 digit operands with a single operator) can even be fed to it.").unwrap();

    let out_path = args.get("out").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../results/bench.md"));
    if let Some(p) = out_path.parent() {
        let _ = fs::create_dir_all(p);
    }
    fs::write(&out_path, &md).unwrap_or_else(|e| die(e));
    print!("{md}");
    eprintln!("wrote {}", out_path.display());
}

/// Mul specialist calls needed by an expression (product of operand lengths per `*`).
fn mul_calls_for(expr: &str) -> u64 {
    let mut r = Router::new(Audit { inner: Direct::new(Op::ALL.into_iter().map(|o| Box::new(Exact(o)) as Box<dyn Specialist>).collect()), wrong: 0, wrong_by_op: [0; 4], calls_by_op: [0; 4] });
    let mut s = String::new();
    r.eval(expr, &mut s).unwrap();
    r.backend.calls_by_op[Op::Mul as usize]
}

/// Split the tests across `n` router threads; return wall time.
fn throughput(tests: &[Test], n: usize, bus: bool, spin: bool, reps: usize, make: &(dyn Fn() -> Vec<Box<dyn Specialist>> + Sync)) -> Duration {
    let chunks: Vec<Vec<Test>> = (0..n).map(|i| tests.iter().skip(i).step_by(n).cloned().collect()).collect();
    let t0 = Instant::now();
    if bus {
        let mut b = Bus::spawn(make(), n, spin);
        let clients: Vec<_> = (0..n).map(|i| b.client(i)).collect();
        std::thread::scope(|s| {
            for (c, chunk) in clients.into_iter().zip(&chunks) {
                s.spawn(move || {
                    let mut r = Router::new(c);
                    let mut out = String::new();
                    for _ in 0..reps {
                        for t in chunk {
                            out.clear();
                            r.eval(&t.expr, &mut out).unwrap();
                        }
                    }
                });
            }
        });
        let dt = t0.elapsed();
        b.shutdown();
        dt
    } else {
        let societies: Vec<_> = (0..n).map(|_| make()).collect();
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for (soc, chunk) in societies.into_iter().zip(&chunks) {
                s.spawn(move || {
                    let mut r = Router::new(Direct::new(soc));
                    let mut out = String::new();
                    for _ in 0..reps {
                        for t in chunk {
                            out.clear();
                            r.eval(&t.expr, &mut out).unwrap();
                        }
                    }
                });
            }
        });
        t0.elapsed()
    }
}
