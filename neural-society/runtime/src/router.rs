//! Router / planner. Exact code parses the expression (shunting-yard) into an RPN plan,
//! then orchestrates big-number long addition, subtraction and multiplication digit by
//! digit, asking specialists for every digit-level operation. All values live on the
//! blackboard. Sign handling asks the `cmp` specialist which magnitude is larger.
//!
//! Division of labour: the router only does *structural* work (indexing digits, splitting
//! a specialist's class output into digit + carry with /10 and %10, comparing lengths,
//! XOR of signs). Every arithmetic fact comes from a specialist.

use crate::blackboard::{Blackboard, Kind, LowConf, SlotId};
use crate::bus::Backend;
use crate::specialists::Op;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
enum Tok {
    Num(SlotId),
    Op(u8),
    LParen,
}

fn prec(op: u8) -> u8 {
    if op == b'*' {
        2
    } else {
        1
    }
}

pub struct Router<B: Backend> {
    pub backend: B,
    pub bb: Blackboard,
    /// Replies with confidence below this are logged on the blackboard.
    pub threshold: f32,
    /// If set, low-confidence replies are replaced by the exact-code specialist.
    pub escalate: bool,
    pub calls: u64,
    pub escalations: u64,
    rpn: Vec<Tok>,
    ops: Vec<Tok>,
    stack: Vec<SlotId>,
}

impl<B: Backend> Router<B> {
    pub fn new(backend: B) -> Self {
        Router {
            backend,
            bb: Blackboard::with_capacity(1 << 16, 1 << 10),
            threshold: 0.0,
            escalate: false,
            calls: 0,
            escalations: 0,
            rpn: Vec::with_capacity(64),
            ops: Vec::with_capacity(64),
            stack: Vec::with_capacity(64),
        }
    }

    pub fn with_threshold(mut self, t: f32, escalate: bool) -> Self {
        self.threshold = t;
        self.escalate = escalate;
        self
    }

    /// Evaluate `expr` (non-negative integers, `+ - *`, parentheses), appending the decimal
    /// result to `out`. The blackboard is reset first and holds the full trace afterwards.
    pub fn eval(&mut self, expr: &str, out: &mut String) -> Result<(), String> {
        self.bb.reset();
        self.plan(expr.as_bytes())?;
        self.stack.clear();
        for i in 0..self.rpn.len() {
            match self.rpn[i] {
                Tok::Num(s) => self.stack.push(s),
                Tok::Op(op) => {
                    let b = self.stack.pop().ok_or("missing operand")?;
                    let a = self.stack.pop().ok_or("missing operand")?;
                    let r = self.apply(op, a, b);
                    self.stack.push(r);
                }
                Tok::LParen => unreachable!(),
            }
        }
        if self.stack.len() != 1 {
            return Err("malformed expression".into());
        }
        self.bb.write_decimal(self.stack[0], out);
        Ok(())
    }

    /// Shunting-yard: tokenize into operand slots on the blackboard and an RPN plan.
    fn plan(&mut self, s: &[u8]) -> Result<(), String> {
        self.rpn.clear();
        self.ops.clear();
        let mut expect_operand = true;
        let mut i = 0;
        while i < s.len() {
            let c = s[i];
            match c {
                b' ' | b'\t' => i += 1,
                b'0'..=b'9' if expect_operand => {
                    let start = i;
                    while i < s.len() && s[i].is_ascii_digit() {
                        i += 1;
                    }
                    let slot = self.bb.alloc(Kind::Operand);
                    for &d in s[start..i].iter().rev() {
                        self.bb.push(slot, d - b'0');
                    }
                    self.bb.finish(slot, false);
                    self.rpn.push(Tok::Num(slot));
                    expect_operand = false;
                }
                b'(' if expect_operand => {
                    self.ops.push(Tok::LParen);
                    i += 1;
                }
                b')' if !expect_operand => {
                    loop {
                        match self.ops.pop() {
                            Some(Tok::LParen) => break,
                            Some(t) => self.rpn.push(t),
                            None => return Err("unbalanced ')'".into()),
                        }
                    }
                    i += 1;
                }
                b'+' | b'-' | b'*' if !expect_operand => {
                    while let Some(&Tok::Op(top)) = self.ops.last() {
                        if prec(top) >= prec(c) {
                            self.rpn.push(Tok::Op(top));
                            self.ops.pop();
                        } else {
                            break;
                        }
                    }
                    self.ops.push(Tok::Op(c));
                    expect_operand = true;
                    i += 1;
                }
                _ => return Err(format!("unexpected '{}' at {i}", c as char)),
            }
        }
        if expect_operand {
            return Err("expression ends without an operand".into());
        }
        while let Some(t) = self.ops.pop() {
            match t {
                Tok::LParen => return Err("unbalanced '('".into()),
                t => self.rpn.push(t),
            }
        }
        Ok(())
    }

    /// One digit-level question to a specialist, with confidence logging / escalation.
    #[inline]
    fn ask(&mut self, op: Op, args: [u8; 4]) -> u16 {
        self.calls += 1;
        let (mut r, conf) = self.backend.call(op, args);
        if conf < self.threshold {
            if self.escalate {
                r = op.exact(args);
                self.escalations += 1;
            }
            self.bb.low_conf.push(LowConf {
                call: self.calls,
                op,
                args,
                result: r,
                conf,
                escalated: self.escalate,
            });
        }
        r
    }

    fn apply(&mut self, op: u8, a: SlotId, b: SlotId) -> SlotId {
        let (an, bn) = (self.bb.neg(a), self.bb.neg(b));
        match op {
            b'+' => self.signed_add(a, an, b, bn),
            b'-' => self.signed_add(a, an, b, !bn),
            b'*' => {
                let r = self.mul_mag(a, b);
                self.bb.set_kind(r, Kind::StepResult);
                self.bb.finish(r, an ^ bn);
                r
            }
            _ => unreachable!(),
        }
    }

    /// (+/-|a|) + (+/-|b|).
    fn signed_add(&mut self, a: SlotId, an: bool, b: SlotId, bn: bool) -> SlotId {
        if an == bn {
            let r = self.add_mag(a, b, 0, Kind::StepResult);
            self.bb.finish(r, an);
            r
        } else {
            match self.cmp_mag(a, b) {
                Ordering::Equal => {
                    let r = self.bb.alloc(Kind::StepResult);
                    self.bb.push(r, 0);
                    self.bb.finish(r, false);
                    r
                }
                Ordering::Greater => {
                    let r = self.sub_mag(a, b);
                    self.bb.finish(r, an);
                    r
                }
                Ordering::Less => {
                    let r = self.sub_mag(b, a);
                    self.bb.finish(r, bn);
                    r
                }
            }
        }
    }

    /// Compare magnitudes: lengths structurally, then digits MS-first via `cmp`.
    fn cmp_mag(&mut self, a: SlotId, b: SlotId) -> Ordering {
        let (la, lb) = (self.bb.len(a), self.bb.len(b));
        if la != lb {
            return la.cmp(&lb);
        }
        for i in (0..la).rev() {
            match self.ask(Op::Cmp, [self.bb.digit(a, i), self.bb.digit(b, i), 0, 0]) {
                0 => return Ordering::Less,
                2 => return Ordering::Greater,
                _ => {}
            }
        }
        Ordering::Equal
    }

    /// |x| + |y| * 10^shift into a new (unfinished) slot. Digits of x below the shift
    /// are copied: the carry there is provably 0, so no question needs asking.
    fn add_mag(&mut self, x: SlotId, y: SlotId, shift: usize, kind: Kind) -> SlotId {
        let r = self.bb.alloc(kind);
        let (lx, ly) = (self.bb.len(x), self.bb.len(y) + shift);
        for i in 0..shift {
            let d = self.bb.digit(x, i);
            self.bb.push(r, d);
        }
        let mut carry = 0u8;
        for i in shift..lx.max(ly) {
            let s = self.ask(
                Op::Add,
                [self.bb.digit(x, i), self.bb.digit(y, i - shift), carry, 0],
            );
            self.bb.push(r, (s % 10) as u8);
            carry = (s / 10) as u8;
        }
        if carry > 0 {
            self.bb.push(r, carry);
        }
        r
    }

    /// |x| - |y| for |x| >= |y| into a new (unfinished) slot.
    fn sub_mag(&mut self, x: SlotId, y: SlotId) -> SlotId {
        let r = self.bb.alloc(Kind::StepResult);
        let mut borrow = 0u8;
        for i in 0..self.bb.len(x) {
            let v = self.ask(
                Op::Sub,
                [self.bb.digit(x, i), self.bb.digit(y, i), borrow, 0],
            );
            if v >= 10 {
                self.bb.push(r, (v - 10) as u8);
                borrow = 0;
            } else {
                self.bb.push(r, v as u8);
                borrow = 1;
            }
        }
        r
    }

    /// Long multiplication: one row per digit of the shorter operand (`mul` with carry),
    /// accumulated with shifted long addition (`add`). Returns an unfinished slot.
    fn mul_mag(&mut self, a: SlotId, b: SlotId) -> SlotId {
        let (a, b) = if self.bb.len(a) >= self.bb.len(b) {
            (a, b)
        } else {
            (b, a)
        };
        let mut sum: Option<SlotId> = None;
        let rows = self.bb.len(b);
        for j in 0..rows {
            let d = self.bb.digit(b, j);
            let row = self.bb.alloc(Kind::Intermediate);
            let mut carry = 0u8;
            for i in 0..self.bb.len(a) {
                let p = self.ask(Op::Mul, [self.bb.digit(a, i), d, carry, 0]);
                self.bb.push(row, (p % 10) as u8);
                carry = (p / 10) as u8;
            }
            if carry > 0 {
                self.bb.push(row, carry);
            }
            self.bb.finish(row, false);
            sum = Some(match sum {
                None => row,
                Some(s) => {
                    let kind = if j + 1 == rows {
                        Kind::StepResult
                    } else {
                        Kind::Intermediate
                    };
                    let n = self.add_mag(s, row, j, kind);
                    self.bb.finish(n, false);
                    n
                }
            });
        }
        // `finish` again by the caller sets the sign; it is idempotent on digits.
        sum.expect("operand has at least one digit")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Direct;
    use crate::specialists::{Exact, Specialist};

    fn router() -> Router<Direct> {
        Router::new(Direct::new(
            Op::ALL
                .into_iter()
                .map(|o| Box::new(Exact(o)) as Box<dyn Specialist>)
                .collect(),
        ))
    }

    fn eval(r: &mut Router<Direct>, e: &str) -> String {
        let mut s = String::new();
        r.eval(e, &mut s).unwrap();
        s
    }

    #[test]
    fn arithmetic() {
        let mut r = router();
        for (e, want) in [
            ("0", "0"),
            ("7+5", "12"),
            ("5-7", "-2"),
            ("7-7", "0"),
            ("0*123", "0"),
            ("999*999", "998001"),
            ("2+3*4", "14"),
            ("(2+3)*4", "20"),
            ("10-20-30", "-40"),
            ("1-2*3", "-5"),
            ("1-2*3*0", "1"),
            ("3-5*2+100", "93"),
            ("2-3-4*5*6+7", "-114"),
            (
                "123456789012345678901234567890*987654321098765432109876543210",
                "121932631137021795226185032733622923332237463801111263526900",
            ),
            ("100000000000000000000-1", "99999999999999999999"),
            ("1-100000000000000000000", "-99999999999999999999"),
            ("00012", "12"),
        ] {
            assert_eq!(eval(&mut r, e), want, "{e}");
        }
    }

    #[test]
    fn rejects_bad_input() {
        let mut r = router();
        let mut s = String::new();
        for e in ["", "1+", "+1", "(1+2", "1+2)", "1/2", "1 2"] {
            assert!(r.eval(e, &mut s).is_err(), "{e}");
        }
    }

    #[test]
    fn message_counts() {
        let mut r = router();
        eval(&mut r, "123*45"); // 2 rows x 3 mul; then 615 + 4920 needs 3 adds (digit 0 is copied)
        assert_eq!(r.calls, 6 + 3);
        assert_eq!(r.calls, r.backend.messages());
    }
}
