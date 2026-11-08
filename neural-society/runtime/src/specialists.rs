//! Specialists: one atomic digit operation each. Learned (MLP) and exact-code
//! implementations share one trait, so any specialist can be swapped for code.

use crate::model::{argmax_conf, Mlp, Scratch};
use std::{io, path::Path, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Op {
    /// (a, b, carry 0-1) -> a+b+c            (20 classes)
    Add = 0,
    /// (a, b, borrow 0-1) -> a-b-borrow+10   (20 classes)
    Sub = 1,
    /// (a, b, carry 0-8) -> a*b+c            (90 classes)
    Mul = 2,
    /// (a, b) -> 0 '<', 1 '=', 2 '>'         (3 classes)
    Cmp = 3,
}

impl Op {
    pub const ALL: [Op; 4] = [Op::Add, Op::Sub, Op::Mul, Op::Cmp];

    pub fn from_u8(v: u8) -> Op {
        Op::ALL[v as usize]
    }
    pub fn name(self) -> &'static str {
        ["add", "sub", "mul", "cmp"][self as usize]
    }
    /// One-hot field sizes of the inputs.
    pub fn fields(self) -> &'static [usize] {
        match self {
            Op::Add | Op::Sub => &[10, 10, 2],
            Op::Mul => &[10, 10, 9],
            Op::Cmp => &[10, 10],
        }
    }
    pub fn classes(self) -> usize {
        match self {
            Op::Add | Op::Sub => 20,
            Op::Mul => 90,
            Op::Cmp => 3,
        }
    }
    /// Ground truth, as exact code.
    #[inline]
    pub fn exact(self, args: [u8; 4]) -> u16 {
        let [a, b, c, _] = args.map(|x| x as u16);
        match self {
            Op::Add => a + b + c,
            Op::Sub => a + 10 - b - c,
            Op::Mul => a * b + c,
            Op::Cmp => (a > b) as u16 + (a >= b) as u16,
        }
    }
    /// Every valid input, for exhaustive verification.
    pub fn all_inputs(self) -> Vec<[u8; 4]> {
        let f = self.fields();
        let mut v = Vec::new();
        for a in 0..f[0] as u8 {
            for b in 0..f[1] as u8 {
                for c in 0..*f.get(2).unwrap_or(&1) as u8 {
                    v.push([a, b, c, 0]);
                }
            }
        }
        v
    }
}

pub trait Specialist: Send {
    fn op(&self) -> Op;
    fn kind(&self) -> &str;
    /// `forward(inputs) -> (output class, confidence)`.
    fn forward(&mut self, args: [u8; 4]) -> (u16, f32);
}

pub struct Exact(pub Op);

impl Specialist for Exact {
    fn op(&self) -> Op {
        self.0
    }
    fn kind(&self) -> &str {
        "exact"
    }
    #[inline]
    fn forward(&mut self, args: [u8; 4]) -> (u16, f32) {
        (self.0.exact(args), 1.0)
    }
}

pub struct Learned {
    op: Op,
    mlp: Arc<Mlp>,
    scratch: Scratch,
    offsets: [usize; 3],
    label: String,
}

impl Learned {
    pub fn new(op: Op, mlp: Arc<Mlp>, label: &str) -> io::Result<Self> {
        let f = op.fields();
        let in_dim: usize = f.iter().sum();
        if mlp.in_dim() != in_dim || mlp.out_dim() != op.classes() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{label}: model is {}->{} but {} needs {}->{}",
                    mlp.in_dim(), mlp.out_dim(), op.name(), in_dim, op.classes()
                ),
            ));
        }
        let mut offsets = [0; 3];
        for j in 1..f.len() {
            offsets[j] = offsets[j - 1] + f[j - 1];
        }
        let scratch = Scratch::for_model(&mlp);
        Ok(Learned { op, mlp, scratch, offsets, label: label.to_string() })
    }

    pub fn load(op: Op, path: impl AsRef<Path>, label: &str) -> io::Result<Self> {
        Self::new(op, Arc::new(Mlp::load(path)?), label)
    }

    pub fn n_params(&self) -> usize {
        self.mlp.n_params()
    }
}

impl Specialist for Learned {
    fn op(&self) -> Op {
        self.op
    }
    fn kind(&self) -> &str {
        &self.label
    }
    #[inline]
    fn forward(&mut self, args: [u8; 4]) -> (u16, f32) {
        let f = self.op.fields();
        let mut active = [0usize; 3];
        for j in 0..f.len() {
            debug_assert!((args[j] as usize) < f[j], "{} arg {j} out of range", self.op.name());
            active[j] = self.offsets[j] + args[j] as usize;
        }
        let logits = self.mlp.forward_onehot(&active[..f.len()], &mut self.scratch);
        let (k, conf) = argmax_conf(logits);
        (k as u16, conf)
    }
}

/// Build one specialist per op from a config string.
///
/// `base[,op=kind...]` where base is `exact` or `learned`, and kind is `exact`,
/// `learned`, or `weak` (mul only: the deliberately imperfect multiplier).
/// Examples: `learned`, `exact`, `learned,mul=weak`, `exact,add=learned`.
pub fn build_society(cfg: &str, weights: &Path) -> io::Result<Vec<Box<dyn Specialist>>> {
    let mut parts = cfg.split(',');
    let base = parts.next().unwrap_or("learned");
    let mut kinds = [base; 4];
    for p in parts {
        let (op, kind) = p.split_once('=').ok_or_else(|| inval(format!("bad override '{p}'")))?;
        let op = Op::ALL
            .into_iter()
            .find(|o| o.name() == op)
            .ok_or_else(|| inval(format!("unknown op '{op}'")))?;
        kinds[op as usize] = kind;
    }
    Op::ALL
        .into_iter()
        .map(|op| -> io::Result<Box<dyn Specialist>> {
            match kinds[op as usize] {
                "exact" => Ok(Box::new(Exact(op))),
                "learned" => Ok(Box::new(Learned::load(op, weights.join(format!("{}.bin", op.name())), op.name())?)),
                "weak" if op == Op::Mul => Ok(Box::new(Learned::load(op, weights.join("mul_weak.bin"), "mul_weak")?)),
                k => Err(inval(format!("unknown kind '{k}' for {}", op.name()))),
            }
        })
        .collect()
}

fn inval(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ops() {
        assert_eq!(Op::Add.exact([9, 9, 1, 0]), 19);
        assert_eq!(Op::Sub.exact([0, 9, 1, 0]), 0);
        assert_eq!(Op::Sub.exact([9, 0, 0, 0]), 19);
        assert_eq!(Op::Mul.exact([9, 9, 8, 0]), 89);
        assert_eq!(Op::Cmp.exact([3, 5, 0, 0]), 0);
        assert_eq!(Op::Cmp.exact([5, 5, 0, 0]), 1);
        assert_eq!(Op::Cmp.exact([7, 5, 0, 0]), 2);
        assert_eq!(Op::Mul.all_inputs().len(), 900);
        assert_eq!(Op::Cmp.all_inputs().len(), 100);
    }
}
