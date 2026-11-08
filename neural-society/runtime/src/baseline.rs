//! The monolithic baseline (`train/baseline.py`) run from Rust, for a like-for-like
//! latency and accuracy comparison. It accepts exactly `a op b` with 0 <= a, b <= 999;
//! anything else is structurally out of reach and `predict` returns `None`.

use crate::model::{argmax_conf, Mlp, Scratch};
use std::{io, path::Path};

pub const N_DIG: usize = 3;
pub const N_OUT_DIG: usize = 6;

pub struct Baseline {
    mlp: Mlp,
    scratch: Scratch,
}

impl Baseline {
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let mlp = Mlp::load(path)?;
        if mlp.in_dim() != 63 || mlp.out_dim() != 62 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a baseline model (expects 63 -> 62)",
            ));
        }
        let scratch = Scratch::for_model(&mlp);
        Ok(Baseline { mlp, scratch })
    }

    pub fn n_params(&self) -> usize {
        self.mlp.n_params()
    }

    /// Parse `a op b` into (a, op index, b) if it fits the input layer.
    pub fn accepts(expr: &str) -> Option<(u32, usize, u32)> {
        let pos = expr.find(['+', '-', '*'])?;
        let (a, rest) = expr.split_at(pos);
        let op = "+-*".find(&rest[..1])?;
        let b = &rest[1..];
        let ok =
            |s: &str| !s.is_empty() && s.len() <= N_DIG && s.bytes().all(|c| c.is_ascii_digit());
        if !ok(a) || !ok(b) {
            return None;
        }
        Some((a.parse().ok()?, op, b.parse().ok()?))
    }

    pub fn predict(&mut self, expr: &str, out: &mut String) -> bool {
        let Some((a, op, b)) = Self::accepts(expr) else {
            return false;
        };
        let mut active = [0usize; 2 * N_DIG + 1];
        for i in 0..N_DIG {
            let p = 10u32.pow((N_DIG - 1 - i) as u32);
            active[i] = 10 * i + ((a / p) % 10) as usize;
            active[N_DIG + 1 + i] = 33 + 10 * i + ((b / p) % 10) as usize;
        }
        active[N_DIG] = 30 + op;
        let logits = self.mlp.forward_onehot(&active, &mut self.scratch);
        let mut mag: u64 = 0;
        for h in 0..N_OUT_DIG {
            mag = mag * 10 + argmax_conf(&logits[10 * h..10 * h + 10]).0 as u64;
        }
        let neg = argmax_conf(&logits[60..62]).0 == 1 && mag != 0;
        if neg {
            out.push('-');
        }
        out.push_str(&mag.to_string());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepts_only_its_input_space() {
        assert_eq!(Baseline::accepts("12*999"), Some((12, 2, 999)));
        assert_eq!(Baseline::accepts("0-5"), Some((0, 1, 5)));
        assert_eq!(Baseline::accepts("1000+1"), None);
        assert_eq!(Baseline::accepts("1+2+3"), None);
    }
}
