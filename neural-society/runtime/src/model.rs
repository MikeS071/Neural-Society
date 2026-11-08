//! MLP loader + forward pass for the `NSOC` weight format written by `train/nn.py`.
//!
//! Format (little-endian): b"NSOC", u32 n_layers, then per layer
//! u32 in, u32 out, f32[out*in] W (row-major, out x in), f32[out] b.
//! Hidden layers use ReLU; the last layer emits logits.

use std::{fs, io, path::Path};

pub struct Layer {
    pub inp: usize,
    pub out: usize,
    /// Row-major, `out x inp`.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

pub struct Mlp {
    pub layers: Vec<Layer>,
    /// First-layer weights transposed to `inp x out`, so a one-hot input is a sum of
    /// contiguous rows instead of a strided column gather.
    w0t: Vec<f32>,
}

/// Reusable activation buffers so the forward pass never allocates.
pub struct Scratch {
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Scratch {
    pub fn for_model(m: &Mlp) -> Self {
        let w = m.layers.iter().map(|l| l.out.max(l.inp)).max().unwrap_or(0);
        Scratch { a: vec![0.0; w], b: vec![0.0; w] }
    }
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Mlp {
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        Self::parse(&bytes).map_err(|e| bad(format!("{}: {}", path.display(), e)))
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let s = bytes.get(pos..pos + n).ok_or("truncated file")?;
            pos += n;
            Ok(s)
        };
        if take(4)? != b"NSOC" {
            return Err("bad magic (expected NSOC)".into());
        }
        let u32_ = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize;
        let n_layers = u32_(take(4)?);
        if n_layers == 0 || n_layers > 64 {
            return Err(format!("implausible layer count {n_layers}"));
        }
        let mut layers = Vec::with_capacity(n_layers);
        for k in 0..n_layers {
            let inp = u32_(take(4)?);
            let out = u32_(take(4)?);
            if let Some(prev) = layers.last() {
                let prev: &Layer = prev;
                if prev.out != inp {
                    return Err(format!("layer {k}: in {inp} != previous out {}", prev.out));
                }
            }
            let floats = |s: &[u8]| -> Vec<f32> {
                s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            };
            let w = floats(take(4 * inp * out)?);
            let b = floats(take(4 * out)?);
            layers.push(Layer { inp, out, w, b });
        }
        if pos != bytes.len() {
            return Err(format!("{} trailing bytes", bytes.len() - pos));
        }
        let l0 = &layers[0];
        let mut w0t = vec![0.0; l0.inp * l0.out];
        for o in 0..l0.out {
            for i in 0..l0.inp {
                w0t[i * l0.out + o] = l0.w[o * l0.inp + i];
            }
        }
        Ok(Mlp { layers, w0t })
    }

    pub fn in_dim(&self) -> usize {
        self.layers[0].inp
    }
    pub fn out_dim(&self) -> usize {
        self.layers.last().unwrap().out
    }
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }

    /// Dense forward pass; returns the logits (a view into `s`).
    pub fn forward<'s>(&self, x: &[f32], s: &'s mut Scratch) -> &'s [f32] {
        assert_eq!(x.len(), self.in_dim());
        s.a[..x.len()].copy_from_slice(x);
        self.run_from(0, s)
    }

    /// Forward pass for a concatenated one-hot input given the indices of the hot units.
    pub fn forward_onehot<'s>(&self, active: &[usize], s: &'s mut Scratch) -> &'s [f32] {
        let l0 = &self.layers[0];
        let h = &mut s.a[..l0.out];
        h.copy_from_slice(&l0.b);
        for &i in active {
            debug_assert!(i < l0.inp);
            let row = &self.w0t[i * l0.out..(i + 1) * l0.out];
            for (hv, &w) in h.iter_mut().zip(row) {
                *hv += w;
            }
        }
        if self.layers.len() == 1 {
            return &s.a[..l0.out];
        }
        for v in h.iter_mut() {
            *v = v.max(0.0);
        }
        self.run_from(1, s)
    }

    /// Apply layers `first..` to the activations in `s.a`.
    fn run_from<'s>(&self, first: usize, s: &'s mut Scratch) -> &'s [f32] {
        let last = self.layers.len() - 1;
        for (k, l) in self.layers.iter().enumerate().skip(first) {
            let (x, y) = (&s.a[..l.inp], &mut s.b[..l.out]);
            for o in 0..l.out {
                let v = l.b[o] + dot(&l.w[o * l.inp..(o + 1) * l.inp], x);
                y[o] = if k < last { v.max(0.0) } else { v };
            }
            std::mem::swap(&mut s.a, &mut s.b);
        }
        &s.a[..self.out_dim()]
    }
}

/// 8-lane dot product so the compiler can vectorize without reassociation flags.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Argmax and its softmax probability (the specialist's confidence).
#[inline]
pub fn argmax_conf(logits: &[f32]) -> (usize, f32) {
    let (mut best, mut zmax) = (0usize, f32::NEG_INFINITY);
    for (i, &z) in logits.iter().enumerate() {
        if z > zmax {
            best = i;
            zmax = z;
        }
    }
    let denom: f32 = logits.iter().map(|&z| (z - zmax).exp()).sum();
    (best, 1.0 / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Vec<u8> {
        // 2 layers: 3 -> 2 (ReLU) -> 2
        let mut v = b"NSOC".to_vec();
        v.extend(2u32.to_le_bytes());
        let mut layer = |inp: u32, out: u32, w: &[f32], b: &[f32]| {
            v.extend(inp.to_le_bytes());
            v.extend(out.to_le_bytes());
            w.iter().chain(b).for_each(|f| v.extend(f.to_le_bytes()));
        };
        layer(3, 2, &[1.0, -1.0, 0.5, 0.0, 2.0, -3.0], &[0.0, 1.0]);
        layer(2, 2, &[1.0, 1.0, -1.0, 2.0], &[0.5, 0.0]);
        v
    }

    #[test]
    fn dense_and_onehot_agree() {
        let m = Mlp::parse(&tiny()).unwrap();
        let mut s = Scratch::for_model(&m);
        let d = m.forward(&[0.0, 1.0, 1.0], &mut s).to_vec();
        // h = relu([ -1+0.5, 2-3+1 ]) = [0, 0]; out = [0.5, 0]
        assert_eq!(d, vec![0.5, 0.0]);
        let o = m.forward_onehot(&[1, 2], &mut s).to_vec();
        assert_eq!(d, o);
        let (i, c) = argmax_conf(&o);
        assert_eq!(i, 0);
        assert!((c - 1.0 / (1.0 + (-0.5f32).exp())).abs() < 1e-6);
    }

    #[test]
    fn rejects_garbage() {
        assert!(Mlp::parse(b"NOPE").is_err());
        let mut t = tiny();
        t.pop();
        assert!(Mlp::parse(&t).is_err());
    }
}
