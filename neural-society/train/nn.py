"""Minimal NumPy MLP with manual backprop + Adam. No PyTorch required.

Weight export format (little-endian), read by the Rust runtime:
  magic   b"NSOC"
  u32     n_layers
  per layer: u32 in_dim, u32 out_dim, f32[out_dim*in_dim] W (row-major, out x in), f32[out_dim] b
Hidden layers use ReLU; the last layer emits raw logits.
"""
import struct
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / i), (o, i)).astype(np.float32)
                  for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o, np.float32) for o in sizes[1:]]
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def n_params(self):
        return int(sum(p.size for p in self.params))

    def forward(self, x):
        acts = [x]
        h = x
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.T + b
            if k < len(self.W) - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        return acts

    def logits(self, x):
        return self.forward(x)[-1]

    def step(self, x, heads, ys, lr=3e-3, wd=0.0):
        """heads: list of (start, end) slices of the output; ys: list of int targets per head.
        Cross-entropy summed over heads. Returns mean loss."""
        acts = self.forward(x)
        out = acts[-1]
        grad = np.zeros_like(out)
        loss = 0.0
        n = x.shape[0]
        for (s, e), y in zip(heads, ys):
            z = out[:, s:e]
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            loss += -np.log(p[np.arange(n), y] + 1e-12).mean()
            p[np.arange(n), y] -= 1
            grad[:, s:e] = p / n
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        g = grad
        for k in reversed(range(len(self.W))):
            gW[k] = g.T @ acts[k] + wd * self.W[k]
            gb[k] = g.sum(0)
            if k > 0:
                g = (g @ self.W[k]) * (acts[k] > 0)
        grads = gW + gb
        self.t += 1
        b1, b2 = 0.9, 0.999
        for p, gr, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1; m += (1 - b1) * gr
            v *= b2; v += (1 - b2) * gr * gr
            mh = m / (1 - b1 ** self.t)
            vh = v / (1 - b2 ** self.t)
            p -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return loss

    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"NSOC")
            f.write(struct.pack("<I", len(self.W)))
            for W, b in zip(self.W, self.b):
                f.write(struct.pack("<II", W.shape[1], W.shape[0]))
                f.write(W.astype("<f4").tobytes())
                f.write(b.astype("<f4").tobytes())


def one_hot(vals, sizes):
    """vals: (n, k) ints; sizes: k field sizes -> concatenated one-hot (n, sum(sizes))."""
    n = vals.shape[0]
    out = np.zeros((n, sum(sizes)), np.float32)
    off = 0
    for j, s in enumerate(sizes):
        out[np.arange(n), off + vals[:, j]] = 1
        off += s
    return out
