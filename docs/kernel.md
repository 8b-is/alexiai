# The kernel — GAIA-MLX-QUANT

The local inference substrate: BitNet b1.58 ternary weights, packed, and the
masked accumulation that multiplies without materializing floats.

## packing

5 trits per byte, base-3 little-endian, `-1 → 2`, `0 → 0`, `+1 → 1`:

```
bytes = ceil(weights / 5)          bits per weight = log2(3) ≈ 1.5850
compression vs float32             = 32 / 1.5850  ≈ 20.18×
```

`packTernary` and `unpackTernary` are exact inverses — pinned by round-trip
tests at every length modulo 5. A trit outside `{-1,0,+1}` is a thrown
`RangeError`, not a silent corruption.

## quantization

Absmean, per row (the scheme real b1.58 models are trained into via the
straight-through estimator):

```
scale_r        = mean(|W[r, :]|)
W_q[r, c]      = clip(round(W[r, c] / scale_r), -1, +1)
```

A `{rows, cols}` shape selects per-row scaling; without it, a single global
scale is used. The reconstruction is `W ≈ scale_r · W_q`.

## the kernel

$$Y_i = \gamma \cdot \sum_j \left( \mathbf{W}^{+}_{j,i} - \mathbf{W}^{-}_{j,i} \right) \cdot \text{scale}_j \cdot x_j$$

The positive and negative planes are byte masks derived from the packed
buffer — the hot path never allocates a float matrix. Accumulation is in
float64 so long columns never drift.

## honesty

The bench reports *measured* error, including the worst case. On adversarial
random weights a single projection is lossy by design (≈ 0.4–0.5 relative
error, uniform or Gaussian — this is true of one-shot absmean ternary, and it
is why b1.58 models are *trained* into ternary-shaped weights rather than
quantized once and shipped). On ternary-shaped weights the projection is
tight (< 0.34, pinned by test). This document exists so the number is never
misread as a marketing claim.

*the constellation · 0 + 1 · fine touch from within · vaked.dev*
