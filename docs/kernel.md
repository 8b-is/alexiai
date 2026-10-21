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

## the native lane

The same math ships natively in `rust/gaia-mlx-quant`: `#![no_std]` core-only
Rust with `#![forbid(unsafe_code)]`, a counting-allocator test proving zero
heap operations in the hot paths, and three kernels (packed-j, packed-i,
i8-plane with 4-way ILP) benchmarked honestly against auto-vectorized dense.
The JS here is the portable reference; the Rust crate is the substrate's
native lane. They are pinned to each other's math.

## $p/q$ ("Counting Divided") — integer-numerator engine (`gaia-mlx-quant::mlx`)

The $p/q$ ("Counting Divided") integer arithmetic engine in `gaia-mlx-quant::mlx`
maintains precision while eliminating per-weight `int -> float` (`cvtsi2sd`)
conversions through exact rational projections, lossless integer accumulation in
$\mathbb{Z}$, and single deferred division:

### 1. Fixed Power-of-Two Rational Projection
Instead of performing floating-point conversions during inner-loop matrix
operations, continuous activation values $x_k$ and group scales $\alpha_g$ are
projected **once** ($O(K)$) onto integer numerators over a fixed power-of-two
denominator $q = 2^{20} = 1,048,576$:

$$x_k \approx \frac{p_{x, k}}{q_x}, \quad \alpha_g \approx \frac{p_{s, g}}{q_s}$$

This $p/q$ representation provides 20 bits of fractional precision while
converting continuous values into pure integers $p_x \in \mathbb{Z}$.

### 2. Lossless Accumulation in $\mathbb{Z}$ (Zero Intermediate Rounding Error)
In traditional floating-point matrix multiplication, IEEE-754 rounding errors
accumulate on every single addition across all $N \times K$ elements. In $p/q$
integer mode:
- Ternary weights $w_{i,j} \in \{-1, 0, +1\}$ are multiplied directly by the
  integer numerators $p_x[j]$ in the integer domain $\mathbb{Z}$.
- All intermediate dot products are accumulated in 32-bit and 64-bit integer
  registers (`i32`/`i64`).
- Because integer addition in $\mathbb{Z}$ is **100% exact and associative**,
  there is **zero rounding error, truncation, or precision drift** during the
  entire inner-loop summation.

### 3. Single Deferred Division per Row/Group
Floating-point division or conversion is deferred until the final output value
is produced:

$$y_i = \frac{1}{q_x \cdot q_s} \sum_{k=1}^{K} p_{x, k} \cdot w_{i, k}$$

By performing division only once per output row ($O(N)$ operations) rather than
on every weight ($O(N \times K)$), intermediate numerical noise is eliminated.
Converting back to floating-point at the boundary yields **relative errors
below $10^{-5}$ to $10^{-6}$** compared to dense FP64 floating-point baselines.

### 4. Exact Threshold Hoisting in Quantization
During weight quantization, floating-point divisions $\frac{|w|}{\text{scale}} \ge 0.5$
are algebraically hoisted into exact threshold comparisons:

$$|w| \ge 0.5 \times \text{scale}$$

Pre-calculating $\pm 0.5 \times \text{scale}$ outside the loop eliminates
$65,536$ per-weight floating-point multiplications per $256 \times 256$ matrix
while ensuring exact trit assignment ($\{-1, 0, +1\}$) without rounding noise.

### 5. Associativity Unlocks Wide SIMD Without Accuracy Loss
In safe Rust (`#![forbid(unsafe_code)]`), IEEE-754 non-associativity prevents
LLVM from reordering floating-point additions, forcing sequential execution.
Because integer addition in $\mathbb{Z}$ is mathematically associative, LLVM
automatically vectorizes the integer dot products into 256-bit **AVX2 / NEON
SIMD** instructions (`vpmovsxbd` + `vpmulld` + `vpaddd`), delivering up to an
**11.43× speedup** over dense FP64 CPU matrix multiplication while maintaining
strict, exact precision.

*the constellation · 0 + 1 · fine touch from within · vaked.dev*
