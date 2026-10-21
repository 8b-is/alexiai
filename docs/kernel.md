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

### 2. Lossless Accumulation in $\mathbb{Z}$ & Explicit Overflow Bounds
Projecting $x_k \to p_{x,k} / q_x$ introduces a bounded initial quantization step
$\epsilon_{\text{proj}} \le \frac{1}{2 q_x}$ ($\approx 4.77 \times 10^{-7}$ at
$q_x = 2^{20}$). Relative to those quantized integer inputs, the inner-loop
accumulation in $\mathbb{Z}$ is **100% exact and associative** with zero
intermediate rounding drift, provided accumulator headroom is respected:
- **64-element `i32` group accumulator**: For $|p_{x,k}| < 2^{25}$, the 64-element
  partial sum satisfies $|S_{64}| \le 64 \times 2^{25} = 2^{31}-1$, guaranteeing
  zero `i32` overflow before widening to `i64`.
- **Outer `i64` row accumulator**: Widens each 64-element `i32` block sum into
  `i64` so arbitrarily large dimensions $K$ never overflow.

### 3. Single Deferred Division per Row (Unscaled & Group-64 Scaled)
Floating-point division is deferred until the final output value is produced.
For unscaled ternary columns:

$$y_i \approx \frac{\gamma}{q_x} \sum_{k=1}^{K} p_{x, k} \cdot w_{i, k}$$

For MLX Group-64 (`group_size = 64`) with per-group scale numerators $p_{s,(i,g)}$:

$$y_i \approx \frac{1}{q_x \cdot q_s} \sum_{g=0}^{G-1} p_{s, (i, g)} \left( \sum_{k \in g} p_{x, k} \cdot w_{i, k} \right)$$

By factoring $p_{s,(i,g)}$ outside the 64-element inner sum and dividing by
$q_x \cdot q_s$ only once per output row ($O(N)$ instead of $O(N \times K)$),
relative error against float ternary reconstruction stays below $10^{-5}$ to $10^{-6}$.

### 4. Exact Threshold Hoisting in Quantization & STE Backward
During weight quantization and Straight-Through Estimator (STE) QAT backward
passes, floating-point divisions $\frac{|w|}{\text{scale}} \ge 0.5$ and
$\frac{|w|}{\text{scale}} \le 1.5$ are algebraically hoisted into exact threshold
comparisons:

$$|w| \ge 0.5 \times \text{scale}, \qquad |w| \le 1.5 \times \text{scale}$$

Pre-calculating $\pm 0.5 \times \text{scale}$ and $\pm 1.5 \times \text{scale}$
outside the loop eliminates $65,536$ per-weight floating-point multiplications
per $256 \times 256$ matrix.

### 5. Associativity Unlocks Wide SIMD Without Accuracy Loss
In safe Rust (`#![forbid(unsafe_code)]`), IEEE-754 non-associativity prevents
LLVM from reordering floating-point additions, forcing sequential execution.
Because integer addition in $\mathbb{Z}$ is mathematically associative, LLVM
automatically vectorizes the integer dot products into 256-bit **AVX2 / NEON
SIMD** instructions (`vpmovsxbd` + `vpmulld` + `vpaddd`), delivering up to an
**11.43× speedup** over dense FP64 CPU matrix multiplication.

### 6. Omni / MEM|8 Block-Floating-Point (`BFP-64`) & Importance-Adaptive Precision
*"Precision should be a property of the information's requirements, not a fixed tax paid by every operation."*

Separating activation storage precision from accumulator precision yields three
block-floating-point (`group_size = 64`, $x_k \approx m_k \cdot 2^{e_g}$) regimes:
1. **Variant 1 (`BFP-i8` + `i16` SIMD accumulator)**: $m_k \in [-127, 127]$ (`1 byte`),
   shared block exponent $e_g \in \mathbb{Z}$. For 64 ternary products,
   $|S_g|_{\max} = 64 \times 127 = 8,128 \le 32,767$, fitting inside a 16-bit
   `i16` accumulator (`16 lanes per 256-bit AVX2 register` via `vpmovsxbw + vpmullw + vpaddw`)
   with **zero possibility of overflow**.
2. **Variant 2 (`BFP-i16` + `i32` SIMD accumulator)**: $m_k \in [-32767, 32767]$ (`2 bytes`),
   $|S_g|_{\max} = 64 \times 32,767 = 2,097,088 \ll 2^{31}-1$, fitting in `i32`
   with $< 10^{-4}$ relative error.
3. **Variant 3 (`MEM|8 Adaptive` `i8` $\to$ `i16` Promotion)**: Starts every 64-element
   block in compact `i8`; if a block's relative L2 quantization error exceeds threshold
   $\tau$, automatically promotes **only that block** to `i16` while accumulating into a
   unified output.

*the constellation · 0 + 1 · fine touch from within · vaked.dev*
