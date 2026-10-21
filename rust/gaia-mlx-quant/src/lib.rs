//! # gaia-mlx-quant
//!
//! The native lane of the GAIA-MLX-QUANT substrate: ternary BitNet b1.58
//! packing and the masked accumulation kernel, in pure `core` Rust.
//!
//! ## the promises, all enforced at compile time or by test
//!
//! - **zero `unsafe`** — `#![forbid(unsafe_code)]` makes this a compile error,
//!   not a guideline.
//! - **zero allocations** — the kernel writes into caller-owned buffers; a
//!   counting global allocator in the test suite proves the hot path performs
//!   exactly zero heap operations.
//! - **`#![no_std]`** — the crate is `core`-only. It runs anywhere Rust's core
//!   runs: Linux, macOS, bare metal, wasm.
//! - **100% Linux + macOS** — CI runs the suite on both `ubuntu-latest` and
//!   `macos-latest`.
//!
//! ## the bithacks
//!
//! - 5 trits per byte, base-3 — log₂3 ≈ 1.585 bits/weight, the optimal
//!   ternary packing.
//! - branchless digit mapping: `digit = t as u8 + (t as u8 >> 7) * 3` maps
//!   `-1 → 2`, `0 → 0`, `+1 → 1` with no branch.
//! - const-evaluated 256-entry slot LUTs: no division or modulo in the hot
//!   path, ever.
//! - FMA accumulation via [`f64::mul_add`].
//! - skip-zero-byte fast path: a byte of all-zero trits is skipped before
//!   touching its five accumulators.
//!
//! ```
//! use gaia_mlx_quant::{pack_ternary_into, ternary_matmul};
//!
//! let rows = 4usize;
//! let cols = 5usize;
//! let trits: [i8; 20] = [
//!     1, -1, 0, 1, -1,
//!     0, 1, 1, -1, 0,
//!     -1, 0, 0, 1, 1,
//!     1, 1, -1, 0, 0,
//! ];
//! let mut packed = [0u8; 4];
//! pack_ternary_into(&trits, &mut packed);
//!
//! let x = [1.0, 2.0, 3.0, 4.0];
//! let mut out = [0f64; 5];
//! ternary_matmul(&packed, rows, cols, &x, None, 1.0, &mut out);
//! // out is the masked accumulation of trits · x
//! # assert_eq!(out[0], 2.0); // (1*1 + 0*2 + -1*3 + 1*4) ... see the tests
//! ```

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
extern crate alloc;

pub mod engram;
pub mod gaia;
pub mod kernel;
pub mod mlx;
pub mod pack;

pub use engram::{
    Bfp64HeadRow, ENGRAM_DEFAULT_HC_MULT, ENGRAM_HEAD_DIM, Mem8AdaptiveHeadRow, PrimeBucketMod,
    engram_bfp64_int_aligned_matmul_into, engram_bfp64_prequant_matmul_into,
    engram_gather_bfp64_heads_into, engram_hash_ngram_mod_into, engram_hash_ngram_pq_into,
    engram_hc_gate, engram_mem8_adaptive_matmul_into, engram_step_bfp64_into,
    engram_step_dense_fp64_into,
};

pub use gaia::{
    ENTROPY_REST, GRAVITY, Layer, PHI, RESONANCE, WEATHER_REST, Fold, Sampling, fold,
    fold_spectrum, sampling, seed_from,
};
pub use kernel::{
    dense_matmul, pack_ternary_i_major_into, pack_ternary_planes_i_into, relative_error,
    ternary_matmul, ternary_matmul_i_major, ternary_matmul_planes_i,
};
#[cfg(feature = "std")]
pub use kernel::{Quantized, quantize_ternary};
pub use mlx::{
    MLX_GROUP_SIZE, MLX_GROUP64_COMPRESSION, PQ_DEFAULT_Q, Pq, QuantMode, StridedView,
    TRITS_PER_U32, b158_matmul_pq_i_major_into, gather_qmm_into, gather_qmm_pq_into,
    maybe_matmul_into, mergeq_matmul_into, mlx_code_of, mlx_dequantize_group64_into,
    mlx_matmul_pq_group64_into, mlx_quantize_group64_into, mlx_quantized_matmul_into, mlx_trit_of,
    quantize_bfp_i8_group64_into, quantize_bfp_i16_group64_into, quantize_pq_adaptive_into,
    quantize_pq_into, rms_norm_into, rope_into, sdpa_pq_single_query_into, sdpa_single_query_into,
    ste_linear_backward_into, swiglu_into, ternary_bfp_adaptive_mem8_matmul_into,
    ternary_bfp_i8_matmul_into, ternary_bfp_i16_matmul_into, ternary_simd_pq_matmul_into,
    ternary_transformer_step_into,
};
#[cfg(feature = "std")]
pub use mlx::MlxQuantizedMatrix;
pub use pack::{
    BITS_PER_WEIGHT, COMPRESSION_VS_F32, POW3, TRITS_PER_BYTE, digit_of, pack_ternary_into,
    trit_of, unpack_ternary_into,
};
#[cfg(feature = "std")]
pub use pack::{pack_ternary, unpack_ternary};
