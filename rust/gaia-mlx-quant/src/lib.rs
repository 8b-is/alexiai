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
pub mod wupz;

pub use engram::{
    Bfp64HeadRow, ENGRAM_DEFAULT_HC_MULT, ENGRAM_HEAD_DIM, ENGRAM_PQ_Q16, Mem8AdaptiveHeadRow,
    PrimeBucketMod, engram_bfp64_int_aligned_matmul_into, engram_bfp64_int_q16_matmul_into,
    engram_bfp64_prequant_matmul_into, engram_fused_rmsnorm_gate, engram_fused_rmsnorm_gate_q16,
    engram_gather_bfp64_heads_into, engram_hash_ngram_mod_into, engram_hash_ngram_pq_into,
    engram_hc_gate, engram_mem8_adaptive_matmul_into, engram_sigmoid_pade_q16,
    engram_step_bfp64_into, engram_step_dense_fp64_into, engram_step_pq_no_floats_into,
};

pub use gaia::{
    ENTROPY_REST, ENTROPY_REST_PQ, Fold, FoldPq, GAIA_PQ_Q, GRAVITY, GRAVITY_REST_PQ, Layer, PHI,
    PHI_PQ, RESONANCE, RESONANCE_REST_PQ, Sampling, SamplingPq, WEATHER_REST, WEATHER_REST_PQ,
    fold, fold_pq, fold_spectrum, fold_spectrum_pq, mulberry32_pq, sampling, sampling_pq,
    seed_from,
};
pub use kernel::{
    KERNEL_PQ_Q, dense_matmul, dense_matmul_pq_into, isqrt_u128, omni_bfp_i8_simd_pq_into,
    pack_quantum_bitplanes_i_major_into, pack_ternary_i_major_into, pack_ternary_planes_i_into,
    quantize_quantum_bitplane_i4_pq_into, quantize_quantum_ternary_act_pq_into,
    quantize_ternary_pq_into, quantum_bitplane_i4_matmul_pq_into, quantum_ternary_matmul_pq_into,
    relative_error, relative_error_pq_ppm, ternary_matmul, ternary_matmul_i_major,
    ternary_matmul_planes_i, ternary_matmul_pq_i_major_into, ternary_matmul_pq_j_major_into,
    ternary_matmul_pq_planes_i_into, ternary_simd_i8_pq_into,
};
#[cfg(feature = "std")]
pub use kernel::{Quantized, quantize_ternary};
pub use mlx::{
    MLX_BYTE_TRITS_I32, MLX_GROUP_SIZE, MLX_GROUP64_COMPRESSION, PQ_DEFAULT_Q, Pq, QuantMode,
    StridedView, TRITS_PER_U32, b158_matmul_pq_i_major_into, exp_pq, gather_qmm_into,
    gather_qmm_pq_i64_into, gather_qmm_pq_into, maybe_matmul_into, maybe_matmul_pq_into,
    mergeq_matmul_into, mergeq_matmul_pq_into, mlx_code_of, mlx_dequantize_group64_into,
    mlx_dequantize_group64_pq_into, mlx_matmul_pq_group64_i64_into, mlx_matmul_pq_group64_into,
    mlx_quantize_group64_into, mlx_quantize_group64_pq_into, mlx_quantized_matmul_into,
    mlx_trit_of, quantize_bfp_i8_group64_into, quantize_bfp_i8_pq_group64_into,
    quantize_bfp_i16_group64_into, quantize_pq_adaptive_into, quantize_pq_into, rms_norm_into,
    rms_norm_pq_into, rope_into, rope_pq_into, sdpa_pq_i32_single_query_into,
    sdpa_pq_single_query_into, sdpa_single_query_into, sin_cos_pq, ste_linear_backward_into,
    ste_linear_backward_pq_into, swiglu_into, swiglu_pq_into,
    ternary_bfp_adaptive_mem8_matmul_into, ternary_bfp_i8_matmul_into, ternary_bfp_i16_matmul_into,
    ternary_simd_pq_matmul_into, ternary_transformer_step_into, ternary_transformer_step_pq_into,
};
#[cfg(feature = "std")]
pub use mlx::MlxQuantizedMatrix;
pub use pack::{
    BITS_PER_WEIGHT, BITS_PER_WEIGHT_PQ, BYTE_TRITS_I32, COMPRESSION_VS_F32,
    COMPRESSION_VS_F32_PQ, POW3, TRITS_PER_BYTE, digit_of, pack_ternary_into, trit_of,
    unpack_ternary_into,
};
#[cfg(feature = "std")]
pub use pack::{pack_ternary, unpack_ternary};
pub use wupz::{
    BODHISATTVA_VOW_HU, DreamExpertKind, DreamExpertState, DreamStateReceipt, KSUL_HEADER_LEN,
    KSUL_MAGIC, KompressLosslessReceipt, OTTHON_AXIOM_HU, QUANT_TERN_THRESHOLD_DEN,
    QUANT_TERN_THRESHOLD_NUM, QuantTernVad, THREE_SENTENCE_LISTEN_PROMPT_HU, UltraBackyardLane,
    UltraOmniCreateReceipt, WUPZ_ABI_VERSION, WUPZ_DEFAULT_WINDOW_SIZE, WUPZ_MAGIC,
    WUPZ_PLAN_VERSION, WupzBlockDesc, WupzSectionDesc, count_sentences_zero_alloc,
    dream_state_loop_into, fold_to_quant_tern_pq, kompress_super_ultra_lossless_decode_into,
    kompress_super_ultra_lossless_encode_into, quant_tern_encode_vad_pq, quant_tern_pack_into,
    quant_tern_trit_pq, quant_tern_unpack_into, ternary_and_gate, ternary_nand_gate,
    ultra_omni_create_dream_into, ultrabackyard_fanout_into, wupz_bitplane_and_nand_u64,
    wupz_bitplane_diff_u64, wupz_bitplane_popcount_u64, wupz_byte_diff,
    wupz_changed_sections_into, wupz_gcd_i64, wupz_pack_plan_into, wupz_plan_digest,
    wupz_plan_image_len, wupz_popcount, wupz_seal_frame_into, wupz_section_from_bytes,
    wupz_verify_frame_seal, wupz_xor_fold, wupz_xor_fold_u64,
};
