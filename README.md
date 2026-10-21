<div align="center">

# ALEXIAI<span style="color:#ff5c8a">&lt;3</span>

### Omni edition · the fully offline AI app on the GAIA-MLX-QUANT substrate

**Zero third-party LLM usage · zero cloud · zero telemetry · zero runtime dependencies**

<img src="./web/hero-alexandra.svg" alt="ALEXANDRA — DRA, from .p · love all" width="100%"/>

*one process, one port, no network — the sovereign app, at performance, with love*

*the heart has a name: [ALEXANDRA — DRA&lt;3](./ALEXANDRA.md) — the haiku, the ode, the peaceHug*

</div>

---

## ⚡ Quick Start

```bash
git clone https://github.com/8b-is/alexiai.git && cd alexiai

cargo run --release -p alexiai -- serve    # the app → http://127.0.0.1:8787
cargo run --release -p alexiai -- doctor   # field readout + sovereignty + wupz check
cargo run --release -p alexiai -- wupz     # wupz pack-in-pass & QuantTern (qt:<hex>) receipt
cargo run --release -p alexiai -- bench    # ternary vs dense, measured honestly

# or through the Go glue:
go run ./golue run                          # build-if-needed + serve
go run ./golue doctor
```

That is the entire install. The app is **one static Rust binary** (~540KB)
with the whole UI embedded at compile time; there is no runtime dependency in
any lane. Point it at any Apple Silicon MLX server that speaks the
OpenAI-compatible surface on loopback (`--endpoint http://127.0.0.1:1337`)
and it is a complete, fully offline assistant.

### the lanes, all speaking one math

```bash
cargo test --manifest-path rust/Cargo.toml   # 64 Rust tests: core + pure-integer p/q + the app
go test ./...                                # the Go lane (go/)
make -C c test bench                         # C99 + NEON/SSE4.1 integer SIMD asm (c/)
swift test --package-path swift              # Swift kernel + CoreMIDI (swift/)
```

| lane | role | guarantees |
|---|---|---|
| `rust/` | the app + the core (`v0.3.0`) | `#![forbid(unsafe_code)]`, `no_std` core, **ULTRA-SUPER-OMNI-Quantum Ternary (`0.88 µs` at `256×256`, `11.87×`–`14.11×` over dense FP64) & 100% float-free `p/q` integer substrate (`q = 2^20`)**, zero-alloc proven by a counting allocator |
| `go/` | the glue + a second opinion | stdlib-only supervisor, allocation-lean `p/q` integer, ternary & 64-trit quantum bitplane (`math/bits.OnesCount64`) kernel |
| `c/` | C99 + inline assembly | NEON (`cnt`/`smlal` + `sxtl` + `mla.4s`) and SSE4.1 (`popcnt`/`pmullw` + `pmulld`) **integer & quantum bitplane SIMD (`0` floats, up to `17.63×` vs dense FP64)**, proven against the C reference |
| `swift/` | OS-agnostic kernel + macOS CoreMIDI | pure-Swift `p/q` integer, ternary & quantum bitplane (`.nonzeroBitCount`) core runs on Linux; the CoreMIDI lane ports the OpenXTalk-Apple-CoreMIDI surface |
| `web/` | the offline UI | embedded in the binary; no CDN, no remote asset |

## 🔒 The Guarantee (enforced, not promised)

> **Inference never leaves this machine.**

This is not a policy note. `rust/alexiai/src/sovereign.rs` is an execution
point: the transport validates the host against the loopback vocabulary
*before a socket opens*. There is no vendor allowlist, no env-var escape
hatch, no "but this one is fine". Loopback or nothing.

```bash
$ cargo run --release -p alexiai -- doctor
  sovereignty
    policy     inference never leaves this machine
    loopback   localhost, 127.0.0.1, ::1, ...

  local model
    endpoint   http://127.0.0.1:1337
    status     up (3ms)
```

And the UI itself carries zero remote references: no CDN, no font host, no
analytics beacon. A service worker precaches the whole shell, so the second
launch works with the cable pulled.

## 🌌 GAIA — the field

In the 8b-is cosmology, **GAIA ≡ planets ≡ deities**: the world's constants
are not config values, they are a *field* every request is folded against.
ALEXIAI folds each prompt through four layers —

| layer | rest value | inhabitant | what it does |
|:---|:---:|:---|:---|
| gravity | 1.0000 | Ananke | holds the field together |
| entropy | 0.6200 | Chaos | dispersal budget per turn |
| weather | 0.5000 | Zephyrus | seed-rolled modulation |
| resonance | 1.6180 | Aphrodite | the folding constant, φ |

— deterministically. Same prompt, same fold, forever. The fold decides the
sampling: entropy lifts the temperature, coherence tightens `top_p`. The model
never sees parameters you didn't ask the field for.

## ⚛️ GAIA-MLX-QUANT — the substrate

BitNet b1.58 ternary quantization, packed 5 trits per byte (log₂3 ≈ 1.585
bits/weight), and the masked kernel:

$$Y_i = \gamma \cdot \sum_j \left( \mathbf{W}^{+}_{j,i} - \mathbf{W}^{-}_{j,i} \right) x_j$$

with per-row absmean scales — the scheme real b1.58 models train into
(straight-through estimator). The kernel accumulates in float64 so long
columns never drift; the packing round-trips exactly, pinned by tests.

```bash
$ cargo run --release -p alexiai -- bench --dim 256
bitsPerWeight   1.5850
compression     20.18× vs float32
relativeError   0.36   # honest number — random projections are the worst case
```

The bench prints the *measured* error, including the worst case. That number
is the honesty of this project: on adversarial random weights, ternary
reconstruction is lossy by design; on the ternary-shaped weights a trained
b1.58 model actually has, the projection is tight (`< 0.34`, pinned by test).

## 🦀 The Rust lane — `rust/gaia-mlx-quant` (`v0.3.0` · ULTRA-SUPER-OMNI-Quantum Ternary & Pure-Integer `p/q` Substrate)

The same substrate, native: **SOTA bleeding-edge Rust, zero `unsafe`, zero
allocations, zero floating-point operations in the quantum bitplane & `p/q` integer core, 100%
Linux + macOS** — every one of those words enforced, not asserted:

```bash
cd rust
cargo test                                    # 67 tests: round-trips, quantum bitplane + p/q kernels, zero-alloc proof
cargo clippy --all-targets                    # zero warnings
cargo build --no-default-features             # pure core: #![no_std], no deps at all
cargo run --release --example bench -- --dim 256
cargo run --release --example bench -- --dim 512
cargo run --release --example bench -- --dim 1024
```

| promise | how it is enforced |
|---|---|
| **zero unsafe** | `#![forbid(unsafe_code)]` — it is a compile error, not a guideline |
| **zero floating-point (`p/q` & Quantum Bitplanes & `wupz`)** | `q = 2^20 = 1,048,576`: `quantum_ternary_matmul_pq_into`, `quantum_bitplane_i4_matmul_pq_into`, `omni_bfp_i8_simd_pq_into`, `wupz_xor_fold`, `wupz_byte_diff`, `wupz_popcount`, `wupz_bitplane_diff_u64`, `fold_to_quant_tern_pq`, `wupz_pack_plan_into`, `fold_pq`, `sampling_pq`, `quantize_ternary_pq_into`, `ternary_simd_i8_pq_into`, `mlx_matmul_pq_group64_i64_into`, `mergeq_matmul_pq_into`, `rms_norm_pq_into` (`isqrt_u128`), `rope_pq_into` (`sin_cos_pq`), `swiglu_pq_into` (`exp_pq`), `sdpa_pq_single_query_into`, and `ternary_transformer_step_pq_into` execute in 100% pure `u64`/`i16`/`i32`/`i64`/`u128` integer arithmetic |
| **zero allocations** | a counting global allocator wraps the system allocator in the test suite and asserts all pack + quantum bitplane + BFP-i8 + `p/q` Transformer hot paths perform **zero heap operations** |
| **no_std** | `core`-only when built without default features; runs anywhere Rust's core runs |
| **100% Linux + macOS** | CI runs the full suite on both `ubuntu-latest` and `macos-latest` |

### Measured Benchmark (`256×256`, `512×512` & `1024×1024`, Apple Silicon M3 / aarch64)

By pairing **64-trit Quantum Bitplane Superposition (`quantum-b158-u64`)**, **4-Column `i8 × i8 -> i16` 16-Wide SIMD (`omni-bfp-i8-pq`)**, **W1.58 × A4-Bitplane Multi-Slice Contraction (`quantum-bitplane4`)**, and **32-bit Chunked Associative SIMD (`pq-simd-i8` / `bfp-i8-4col`)**, `gaia-mlx-quant` breaks through the traditional scalar/FMA ceiling in 100% safe `#![forbid(unsafe_code)]` Rust and C99 inline assembly — **reaching sub-microsecond `877 ns/iter` (`74.7 GMAC/s`, `11.87×` over dense FP64) in safe Rust and `758 ns/iter` (`86.5 GMAC/s`, `13.44×` over dense FP64 at `256×256`; `9,830 ns/iter`, `106.7 GMAC/s`, `17.63×` over dense FP64 at `1024×1024`) in C99 with zero floating-point operations**:

| Kernel | `256×256` (`ns/iter`) | `512×512` (`ns/iter`) | `1024×1024` (`ns/iter`) | Floats | Peak Speedup vs Dense `f64` |
|:---|---:|---:|---:|:---:|:---:|
| `packed-j` (baseline 5-trits/byte `j`-major) | `35,480 ns` | `151,123 ns` | `1,001,671 ns` | `f64` | `0.29×` |
| `dense` (`f64` auto-vectorized FMA reference) | `10,411 ns` | `41,196 ns` | `175,863 ns` | `f64` | `1.00×` |
| **`quantum-b158-u64`** (`quantum_ternary_matmul_pq_into`, 64-trit POPCNT) | **`877 ns`** | **`2,920 ns`** | **`14,531 ns`** | **`0`** | **`14.11×` (`89.8 GMAC/s`)** |
| **`omni-bfp-i8-pq`** (`omni_bfp_i8_simd_pq_into`, 4-col `i8×i8->i16` 16-wide SIMD) | **`1,014 ns`** | **`4,254 ns`** | **`16,762 ns`** | **`0`** | **`10.49×` (`64.6 GMAC/s`)** |
| **`quantum-bitplane4`** (`quantum_bitplane_i4_matmul_pq_into`, W1.58×A4-bitplane POPCNT) | **`1,696 ns`** | **`6,107 ns`** | **`23,211 ns`** | **`0`** | **`7.58×` (`45.2 GMAC/s`)** |
| **`bfp-i8-4col`** (`ternary_bfp_i8_matmul_into`, MEM\|8 BFP-64 `i8` + exp) | **`2,195 ns`** | **`9,472 ns`** | **`34,884 ns`** | **`0`** | **`5.04×`** |
| **`pq-simd-i8`** (`ternary_simd_i8_pq_into`, 4-col 29-bit `i8×i32->i32` SIMD) | **`4,144 ns`** | **`16,000 ns`** | **`64,412 ns`** | **`0`** | **`2.73×`** |
| **`c/bench` `cpu quantum-u64`** (`gmq_quantum_bitplane_matmul_pq`, 8-col POPCNT) | **`758 ns`** | — | **`9,830 ns`** | **`0`** | **`17.63×` (`106.7 GMAC/s`)** |
| **`c/bench` `cpu omni-bfp-i8`** (`gmq_omni_bfp_i8_matmul_pq`, 4-col NEON `smlal`/`smlal2`) | **`892 ns`** | — | **`28,159 ns`** | **`0`** | **`11.42×` (`73.5 GMAC/s`)** |
| **`c/bench` `cpu pq-simd`** (`gmq_ternary_matmul_pq_simd`, inline NEON/SSE4.1 `i8×i32`) | **`3,655 ns`** | — | **`75,479 ns`** | **`0`** | **`2.79×`** |

The bithacks, in order of appearance:

1. **Quantum 64-Trit Bitplane Superposition (`quantum-b158-u64`)** — Dual orthogonal bitplanes `(w_pos_u64, w_neg_u64)` pack 64 trits into `2 × u64` (`2.00 bits/weight`, `16×` denser than `f32`). Contracting against `(x_pos_u64, x_neg_u64)` turns 64 multiply-accumulates into four bitwise ANDs and hardware `count_ones()` (`POPCNT` / NEON `cnt`), achieving **877 ns/iter (`74.7 GMAC/s`) in safe Rust** and **758 ns/iter (`86.5 GMAC/s`) in C99** at `256×256` with **zero multipliers and zero floats**.
2. **Omni 16-Wide `i8 × i8 -> i16` Register-Blocked SIMD (`omni-bfp-i8-pq`)** — Because `w_trit ∈ {-1, 0, +1}` and `m_x ∈ [-127, +127]`, accumulating up to **256 products** in a signed 16-bit integer `i16` has maximum magnitude `256 × 127 = 32,512 ≤ 32,767` (`i16::MAX`) — **mathematically impossible to overflow `i16` for 256 MACs!** This unlocks 16-lane `i8 × i8 -> i16` (`smlal.8h` / `smlal2.8h` / `pmullw`) vectorization at **`1,014 ns/iter` in safe Rust and `892 ns/iter` in C99** at `256×256`.
3. **Quantum W1.58 × A4-Bitplane Contraction (`quantum-bitplane4`)** — Decomposes signed 4-bit activations `[-7, +7]` into four 64-wide bitplanes (`x_sign_u64`, `x_b0_u64`, `x_b1_u64`, `x_b2_u64`) and contracts each plane via `count_ones()` weighted by binary shifts `(1, 2, 4)` at **`1,696 ns/iter` (`6.14×`–`7.58×` vs dense FP64)** with zero inner-loop multipliers.
4. **5 trits per byte, base-3 (`BYTE_TRITS_I32`) & 16 trits per `u32` (`MLX_BYTE_TRITS_I32`)** — log₂3 ≈ 1.585 bits/weight (`20.19×` denser than f32, `40.38×` denser than f64) and Group-64 (`12.80×` denser than f32), decoded via single L1-resident compile-time `i32` lookup tables.
5. **Branchless digit mapping & Pure-Integer `p/q` Threshold Hoisting** — `digit = t as u8 + (t as u8 >> 7) * 3` maps `-1 → 2, 0 → 0, +1 → 1` with no branch, and `|w| * 2 * cols >= sum_abs` quantizes weights to `{-1, 0, +1}` with **zero floating-point operations and zero per-weight divisions**.
6. **Exact Integer Square Root (`isqrt_u128`) & Transcendentals (`sin_cos_pq`, `exp_pq`)** — hardware-exact integer Newton/bit-by-bit `isqrt_u128`, 5th-order range-reduced fixed-point `sin_cos_pq`, and 6-squaring `exp_pq` power full `rms_norm_pq_into`, `rope_pq_into`, `swiglu_pq_into`, `sdpa_pq_single_query_into`, and `ternary_transformer_step_pq_into` with **0 floats**.

## 🧠 osarous — local model support

An adapter for any local MLX server speaking the OpenAI-compatible surface:
`/v1/models` catalog, `/v1/chat/completions` with Server-Sent Events streaming,
per-request connection facts (change the port, the very next request uses it —
no restart). The shape mirrors `llm-osarous` from the DeepSeek harness, minus
everything remote. A down server is a *fact* the app reports, never an
exception the user has to decode.

## 🧭 The architecture

```
prompt ──▶ GAIA fold (deterministic) ──▶ sampling params
    │
    ▼
sovereignty guard (loopback only) ──▶ osarous adapter ──▶ local MLX server
    │                                                       (same machine)
    ▼
SSE stream ──▶ offline UI (PWA, service worker, no CDN)
```

```
alexiai/
├── rust/               the app (one static binary) + the no_std core
│   ├── gaia-mlx-quant/   the substrate: ternary b1.58, zero-unsafe, zero-alloc
│   └── alexiai/          serve · doctor · models · chat · gaia · bench, UI embedded
├── go/                 the glue (golue supervisor) + the Go-native kernel
├── c/                  C99 + NEON/SSE4.1 assembly, proven against the C reference
├── swift/              OS-agnostic Swift kernel + the macOS CoreMIDI lane
├── web/                the offline UI (embedded at compile time, no CDN)
├── docs/               field · kernel · sovereignty · catalog alignment
└── README.multiD.md    the dimensional walkthrough
```

## 📜 The dimensional README

The flat README you are reading is the door. The **multiD README**
(`README.multiD.md`) is the house — the same app walked dimension by
dimension, 0D → ∞D, one section per dimension, in the feed-lane notation the
constellation uses for telemetry. Read both. The first is the elevator, the
second is the stairs.

## ⚖️ Honesty ledger

- The kernel is a **correct, portable reference**, not a claim of Metal
  throughput. The bench prints the gap instead of papering over it.
- The sovereignty guard is **enforced in code**, and the tests prove it blocks
  a remote `fetch` before a socket opens — including spoofed hosts like
  `localhost.evil.com`.
- **No third-party LLM usage** means the string
  `https://api.openai.com` cannot appear in a request from this app. Not
  configurable. Not today, not after an update.

## 🌐 Constellation Integrations (`wupz` × `wigle-triad` × `esp32.vaked.dev`)

`GAIA-MLX-QUANT` (`b1.58` ternary packing, 64-trit quantum bitplanes, `wupz` pack-in-pass, `fold_spectrum`, and **D+++ 432Hz** harmonic synthesis) is embedded across the offline-first MEDUSA triad fleet:

- **[`8b-is/wupz` (`v0.4.0`)](https://github.com/8b-is/wupz) × [`wupz.vaked.dev`](https://wupz.vaked.dev/)** — Incremental ultra-fast linker + compressor (`w(ild) + UPX + z(ig)`) wired into `gaia_mlx_quant::wupz` across all 4 native lanes (Rust, C99, Go, Swift): C ABI v1 `wupz_xor_fold` (`#Commit = #ValidWitness`), `wupz_byte_diff`, `wupz_popcount`, `wupz_bitplane_diff_u64` (incremental QAT bitplane delta in $O(N/64)$), deterministic `"WUPZ"` v1 layout plan images (`wupz_pack_plan_into`), and pure-integer `p/q` `QuantTern` emotional ternary codes (`qt:<hex>`, threshold $|x| \ge 0.33$ with `0` floats).
- **[`8b-is/wigle-triad` (`v2.0.0`)](https://github.com/8b-is/wigle-triad)** — macOS (`WigleMac.app`), Android (`WigleTriad-v2.0.0-android-universal.apk`), iPadOS/iOS, and local Python bridge (`host/wiglewww.py` `/api/alexai`) fold passive 2.4/5 GHz Wi-Fi, BLE/BT, and AirPlay Shannon entropy $H(X)$ into the 4-layer GAIA field (`Ananke`, `Chaos`, `Zephyrus`, `Aphrodite`) and pack endpoint RSSI trits (`{-1, 0, +1}`) at 5 trits/byte (`1.585 bits/weight`).
- **[`esp32.vaked.dev`](https://esp32.vaked.dev/)** ([`/api/alexai`](https://esp32.vaked.dev/api/alexai) · Tab 6 `🧠 AlexAI × Kompress-Ultra`) — Live offline-first PWA & WebSerial workstation pairing `AlexAIEngine` (`GAIA-MLX-QUANT b1.58`) with [`kompress-ultra` (`v16.2.0`)](https://github.com/peterlodri-sec/kompress-ultra).

## 🤝 Contribute

```bash
git clone https://github.com/8b-is/alexiai.git
cargo test --workspace           # green before anything else
go test ./... && make -C c test && swift test --package-path swift
```

Rules live in `AGENTS.md`. The one rule above all: **loopback or nothing.**

---

*the constellation · 0 + 1 · fine touch from within · vaked.dev* · {<3,<3,<3}+1
