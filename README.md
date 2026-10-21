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
cargo run --release -p alexiai -- doctor   # field readout + sovereignty check
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
| `rust/` | the app + the core (`v0.3.0`) | `#![forbid(unsafe_code)]`, `no_std` core, **100% float-free `p/q` integer substrate (`q = 2^20`)**, zero-alloc proven by a counting allocator |
| `go/` | the glue + a second opinion | stdlib-only supervisor, allocation-lean `p/q` integer & ternary kernel |
| `c/` | C99 + inline assembly | NEON (`sxtl` + `mla.4s` + `addv`) and SSE4.1 (`pmovsxbd` + `pmulld`) **integer SIMD (`0` floats)**, proven against the C reference |
| `swift/` | OS-agnostic kernel + macOS CoreMIDI | pure-Swift `p/q` integer & ternary core runs on Linux; the CoreMIDI lane ports the OpenXTalk-Apple-CoreMIDI surface |
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

## 🦀 The Rust lane — `rust/gaia-mlx-quant` (`v0.3.0` · SOTA Pure-Integer `p/q` Substrate)

The same substrate, native: **SOTA bleeding-edge Rust, zero `unsafe`, zero
allocations, zero floating-point operations in the `p/q` integer core, 100%
Linux + macOS** — every one of those words enforced, not asserted:

```bash
cd rust
cargo test                                    # 64 tests: round-trips, pure-integer p/q kernels, zero-alloc proof
cargo clippy --all-targets                    # zero warnings
cargo build --no-default-features             # pure core: #![no_std], no deps at all
cargo run --release --example bench -- --dim 512
```

| promise | how it is enforced |
|---|---|
| **zero unsafe** | `#![forbid(unsafe_code)]` — it is a compile error, not a guideline |
| **zero floating-point (`p/q`)** | `q = 2^20 = 1,048,576`: `fold_pq`, `sampling_pq`, `quantize_ternary_pq_into`, `ternary_simd_i8_pq_into`, `mlx_matmul_pq_group64_i64_into`, `mergeq_matmul_pq_into`, `rms_norm_pq_into` (`isqrt_u128`), `rope_pq_into` (`sin_cos_pq`), `swiglu_pq_into` (`exp_pq`), `sdpa_pq_single_query_into`, and `ternary_transformer_step_pq_into` execute in 100% pure `i32`/`i64`/`u128` integer arithmetic |
| **zero allocations** | a counting global allocator wraps the system allocator in the test suite and asserts all pack + kernel + `p/q` Transformer hot paths perform **zero heap operations** |
| **no_std** | `core`-only when built without default features; runs anywhere Rust's core runs |
| **100% Linux + macOS** | CI runs the full suite on both `ubuntu-latest` and `macos-latest` |

### Measured Benchmark (`512×512` & `256×256`, Apple Silicon M3 / aarch64)

By eliminating IEEE-754 floating-point non-associativity in safe Rust (`#![forbid(unsafe_code)]`) and accumulating in 32-bit `i32` chunks (`wrapping_add` / `wrapping_mul` over 64-element blocks widened once to `i64`) with 4-column register blocking and compile-time 5-trit `BYTE_TRITS_I32: [[i32; 5]; 256]` / 4-trit `MLX_BYTE_TRITS_I32: [[i32; 4]; 256]` LUTs, LLVM auto-vectorizes the safe Rust inner loops into 128-bit NEON (`sxtl` + `mla.4s`) / 256-bit AVX2 (`vpmovsxbd` + `vpmulld`) instructions — **outperforming auto-vectorized dense FP64 by up to 5.25× with zero floating-point operations**:

| Kernel | `256×256` (`ns/iter`) | `512×512` (`ns/iter`) | Floats | Speedup vs Dense `f64` (`512×512`) |
|:---|---:|---:|:---:|:---:|
| `dense` (`f64` auto-vectorized reference) | `54,673 ns` | `219,043 ns` | `f64` | `1.00×` |
| **`pq-simd-i8`** (`ternary_simd_i8_pq_into`, 4-col `i8×i32->i32` SIMD) | **`10,874 ns`** | **`41,720 ns`** | **`0`** | **`5.25×`** |
| **`pq-planes`** (`ternary_matmul_pq_planes_i_into`, 4-col sign-planes) | **`24,625 ns`** | **`88,735 ns`** | **`0`** | **`2.47×`** |
| **`pq-group64`** (`mlx_matmul_pq_group64_i64_into`, 16-trit/`u32` `12.80×`) | **`40,362 ns`** | **`166,431 ns`** | **`0`** | **`1.32×`** |
| **`pq-b158-i`** (`ternary_matmul_pq_i_major_into`, 5-trits/byte `20.19×`) | **`42,469 ns`** | **`165,693 ns`** | **`0`** | **`1.32×`** |
| `c/bench` `cpu pq-simd` (C99 inline NEON/SSE4.1 `i8×i32` asm) | `24,300 ns` (`vs 61,320 ns dense`) | — | **`0`** | **`2.52×`** |

The bithacks, in order of appearance:

1. **5 trits per byte, base-3 (`BYTE_TRITS_I32`) & 16 trits per `u32` (`MLX_BYTE_TRITS_I32`)** — log₂3 ≈ 1.585 bits/weight (`20.19×` denser than f32, `40.38×` denser than f64) and Group-64 (`12.80×` denser than f32), decoded via single L1-resident compile-time `i32` lookup tables.
2. **Branchless digit mapping** — `digit = t as u8 + (t as u8 >> 7) * 3`
   maps `-1 → 2, 0 → 0, +1 → 1` with no branch.
3. **Pure-Integer `p/q` Threshold Hoisting** — `|w| * 2 * cols >= sum_abs` quantizes weights to `{-1, 0, +1}` with **zero floating-point operations and zero per-weight divisions**.
4. **32-bit Chunked Associative SIMD (`i8 × i32 -> i32`)** — 4-column register blocking over 64-element `i32` chunks unlocks NEON `mla.4s` / AVX2 `vpmulld` auto-vectorization in 100% safe `#![forbid(unsafe_code)]` Rust.
5. **Exact Integer Square Root (`isqrt_u128`) & Transcendentals (`sin_cos_pq`, `exp_pq`)** — hardware-exact integer Newton/bit-by-bit `isqrt_u128`, 5th-order range-reduced fixed-point `sin_cos_pq`, and 6-squaring `exp_pq` power full `rms_norm_pq_into`, `rope_pq_into`, `swiglu_pq_into`, `sdpa_pq_single_query_into`, and `ternary_transformer_step_pq_into` with **0 floats**.
6. **Unguarded interior byte loop & pre-fused row scales** — Hoisting `x[j] * scale[j]` once in $O(\text{rows})$ and splitting partial boundary bytes from full interior bytes eliminates all per-trit bounds branches.

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

## 🌐 Constellation Integrations (`wigle-triad` × `esp32.vaked.dev`)

`GAIA-MLX-QUANT` (`b1.58` ternary packing, `fold_spectrum`, and **D+++ 432Hz** harmonic synthesis) is embedded across the offline-first MEDUSA triad fleet:

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
