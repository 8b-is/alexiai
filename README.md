<div align="center">

# ALEXIAI<span style="color:#ff5c8a">&lt;3</span>

### Omni edition · the fully offline AI app on the GAIA-MLX-QUANT substrate

**Zero third-party LLM usage · zero cloud · zero telemetry · zero runtime dependencies**

<img src="./web/icon.svg" alt="ALEXIAI mark" width="140"/>

*one process, one port, no network — the sovereign app, at performance, with love*

</div>

---

## ⚡ Quick Start

```bash
git clone https://github.com/8b-is/alexiai.git && cd alexiai

node src/cli.js serve        # the app → http://127.0.0.1:8787
node src/cli.js doctor       # field readout + sovereignty check
node src/cli.js bench        # ternary vs dense, measured honestly
node --test test/*.test.js   # 61 assertions, all green
```

That's the entire install. There is no `npm install`, no build step, no
runtime dependency — the app **is** Node's standard library plus what ships in
this tree. Point it at any Apple Silicon MLX server that speaks the
OpenAI-compatible surface on loopback (`--endpoint http://127.0.0.1:1337`) and
it is a complete, fully offline assistant.

## 🔒 The Guarantee (enforced, not promised)

> **Inference never leaves this machine.**

This is not a policy note. `src/sovereign.js` is an execution point: it wraps
`fetch` process-wide and rejects any request whose host is not loopback,
*before a socket opens*. There is no vendor allowlist, no env-var escape
hatch, no "but this one is fine". Loopback or nothing.

```bash
$ node src/cli.js doctor
  sovereignty
    policy     inference never leaves this machine
    guard      installed            # while the app runs
    loopback   127.0.0.1, ::1, localhost, ...

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
$ node src/cli.js bench --dim 256
bitsPerWeight   1.5850
compression     20.18× vs float32
relativeError   0.36   # honest number — random projections are the worst case
```

The bench prints the *measured* error, including the worst case. That number
is the honesty of this project: on adversarial random weights, ternary
reconstruction is lossy by design; on the ternary-shaped weights a trained
b1.58 model actually has, the projection is tight (`< 0.34`, pinned by test).

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
├── src/gaia.js         the field: fold, layers, sampling
├── src/mlx-quant.js    ternary packing + the kernel
├── src/sovereign.js    the egress guard (the enforcement point)
├── src/osarous.js      the local model adapter
├── src/server.js       the offline HTTP server
├── src/cli.js          serve · doctor · models · chat · gaia · bench
├── web/                the offline UI
├── test/               61 assertions, node:test
├── docs/               field · kernel · sovereignty theory
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

## 🤝 Contribute

```bash
git clone https://github.com/8b-is/alexiai.git
node --test test/*.test.js     # green before anything else
```

Rules live in `AGENTS.md`. The one rule above all: **loopback or nothing.**

---

*the constellation · 0 + 1 · fine touch from within · vaked.dev* · {<3,<3,<3}+1
