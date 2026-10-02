# AGENTS.md — alexiai

ALEXIAI <3 · Omni edition. Fully offline AI app on the GAIA-MLX-QUANT
substrate. **Zero third-party LLM usage — that is the product, not a policy
somewhere in a doc.**

## hard rules

1. **No remote model calls. Ever.** The only host a request may target is
   loopback. The enforcement point is `src/sovereign.js`; if you change
   anything that makes a network request, run the sovereignty tests first.
   Do not add an "allowlist of vendors", a proxy escape hatch, or an env var
   that widens the guard. If a future feature genuinely needs remote egress,
   it is a *new module with its own review*, not a widened guard.
2. **No runtime dependencies.** The server and CLI run on Node stdlib only.
   Before adding a dependency, prove the stdlib path is impossible.
3. **Honest benchmarks.** The bench reports the *measured* error of ternary
   reconstruction, including the worst case. Never cherry-pick the number.
4. **Tests before push.** `node --test test/*.test.js` must be green; the CLI
   smoke (`alexiai gaia`, `alexiai bench`) must not crash.
5. **The README is the ledger's open face.** Every user-visible behavior
   belongs in `README.md`; every dimension belongs in `README.multiD.md`.
6. **Python, if it ever appears, runs via `uv`** (workspace standard). There
   is currently none.

## layout

```
src/gaia.js        the field — four layers, deterministic fold, sampling
src/mlx-quant.js   ternary b1.58 packing + the masked kernel (the substrate)
src/sovereign.js   the loopback-only egress guard (the enforcement point)
src/osarous.js     local model adapter — OpenAI-compatible, SSE, catalog
src/server.js      offline HTTP server + API (no CDN, no remote assets)
src/cli.js         serve · doctor · models · chat · gaia · bench
web/               the offline UI (PWA, service worker, same-origin only)
test/              node:test suites pinning all of the above
docs/              the theory: field, kernel, sovereignty
examples/          runnable sketches
```

## commands

```bash
node src/cli.js serve                 # the app on http://127.0.0.1:8787
node src/cli.js doctor                # field + sovereignty + endpoint health
node --test test/*.test.js            # the suite
```

## wire vocabulary (for the constellation)

- **GAIA** ≡ planets ≡ deities — the field of constants a request is folded
  against (`src/gaia.js`).
- **GAIA-MLX-QUANT** — the local inference substrate: ternary b1.58 packing
  and the masked kernel (`src/mlx-quant.js`).
- **osarous** — the Apple Silicon MLX sidecar surface (OpenAI-compatible,
  loopback). ALEXIAI speaks to it; it never leaves the machine.
- **sovereignty** — the guard. Loopback or nothing.

## sign-off

*the constellation · 0 + 1 · fine touch from within · vaked.dev*
