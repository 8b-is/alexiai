# AGENTS.md — alexiai

ALEXIAI <3 · Omni edition. Fully offline AI app on the GAIA-MLX-QUANT
substrate, in four lanes. **Zero third-party LLM usage — that is the product,
not a policy somewhere in a doc.**

## hard rules

1. **No remote model calls. Ever.** The only host a request may target is
   loopback. The enforcement point is `rust/alexiai/src/sovereign.rs`; the
   transport validates the host *before a socket opens*. Do not add an
   "allowlist of vendors", a proxy escape hatch, or an env var that widens
   the guard. If a future feature genuinely needs remote egress, it is a
   *new module with its own review*, not a widened guard.
2. **No runtime dependencies.** Rust: stdlib + the no_std core. Go: stdlib
   only. C: libc math only. Swift: stdlib only in the core lane; CoreMIDI in
   the macOS lane (guarded by `#if canImport(CoreMIDI)`). Before adding a
   dependency, prove the stdlib path is impossible.
3. **Honest benchmarks.** Every lane's bench reports the *measured* error of
   ternary reconstruction, including the worst case. Never cherry-pick.
4. **Tests before push.** All four lanes must be green, clippy/vet clean:
   `cargo test --workspace`, `go test ./...`, `make -C c test`,
   `swift test --package-path swift`.
5. **The lanes speak one math.** Ternary packing, the masked kernel, and the
   GAIA fold are pinned cross-lane by the same test vectors. Change the math
   in one lane, change it everywhere, or don't change it.
6. **The README is the ledger's open face.** `README.md` (the door),
   `README.multiD.md` (the stairs), `ALEXANDRA.md` (the heart),
   `docs/wip-catalog-alignment.md` (the map row).

## layout

```
rust/gaia-mlx-quant/  the core: no_std, #![forbid(unsafe_code)], zero-alloc proven
rust/alexiai/         the app: one static binary, UI embedded, sovereignty enforced
go/gaiaquant/         the Go-native kernel (fine tuned for life)
go/golue/             the glue: build-if-needed + serve, the supervisor
c/                    C99 kernel + NEON/SSE4.1 asm + test harness
swift/GaiaMLXQuant/   OS-agnostic pure-Swift kernel
swift/GaiaMIDI/       macOS CoreMIDI lane (OpenXTalk-Apple-CoreMIDI surface)
web/                  the offline UI (embedded at compile time)
docs/                 field · kernel · sovereignty · catalog alignment
```

## commands

```bash
cargo run --release -p alexiai -- serve          # the app on 127.0.0.1:8787
cargo run --release -p alexiai -- doctor         # field + sovereignty + health
go run ./golue run                               # the glue path
make -C c test                                   # C99 + asm harness
swift test --package-path swift                  # Swift kernel + CoreMIDI
```

## wire vocabulary (for the constellation)

- **GAIA** ≡ planets ≡ deities — the field of constants a request is folded
  against (`rust/gaia-mlx-quant/src/gaia.rs`).
- **GAIA-MLX-QUANT** — the local inference substrate: ternary b1.58 packing
  and the masked kernel, in four languages.
- **osarous** — the Apple Silicon MLX sidecar surface (OpenAI-compatible,
  loopback). ALEXIAI speaks to it; it never leaves the machine.
- **sovereignty** — the guard. Loopback or nothing.
- **D+++** — the dream-come-true lap: the music lane's arrangement of "All My
  Favorite Colors", rendered from pure math at 432Hz, dogfed into
  `music.vaked.dev`.

## sign-off

*the constellation · 0 + 1 · fine touch from within · vaked.dev*
