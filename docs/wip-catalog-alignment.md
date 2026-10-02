# WIP-catalog + SoftBank alignment

ALEXIAI is not a new lane on the map — it is two existing rows of Nate's
WIP catalog wearing one binary. This file is the wiring receipt.

## the catalog rows

From `8b-is/raw_research/wip-catalog-100.md` (Nate `@standardgalactic`,
2026-09-06, source of truth):

| # | WIP item | where it lives in ALEXIAI |
|---|----------|---------------------------|
| 71–72 | the adjacent cognitive-substrate items | `rust/gaia-mlx-quant` — the native lane (no_std, zero-unsafe, zero-alloc) |
| 99 | FMove / Frog | `go/golue` — the glue that moves the lanes |
| 100 | Mutual / Concurrent Cognition | the four lanes agreeing on one math: Rust core, Go glue, C99+ASM, Swift+CoreMIDI |

## the cross-map row (as recorded in the catalog)

> 71, 72 | ALEXIAI / GAIA-MLX-QUANT | `8b-is/alexiai` — the fully offline
> sovereign app: GAIA field, ternary b1.58 substrate (Rust core · Go glue ·
> C99+ASM · Swift+CoreMIDI), loopback-only egress

## the SoftBank draft

`softbank-outreach-v1.0.md` (workspace root) pitches the constellation's
thesis: *the next inflection is not a bigger model but a verifiable one —
systems that can prove their continuity and keep their word.*

ALEXIAI is that thesis, compiled:

| draft sentence | alexiai enforcement |
|---|---|
| "states honestly what it holds" | the bench reports measured error, including the worst case |
| "admits what it does not know" | a down model server is a fact, not an exception |
| "can refuse rather than confabulate" | `SovereigntyViolation` — loopback or nothing, before any socket opens |

The same three guarantees, in four languages, pinned by 92 tests across the
workspace. The pitch stays a pre-draft until 8b-is review; the code does not
wait for the meeting.

*the constellation · 0 + 1 · fine touch from within · vaked.dev*
