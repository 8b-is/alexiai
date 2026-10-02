/**
 * ALEXIAI <3 — Omni edition.
 *
 * A fully offline AI app on the GAIA-MLX-QUANT substrate. Public surface:
 *
 * - `gaia` — the field: four layers, folded deterministically per prompt
 * - `mlx-quant` — ternary BitNet b1.58 packing and the masked kernel
 * - `sovereign` — the loopback-only egress guard that makes "offline" enforced
 * - `osarous` — the local model adapter (OpenAI-compatible, per-request facts)
 * - `server` / `cli` — the app itself
 *
 * @module alexiai
 */

export { LAYERS, FIELD, INHABITANTS, fold, sampling, readout, rng, seedFrom, clamp } from './gaia.js';
export {
  TRITS,
  TRITS_PER_BYTE,
  quantizeTernary,
  packTernary,
  unpackTernary,
  planes,
  ternaryMatMul,
  denseMatMul,
  relativeError,
  bitsPerWeight,
  compressionVsFloat32,
  syntheticWeights,
} from './mlx-quant.js';
export {
  SovereigntyViolation,
  assertLocal,
  checkLocal,
  isLoopbackHostname,
  installEgressGuard,
  attestation,
} from './sovereign.js';
export {
  OsarousAdapter,
  DEFAULT_BASE_URL,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_TIMEOUT_MS,
  DONE,
  normalizeBase,
  buildBody,
  iterateSSE,
} from './osarous.js';
export { createServer, start, runBench } from './server.js';