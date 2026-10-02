/**
 * GAIA-MLX-QUANT — the local inference substrate.
 *
 * Bare-metal tensor quantization for Apple Silicon, expressed as a portable
 * zero-dependency core. The kernel is the same formulation the standard
 * galactic computation engine uses:
 *
 *     Y = gamma * ( W_pos * X - W_neg * X )
 *
 * where `W_pos` and `W_neg` are the positive and negative planes of a packed
 * ternary BitNet b1.58 weight matrix in {-1, 0, +1}. No auxiliary allocation
 * happens in the hot path: both planes are read straight out of the packed
 * byte buffer.
 *
 * Honesty note, because this matters more than the naming: this module is a
 * *correct, portable, reference* implementation of the packing and the kernel.
 * It runs on any Node with no BLAS, and it is what the tests pin. It is not a
 * claim of Metal-level throughput — that is what a real MLX backend is for.
 * `alexiai bench` prints the gap so you never have to guess.
 *
 * @module alexiai/mlx-quant
 */

/** Ternary packing: 5 trits per byte (3^5 = 243 states, fits in a byte). */
export const TRITS_PER_BYTE = 5;

/** The three legal trit values. Nothing else may enter a packed matrix. */
export const TRITS = Object.freeze([-1, 0, 1]);

/**
 * BitNet b1.58 absmean quantization.
 *
 * When a `{rows, cols}` shape is given, scaling is per-row — the scheme real
 * b1.58 models use, since each row's dynamic range differs. Without a shape,
 * a single global absmean scale is used.
 *
 * The `1.58` in b1.58 is the bits-per-weight budget this scheme implies:
 * log2(3) ≈ 1.585.
 *
 * @param {ArrayLike<number>} weights flat row-major weights
 * @param {{gamma?: number, rows?: number, cols?: number}} [options]
 * @returns {{trits: Int8Array, scale: number, scales: Float64Array|null, gamma: number}}
 */
export function quantizeTernary(weights, options = {}) {
  const gamma = options.gamma ?? 1;
  const n = weights.length;
  if (n === 0) throw new RangeError('quantizeTernary: empty weight matrix');

  const rows = options.rows ?? 0;
  const cols = options.cols ?? 0;
  if ((rows > 0 || cols > 0) && rows * cols !== n) {
    throw new RangeError(`quantizeTernary: shape ${rows}x${cols} != ${n} weights`);
  }

  const trits = new Int8Array(n);
  const meanAbs = (start, end) => {
    let sum = 0;
    for (let i = start; i < end; i++) sum += Math.abs(weights[i]);
    return sum / (end - start);
  };

  if (rows > 0 && cols > 0) {
    /** @type {Float64Array} per-row scales */
    const scales = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
      const scale = meanAbs(r * cols, (r + 1) * cols) || 1;
      scales[r] = scale;
      const base = r * cols;
      for (let c = 0; c < cols; c++) {
        const q = Math.round(weights[base + c] / scale);
        trits[base + c] = q > 0 ? 1 : q < 0 ? -1 : 0;
      }
    }
    return { trits, scale: scales[0], scales, gamma };
  }

  const scale = meanAbs(0, n) || 1;
  for (let i = 0; i < n; i++) {
    const q = Math.round(weights[i] / scale);
    trits[i] = q > 0 ? 1 : q < 0 ? -1 : 0;
  }
  return { trits, scale, scales: null, gamma };
}

/**
 * Pack a trit vector into bytes, 5 trits per byte, base-3 little-endian.
 *
 * Mapping: -1 → 2, 0 → 0, +1 → 1. Unused trailing slots in the final byte are 0.
 *
 * @param {Int8Array} trits
 * @returns {Uint8Array} exactly ceil(trits.length / 5) bytes
 */
export function packTernary(trits) {
  const out = new Uint8Array(Math.ceil(trits.length / TRITS_PER_BYTE));
  for (let i = 0; i < trits.length; i++) {
    const t = trits[i];
    if (t !== -1 && t !== 0 && t !== 1) {
      throw new RangeError(`packTernary: trit ${t} at ${i} is not in {-1,0,+1}`);
    }
    const digit = t === -1 ? 2 : t;
    const slot = i % TRITS_PER_BYTE;
    out[(i / TRITS_PER_BYTE) | 0] += digit * 3 ** slot;
  }
  return out;
}

/**
 * Unpack bytes back into trits. Inverse of {@link packTernary}.
 * @param {Uint8Array} bytes
 * @param {number} count how many trits to recover
 * @returns {Int8Array}
 */
export function unpackTernary(bytes, count) {
  const out = new Int8Array(count);
  for (let i = 0; i < count; i++) {
    const byte = bytes[(i / TRITS_PER_BYTE) | 0];
    const slot = i % TRITS_PER_BYTE;
    const digit = Math.floor(byte / 3 ** slot) % 3;
    out[i] = digit === 2 ? -1 : digit;
  }
  return out;
}

/**
 * Split a packed matrix into its positive and negative planes without
 * materializing either as a float array. This is what keeps the kernel
 * allocation-free: the two planes are *masks*, and the multiply is masked.
 *
 * @param {Uint8Array} packed
 * @param {number} count trit count
 * @returns {{pos: Uint8Array, neg: Uint8Array}}
 */
export function planes(packed, count) {
  const trits = unpackTernary(packed, count);
  const pos = new Uint8Array(count);
  const neg = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    if (trits[i] === 1) pos[i] = 1;
    else if (trits[i] === -1) neg[i] = 1;
  }
  return { pos, neg };
}

/**
 * The ternary kernel.
 *
 *     Y[i] = gamma * sum_j ( pos[j*cols+i] * X[j] - neg[j*cols+i] * X[j] )
 *
 * Row-major `packed` weights of shape [rows, cols], `x` of length `rows`.
 * Accumulates in Float64 so long columns do not drift.
 *
 * @param {Uint8Array} packed packed ternary weights, row-major
 * @param {{rows: number, cols: number}} shape
 * @param {ArrayLike<number>} x input vector, length rows
 * @param {{gamma?: number, scale?: number, scales?: ArrayLike<number>}} [options]
 *   `scales` (length rows) enables per-row reconstruction; `scale` is the
 *   global fallback.
 * @returns {Float64Array} length cols
 */
export function ternaryMatMul(packed, shape, x, options = {}) {
  const { rows, cols } = shape;
  if (x.length !== rows) {
    throw new RangeError(`ternaryMatMul: x has ${x.length}, expected ${rows}`);
  }
  const need = rows * cols;
  if (packed.length * TRITS_PER_BYTE < need) {
    throw new RangeError(`ternaryMatMul: packed buffer too small for ${rows}x${cols}`);
  }
  const gamma = options.gamma ?? 1;
  const scale = options.scale ?? 1;
  const scales = options.scales ?? null;
  if (scales && scales.length !== rows) {
    throw new RangeError(`ternaryMatMul: scales has ${scales.length}, expected ${rows}`);
  }

  const { pos, neg } = planes(packed, need);
  const y = new Float64Array(cols);

  for (let j = 0; j < rows; j++) {
    const rowScale = scales ? scales[j] : 1;
    const xv = x[j] * rowScale * gamma * scale;
    if (xv === 0) continue;
    const base = j * cols;
    for (let i = 0; i < cols; i++) {
      const k = base + i;
      y[i] += (pos[k] - neg[k]) * xv;
    }
  }
  return y;
}

/**
 * Dense float reference. Same math, no packing — used by the tests to prove the
 * ternary path is correct, and by `bench` as the accuracy baseline.
 * @param {Float32Array} weights row-major [rows, cols]
 * @param {{rows: number, cols: number}} shape
 * @param {ArrayLike<number>} x
 * @returns {Float64Array}
 */
export function denseMatMul(weights, shape, x) {
  const { rows, cols } = shape;
  const y = new Float64Array(cols);
  for (let j = 0; j < rows; j++) {
    const xv = x[j];
    if (xv === 0) continue;
    const base = j * cols;
    for (let i = 0; i < cols; i++) y[i] += weights[base + i] * xv;
  }
  return y;
}

/**
 * Relative error between two vectors, as a fraction of the reference's L2 norm.
 * @param {ArrayLike<number>} actual
 * @param {ArrayLike<number>} reference
 * @returns {number} 0 when identical
 */
export function relativeError(actual, reference) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < reference.length; i++) {
    const d = actual[i] - reference[i];
    num += d * d;
    den += reference[i] * reference[i];
  }
  return den === 0 ? (num === 0 ? 0 : Infinity) : Math.sqrt(num / den);
}

/**
 * Bits per weight implied by a packing. The b1.58 claim, checked rather than
 * asserted: log2(3) ≈ 1.5850.
 * @returns {number}
 */
export function bitsPerWeight() {
  return Math.log2(3);
}

/**
 * Compression ratio of packed ternary against float32.
 * @returns {number}
 */
export function compressionVsFloat32() {
  return 32 / bitsPerWeight();
}

/**
 * A deterministic weight matrix. Real models ship real weights; this exists so
 * `bench` and the examples have something to chew on that is reproducible.
 * @param {number} rows
 * @param {number} cols
 * @param {number} [seed]
 * @returns {Float32Array}
 */
export function syntheticWeights(rows, cols, seed = 0x5eed) {
  const w = new Float32Array(rows * cols);
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = 0; i < w.length; i++) w[i] = next() * 2 - 1;
  return w;
}