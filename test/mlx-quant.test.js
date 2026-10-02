import test from 'node:test';
import assert from 'node:assert/strict';

import {
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
} from '../src/mlx-quant.js';

test('b1.58 really is log2(3)', () => {
  assert.ok(Math.abs(bitsPerWeight() - 1.58496) < 1e-4);
  assert.ok(compressionVsFloat32() > 20);
});

test('quantization produces only legal trits', () => {
  const w = syntheticWeights(64, 64);
  const { trits } = quantizeTernary(w);
  for (const t of trits) assert.ok(t === -1 || t === 0 || t === 1, `illegal trit ${t}`);
});

test('packing round-trips exactly', () => {
  const trits = new Int8Array(1000);
  for (let i = 0; i < trits.length; i++) trits[i] = [-1, 0, 1][i % 3];
  const packed = packTernary(trits);
  assert.equal(packed.length, Math.ceil(1000 / TRITS_PER_BYTE));
  assert.deepEqual([...unpackTernary(packed, 1000)], [...trits]);
});

test('packing round-trips at every length modulo 5', () => {
  for (let n = 0; n <= 32; n++) {
    const trits = new Int8Array(n);
    for (let i = 0; i < n; i++) trits[i] = i % 2 ? -1 : 1;
    assert.deepEqual([...unpackTernary(packTernary(trits), n)], [...trits], `n=${n}`);
  }
});

test('packing refuses non-trits', () => {
  const trits = new Int8Array([1, 2, 0]);
  assert.throws(() => packTernary(trits), RangeError);
});

test('planes split the trits into disjoint masks', () => {
  const trits = new Int8Array([1, -1, 0, 1, -1]);
  const { pos, neg } = planes(packTernary(trits), 5);
  assert.deepEqual([...pos], [1, 0, 0, 1, 0]);
  assert.deepEqual([...neg], [0, 1, 0, 0, 1]);
});

test('the kernel matches a dense reference on exact trits', () => {
  // Hand-built: no quantization error to muddy the comparison.
  const rows = 3, cols = 4;
  const trits = new Int8Array([
    1, -1, 0, 1,
    -1, 1, 1, 0,
    0, 0, -1, 1,
  ]);
  const dense = Float32Array.from(trits);
  const x = [2, -1, 3];

  const ternary = ternaryMatMul(packTernary(trits), { rows, cols }, x, { gamma: 1 });
  const reference = denseMatMul(dense, { rows, cols }, x);

  assert.deepEqual([...ternary], [...reference]);
});

test('reconstruction is bounded even on adversarial random weights', () => {
  // Random projections are the worst case for b1.58: real models are trained
  // (straight-through estimator) until their weights are ternary-shaped, which
  // is what the next two tests pin. This test only guarantees the error stays
  // bounded rather than exploding — the honest reading of "lossy by design".
  const rows = 64, cols = 64;
  const w = syntheticWeights(rows, cols);
  const x = syntheticWeights(rows, 1, 99);
  const { trits, scales } = quantizeTernary(w, { rows, cols });
  const ternary = ternaryMatMul(packTernary(trits), { rows, cols }, x, { scales });
  const reference = denseMatMul(w, { rows, cols }, x);
  assert.ok(relativeError(ternary, reference) < 0.6, 'error must stay bounded');
});

test('on ternary-shaped weights the projection is tight', () => {
  // What an STE-trained b1.58 weight matrix actually looks like: values hug
  // {-1, 0, +1}. Here the projection is close, which is the whole premise.
  const rows = 64, cols = 64;
  const w = new Float32Array(rows * cols);
  for (let i = 0; i < w.length; i++) w[i] = [-1, 0, 1][i % 3] + Math.sin(i * 0.7) * 0.05;
  const x = syntheticWeights(rows, 1, 99);

  const { trits, scales } = quantizeTernary(w, { rows, cols });
  const error = relativeError(
    ternaryMatMul(packTernary(trits), { rows, cols }, x, { scales }),
    denseMatMul(w, { rows, cols }, x)
  );
  assert.ok(error < 0.34, `projection should be tight, got ${error}`);
});

test('per-row scaling beats a single global scale on rows of distinct magnitude', () => {
  const rows = 64, cols = 64;
  const w = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const alpha = 0.3 + r * 0.01; // every row lives at a different scale
    for (let c = 0; c < cols; c++) w[r * cols + c] = alpha * [-1, 0, 1][(r + c) % 3];
  }
  const x = syntheticWeights(rows, 1, 99);
  const reference = denseMatMul(w, { rows, cols }, x);

  const perRow = quantizeTernary(w, { rows, cols });
  const flat = quantizeTernary(w);

  const errPerRow = relativeError(
    ternaryMatMul(packTernary(perRow.trits), { rows, cols }, x, { scales: perRow.scales }),
    reference
  );
  const errFlat = relativeError(
    ternaryMatMul(packTernary(flat.trits), { rows, cols }, x, { scale: flat.scale }),
    reference
  );
  assert.ok(errPerRow < errFlat, `per-row ${errPerRow} should beat flat ${errFlat}`);
});

test('quantization validates its shape claim', () => {
  assert.throws(() => quantizeTernary([1, 2, 3], { rows: 2, cols: 2 }), RangeError);
  const { scales } = quantizeTernary([1, 2, 3, 4], { rows: 2, cols: 2 });
  assert.equal(scales.length, 2);
});

test('shape mismatches fail loudly', () => {
  const trits = new Int8Array(6);
  const packed = packTernary(trits);
  assert.throws(() => ternaryMatMul(packed, { rows: 2, cols: 3 }, [1]), /expected 2/);
  assert.throws(() => ternaryMatMul(packed, { rows: 4, cols: 4 }, [1, 2, 3, 4]), /too small/);
});

test('an empty matrix is an error, not a silent zero', () => {
  assert.throws(() => quantizeTernary([]), RangeError);
});

test('relative error is zero for identical vectors', () => {
  assert.equal(relativeError([1, 2, 3], [1, 2, 3]), 0);
});

test('synthetic weights are deterministic', () => {
  assert.deepEqual([...syntheticWeights(8, 8, 3)], [...syntheticWeights(8, 8, 3)]);
});