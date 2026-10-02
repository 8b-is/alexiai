import test from 'node:test';
import assert from 'node:assert/strict';

import {
  fold, sampling, readout, rng, seedFrom, LAYERS, FIELD,
} from '../src/gaia.js';

test('the field is frozen', () => {
  assert.throws(() => { FIELD.gravity.value = 9; }, TypeError);
});

test('four layers, and they are the four the cosmology names', () => {
  assert.deepEqual([...LAYERS], ['gravity', 'entropy', 'weather', 'resonance']);
});

test('fold is pure: same input, same output, forever', () => {
  const a = fold('hello field');
  const b = fold('hello field');
  assert.deepEqual(a, b);
});

test('different text lands on a different fold', () => {
  assert.notEqual(fold('a').seed, fold('b').seed);
});

test('an explicit seed overrides the derived one', () => {
  const a = fold('anything', 42);
  const b = fold('other things entirely', 42);
  assert.equal(a.seed, 42);
  assert.equal(b.seed, 42);
  // Weather is pure seed-rolled rng; entropy and coherence respond to the text.
  assert.equal(a.layers.weather, b.layers.weather);
});

test('layers stay in range', () => {
  const long = fold('x'.repeat(10000), 7);
  for (const layer of ['gravity', 'entropy', 'weather']) {
    assert.ok(long.layers[layer] >= 0 && long.layers[layer] <= 1, `${layer} out of range`);
  }
  // Resonance is the golden ratio, the one constant that exceeds one by design.
  assert.ok(long.layers.resonance > 1.6 && long.layers.resonance < 1.62);
});

test('coherence is a probability', () => {
  for (const seed of [0, 1, 2, 99, 123456]) {
    const { coherence } = fold('sample', seed);
    assert.ok(coherence >= 0 && coherence <= 1);
  }
});

test('longer text disperses more', () => {
  const short = fold('hi', 11).layers.entropy;
  const long = fold('hi '.repeat(2000), 11).layers.entropy;
  assert.ok(long > short, `${long} should exceed ${short}`);
});

test('sampling derives from the fold and stays sane', () => {
  const params = sampling(fold('question', 5));
  assert.ok(params.temperature >= 0.05 && params.temperature <= 1.2);
  assert.ok(params.top_p >= 0.5 && params.top_p <= 0.98);
  assert.ok(params.top_k >= 20 && params.top_k <= 80);
  assert.equal(params.seed, 5);
});

test('rng is deterministic and bounded', () => {
  const a = rng(1234);
  const b = rng(1234);
  for (let i = 0; i < 50; i++) {
    const v = a();
    assert.equal(v, b());
    assert.ok(v >= 0 && v < 1);
  }
});

test('seedFrom is stable', () => {
  assert.equal(seedFrom('alexiai'), seedFrom('alexiai'));
  assert.notEqual(seedFrom('alexiai'), seedFrom('alexia'));
});

test('readout names every layer', () => {
  const text = readout(3);
  for (const layer of LAYERS) assert.match(text, new RegExp(layer));
});