/**
 * GAIA — the world's constants.
 *
 * In the 8b-is cosmology, GAIA ≡ planets ≡ deities: the semantic layer of the
 * world-model, the field the constants live in. ALEXIAI treats GAIA as the
 * *configuration substrate* — the immutable field every request is evaluated
 * against before it is allowed to touch a model.
 *
 * This module is dependency-free and total: it never performs I/O, never
 * reaches the network, and every value is frozen at module load. That is the
 * point. The field is the one thing in the system that cannot be moved by
 * anything that runs after it.
 *
 * @module alexiai/gaia
 */

/** @typedef {'gravity'|'entropy'|'weather'|'resonance'} GaiaLayer */

/**
 * The four layers of the field. Order is load-bearing: gravity binds,
 * entropy releases, weather modulates, resonance folds.
 * @type {readonly GaiaLayer[]}
 */
export const LAYERS = Object.freeze(['gravity', 'entropy', 'weather', 'resonance']);

/**
 * The field itself. Frozen; mutation throws in strict mode (all ESM is strict).
 * @type {Readonly<Record<GaiaLayer, {readonly value: number, readonly unit: string, readonly note: string}>>}
 */
export const FIELD = Object.freeze({
  /** What holds the world together. 1.0 = nominal cohesion. */
  gravity: Object.freeze({ value: 1.0, unit: 'g', note: 'cohesion of the field' }),

  /** What the world spends. 0.0 = perfectly ordered, 1.0 = total dispersal. */
  entropy: Object.freeze({ value: 0.62, unit: 's', note: 'dispersal budget per turn' }),

  /** What modulates between turns. Deterministic given the seed. */
  weather: Object.freeze({ value: 0.5, unit: 'w', note: 'turn-to-turn modulation' }),

  /** What folds the layers into one another. The golden ratio, because of course. */
  resonance: Object.freeze({ value: (1 + 5 ** 0.5) / 2, unit: 'φ', note: 'folding constant' }),
});

/**
 * Planet personifications. GAIA ≡ planets ≡ deities: each layer is inhabited.
 * Used by the UI to render the field, and by `fold` to name what it produced.
 * @type {Readonly<Record<GaiaLayer, {planet: string, deity: string}>>}
 */
export const INHABITANTS = Object.freeze({
  gravity: Object.freeze({ planet: 'Menchia', deity: 'Ananke' }),
  entropy: Object.freeze({ planet: 'Sonder', deity: 'Chaos' }),
  weather: Object.freeze({ planet: 'Nimbus', deity: 'Zephyrus' }),
  resonance: Object.freeze({ planet: 'Aurea', deity: 'Aphrodite' }),
});

/**
 * A tiny deterministic PRNG (mulberry32). GAIA must be reproducible: the same
 * seed always yields the same fold, which is what makes a transcript replayable.
 *
 * @param {number} seed 32-bit unsigned seed
 * @returns {() => number} generator yielding floats in [0, 1)
 */
export function rng(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Hash an arbitrary string into the 32-bit seed space (FNV-1a).
 * @param {string} text
 * @returns {number}
 */
export function seedFrom(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * Fold the field for a given prompt at a given seed.
 *
 * The fold is the one place the four layers interact. It is pure: same
 * (prompt, seed) in, same fold out, always. Nothing here reads the clock, the
 * filesystem, or the network.
 *
 * @param {string} prompt the text being folded
 * @param {number} [seed] explicit seed; derived from the prompt when omitted
 * @returns {{seed: number, layers: Record<GaiaLayer, number>, dominant: GaiaLayer, coherence: number, inhabitant: string}}
 */
export function fold(prompt, seed) {
  const s = seed === undefined ? seedFrom(prompt) : seed >>> 0;
  const next = rng(s);

  /** @type {Record<GaiaLayer, number>} */
  const layers = {
    gravity: FIELD.gravity.value,
    entropy: FIELD.entropy.value,
    weather: FIELD.weather.value,
    resonance: FIELD.resonance.value,
  };

  // Prompt length pulls on entropy (longer = more dispersal), the seed rolls weather.
  layers.entropy = clamp(layers.entropy + Math.min(prompt.length / 4096, 0.35) - 0.15, 0, 1);
  layers.weather = clamp(layers.weather + (next() - 0.5) * 0.4, 0, 1);

  // Dominance is whichever layer the fold pushed furthest from rest.
  /** @type {GaiaLayer[]} */
  const candidates = ['gravity', 'entropy', 'weather', 'resonance'];
  let dominant = candidates[0];
  let best = -Infinity;
  for (const layer of candidates) {
    const distance = Math.abs(layers[layer] - FIELD[layer].value);
    if (distance > best) {
      best = distance;
      dominant = layer;
    }
  }

  // Coherence: how tightly the four layers agree after folding. High coherence
  // means the field held; low coherence means the world moved.
  const mean = candidates.reduce((acc, l) => acc + layers[l], 0) / candidates.length;
  const spread = candidates.reduce((acc, l) => acc + Math.abs(layers[l] - mean), 0) / candidates.length;
  const coherence = clamp(1 - spread, 0, 1);

  return { seed: s, layers, dominant, coherence, inhabitant: INHABITANTS[dominant].deity };
}

/**
 * Clamp a number into [min, max].
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export function clamp(value, min, max) {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

/**
 * The sampling parameters GAIA derives from a fold. These are what actually get
 * sent to the local model — the field's opinion about how to answer.
 *
 * Higher entropy ⇒ higher temperature (more dispersal in the answer).
 * Higher coherence ⇒ tighter top-p (the world held, so stay near the centre).
 *
 * @param {ReturnType<typeof fold>} foldResult
 * @returns {{temperature: number, top_p: number, top_k: number, seed: number}}
 */
export function sampling(foldResult) {
  const { entropy, coherence } = foldResult.layers;
  return {
    temperature: Number(clamp(0.15 + entropy * 1.1, 0.05, 1.2).toFixed(4)),
    top_p: Number(clamp(0.5 + coherence * 0.48, 0.5, 0.98).toFixed(4)),
    top_k: Math.round(clamp(20 + (1 - coherence) * 60, 20, 80)),
    seed: foldResult.seed,
  };
}

/**
 * A printable readout of the field — used by `alexiai doctor` and the UI header.
 * @param {number} [seed]
 * @returns {string}
 */
export function readout(seed) {
  const f = fold('', seed);
  const rows = LAYERS.map((l) => {
    const value = f.layers[l].toFixed(4);
    const rest = FIELD[l].value.toFixed(4);
    const delta = (f.layers[l] - FIELD[l].value >= 0 ? '+' : '') +
      (f.layers[l] - FIELD[l].value).toFixed(4);
    return `  ${l.padEnd(10)} ${value}  (rest ${rest}, ${delta})  ${INHABITANTS[l].deity}`;
  });
  return [
    'GAIA field',
    `  seed ${f.seed} · dominant ${f.dominant} · coherence ${f.coherence.toFixed(4)}`,
    ...rows,
  ].join('\n');
}