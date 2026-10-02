import { fold, sampling } from '../src/gaia.js';
import { quantizeTernary, packTernary, ternaryMatMul, denseMatMul, relativeError } from '../src/mlx-quant.js';

// 1. The field, folded over a question. Same fold, every machine, forever.
const question = 'what is the smallest thing that can be said truthfully?';
const field = fold(question);
console.log('field');
console.log(`  dominant  ${field.dominant} (${field.inhabitant})`);
console.log(`  coherence ${field.coherence.toFixed(4)}`);
console.log('  sampling ', sampling(field));

// 2. The substrate: quantize a matrix, pack it, run the kernel.
const rows = 64;
const cols = 64;
const weights = new Float32Array(rows * cols);
for (let i = 0; i < weights.length; i++) weights[i] = [-1, 0, 1][i % 3] + Math.sin(i) * 0.05;

const { trits, scales } = quantizeTernary(weights, { rows, cols });
const packed = packTernary(trits);
const x = new Float32Array(rows).fill(1 / Math.sqrt(rows));

const reference = denseMatMul(weights, { rows, cols }, x);
const ternary = ternaryMatMul(packed, { rows, cols }, x, { scales });

console.log('\nsubstrate');
console.log(`  packed     ${packed.length} bytes for ${weights.length} weights`);
console.log(`  error      ${relativeError(ternary, reference).toFixed(4)} (ternary-shaped weights)`);
