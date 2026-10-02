#!/usr/bin/env node
/**
 * alexiai — the command line.
 *
 * @module alexiai/cli
 */

import { OsarousAdapter, DEFAULT_BASE_URL } from './osarous.js';
import { start, runBench } from './server.js';
import { fold, sampling, readout } from './gaia.js';
import { attestation, SovereigntyViolation } from './sovereign.js';

const USAGE = `alexiai <3 — Omni edition

  alexiai serve [--port 8787] [--endpoint URL]   start the offline app
  alexiai doctor [--endpoint URL]                field readout + sovereignty check
  alexiai models [--endpoint URL]                list local models
  alexiai chat "<prompt>" [--endpoint URL]       one turn, streamed to stdout
  alexiai gaia ["<text>"]                        fold the field over some text
  alexiai bench [--dim 256]                      ternary vs dense

  --endpoint must be loopback. That is not a bug, it is the product.
`;

/** @type {string[]} */
const args = process.argv.slice(2);
const command = args[0] ?? 'serve';

try {
  await main(command, args.slice(1));
} catch (error) {
  if (error instanceof SovereigntyViolation) {
    console.error(`\n  ${error.message}\n`);
    console.error('  ALEXIAI does not talk to third-party model providers. Point --endpoint');
    console.error('  at a local MLX server, or run one first.\n');
    process.exit(2);
  }
  console.error(`\n  ${/** @type {Error} */ (error).message}\n`);
  if (process.env.ALEXIAI_DEBUG) console.error(error);
  process.exit(1);
}

/**
 * @param {string} command
 * @param {string[]} rest
 * @returns {Promise<void>}
 */
async function main(command, rest) {
  const flag = (name, fallback = undefined) => {
    const i = rest.indexOf(`--${name}`);
    return i === -1 ? fallback : rest[i + 1];
  };

  /** @type {OsarousAdapter} */
  let adapter;
  try {
    adapter = new OsarousAdapter({ baseURL: flag('endpoint', DEFAULT_BASE_URL) });
  } catch (error) {
    throw error;
  }

  switch (command) {
    case 'serve': {
      const port = Number(flag('port', '8787'));
      const { url } = await start({ port, adapter });
      const health = await adapter.health();
      console.log(`
  ALEXIAI <3 · Omni edition
  ─────────────────────────
  app       ${url}
  endpoint  ${adapter.baseURL}  ${health.up ? '· up' : '· not running (start your local model server)'}
  egress    loopback only
  offline   yes — no CDN, no remote asset, no third-party model

  open ${url}
`);
      break;
    }

    case 'doctor': {
      const health = await adapter.health();
      console.log('\n' + readout());
      console.log('\n  sovereignty');
      const a = attestation();
      console.log(`    policy     ${a.policy}`);
      console.log(`    guard      ${a.installed ? 'installed' : 'not installed'}`);
      console.log(`    loopback   ${a.loopbackHosts.join(', ')}`);
      console.log('\n  local model');
      console.log(`    endpoint   ${health.baseURL}`);
      console.log(`    status     ${health.up ? `up (${health.latencyMs}ms)` : 'down'} — ${health.detail}`);
      try {
        for (const m of await adapter.models()) {
          console.log(`    model      ${m.id}${m.contextWindow ? ` · ${m.contextWindow} ctx` : ''}`);
        }
      } catch {
        console.log('    model      (catalog unavailable while the endpoint is down)');
      }
      const bench = runBench(256);
      console.log('\n  gaia-mlx-quant');
      console.log(`    ternary    ${bench.bitsPerWeight.toFixed(4)} bits/weight · ${bench.compression}× vs float32`);
      console.log(`    error      ${(bench.relativeError * 100).toFixed(2)}% vs dense @ ${bench.dim}²\n`);
      break;
    }

    case 'models': {
      const connection = await adapter.refresh();
      console.log(`\n  ${connection.baseURL}\n`);
      for (const m of connection.catalog) {
        console.log(`  ${m.id.padEnd(28)} ${m.name ?? ''} ${m.contextWindow ? `${m.contextWindow} ctx` : ''}`);
      }
      console.log('');
      break;
    }

    case 'chat': {
      const prompt = rest.filter((a) => !a.startsWith('--') && rest[rest.indexOf(a) - 1] !== '--endpoint')
        .join(' ');
      if (!prompt) throw new Error('chat needs a prompt: alexiai chat "hello"');
      const field = fold(prompt);
      const params = sampling(field);
      console.error(`\n  field: dominant ${field.dominant} · coherence ${field.coherence.toFixed(3)} · T=${params.temperature}\n`);
      process.stdout.write('  ');
      for await (const chunk of adapter.stream({
        messages: [
          { role: 'system', content: 'You are ALEXIAI, running fully offline. Be precise.' },
          { role: 'user', content: prompt },
        ],
        ...params,
      })) {
        if ('delta' in chunk) process.stdout.write(chunk.delta);
      }
      process.stdout.write('\n\n');
      break;
    }

    case 'gaia': {
      const text = rest.join(' ');
      const f = fold(text);
      console.log('\n' + readout(f.seed));
      console.log(`  sampling: ${JSON.stringify(sampling(f))}\n`);
      break;
    }

    case 'bench': {
      const dim = Number(flag('dim', '256'));
      console.log(`\n  ${JSON.stringify(runBench(dim), null, 2)}\n`);
      break;
    }

    case 'help':
    case '--help':
    case '-h':
      console.log(USAGE);
      break;

    default:
      console.log(USAGE);
      process.exit(1);
  }
}