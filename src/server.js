/**
 * The ALEXIAI server — one process, one port, no network.
 *
 * Serves the offline UI and proxies inference to whatever local model server
 * the user points it at. There is no outbound path from here: the UI is served
 * from disk, and the only upstream this process will talk to is loopback.
 *
 * @module alexiai/server
 */

import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

import { fold, sampling, readout } from './gaia.js';
import { OsarousAdapter, DEFAULT_BASE_URL, DONE } from './osarous.js';
import { installEgressGuard, attestation } from './sovereign.js';
import { relativeError, ternaryMatMul, denseMatMul, packTernary, quantizeTernary, syntheticWeights } from './mlx-quant.js';

const WEB_ROOT = fileURLToPath(new URL('../web/', import.meta.url));

/** Content types for the handful of assets the UI uses. No CDN, ever. */
const CONTENT_TYPES = Object.freeze({
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.webmanifest': 'application/manifest+json',
  '.ico': 'image/x-icon',
});

/**
 * Create the ALEXIAI HTTP server.
 *
 * @param {{adapter?: OsarousAdapter, installGuard?: boolean}} [options]
 * @returns {http.Server}
 */
export function createServer(options = {}) {
  const restore = options.installGuard === false ? () => {} : installEgressGuard();
  const adapter = options.adapter ?? new OsarousAdapter({ baseURL: DEFAULT_BASE_URL });

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
    try {
      if (url.pathname.startsWith('/api/')) {
        await handleApi(url, req, res, adapter);
      } else {
        await serveStatic(url.pathname, res);
      }
    } catch (error) {
      sendJson(res, 500, { error: /** @type {Error} */ (error).message });
    }
  });

  server.on('close', restore);
  // Expose the adapter so `serve` can print a useful banner.
  server.alexiai = { adapter };
  return server;
}

/**
 * API surface. Small on purpose: health, field, catalog, chat, bench.
 * @param {URL} url
 * @param {http.IncomingMessage} req
 * @param {http.ServerResponse} res
 * @param {OsarousAdapter} adapter
 */
async function handleApi(url, req, res, adapter) {
  const route = `${req.method} ${url.pathname}`;

  if (route === 'GET /api/health') {
    const health = await adapter.health();
    return sendJson(res, 200, {
      ok: health.up,
      endpoint: health.baseURL,
      latencyMs: health.latencyMs,
      detail: health.detail,
      sovereignty: attestation(),
      gaia: fold('', Number(url.searchParams.get('seed') ?? 0)).seed,
    });
  }

  if (route === 'GET /api/gaia') {
    const prompt = url.searchParams.get('prompt') ?? '';
    const seedParam = url.searchParams.get('seed');
    const result = fold(prompt, seedParam === null ? undefined : Number(seedParam));
    return sendJson(res, 200, {
      ...result,
      sampling: sampling(result),
      readout: readout(result.seed),
    });
  }

  if (route === 'GET /api/models') {
    try {
      const connection = await adapter.refresh();
      return sendJson(res, 200, connection);
    } catch (error) {
      return sendJson(res, 503, { error: /** @type {Error} */ (error).message, catalog: [adapter.fallbackModel()] });
    }
  }

  if (route === 'GET /api/bench') {
    return sendJson(res, 200, runBench(Number(url.searchParams.get('dim') ?? 256)));
  }

  if (route === 'POST /api/chat') {
    return streamChat(req, res, adapter);
  }

  return sendJson(res, 404, { error: `no route for ${route}` });
}

/**
 * Chat: GAIA folds the prompt into sampling params, then the local model runs.
 * The response is SSE so the UI can render tokens as they land.
 *
 * @param {http.IncomingMessage} req
 * @param {http.ServerResponse} res
 * @param {OsarousAdapter} adapter
 */
async function streamChat(req, res, adapter) {
  /** @type {{prompt?: string, history?: Array<{role: string, content: string}>, model?: string}} */
  const body = await readJson(req);
  const prompt = body.prompt ?? '';
  const history = Array.isArray(body.history) ? body.history : [];

  const field = fold(prompt);
  const params = sampling(field);

  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-store',
    connection: 'keep-alive',
  });
  writeEvent(res, 'gaia', { field, sampling: params });

  /** @type {Array<{role: string, content: string}>} */
  const messages = [
    { role: 'system', content: systemPrompt() },
    ...history,
    { role: 'user', content: prompt },
  ];

  try {
    for await (const chunk of adapter.stream({ messages, model: body.model, ...params })) {
      if ('delta' in chunk) writeEvent(res, 'delta', { text: chunk.delta });
      else writeEvent(res, 'done', { usage: chunk.usage, finishReason: chunk.finishReason });
    }
  } catch (error) {
    writeEvent(res, 'error', { message: /** @type {Error} */ (error).message });
  }
  res.write(`data: ${DONE}\n\n`);
  res.end();
}

/**
 * The field's own instructions to the model. Kept short and honest — it states
 * the substrate, not a persona, because the personality is the user's.
 * @returns {string}
 */
function systemPrompt() {
  return [
    'You are ALEXIAI, running fully offline on the GAIA-MLX-QUANT substrate.',
    'You have no network access and no third-party model providers.',
    'Be precise, say when you do not know, and never claim to have looked anything up.',
  ].join(' ');
}

/**
 * Serve a file from `web/`, refusing anything that escapes the root.
 * @param {string} pathname
 * @param {http.ServerResponse} res
 */
async function serveStatic(pathname, res) {
  const rel = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
  const safe = normalize(rel).replace(/^(\.\.[/\\])+/, '');
  const file = join(WEB_ROOT, safe);
  if (!file.startsWith(WEB_ROOT) && file !== WEB_ROOT) {
    return sendJson(res, 403, { error: 'path escapes web root' });
  }
  try {
    const body = await readFile(file);
    const type = CONTENT_TYPES[extname(file)] ?? 'application/octet-stream';
    // The service worker needs to be revalidated, everything else can be pinned.
    const cache = safe === 'sw.js' ? 'no-cache' : 'no-cache';
    res.writeHead(200, {
      'content-type': type,
      'cache-control': cache,
      'content-length': body.length,
      etag: `"${createHash('sha256').update(body).digest('hex').slice(0, 16)}"`,
      // Lock the page down: no remote anything.
      'content-security-policy':
        "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'",
    });
    res.end(body);
  } catch {
    sendJson(res, 404, { error: `not found: ${pathname}` });
  }
}

/**
 * The bench: ternary packed vs dense float, at a given dimension. Reports the
 * accuracy gap honestly rather than quoting someone else's GFLOPS number.
 * @param {number} dim
 */
export function runBench(dim = 256) {
  const rows = dim;
  const cols = dim;
  const weights = syntheticWeights(rows, cols);
  const x = syntheticWeights(rows, 1, 0xc0ffee);

  const { trits, scales } = quantizeTernary(weights, { rows, cols, gamma: 1 });
  const packed = packTernary(trits);

  const started = process.hrtime.bigint();
  const dense = denseMatMul(weights, { rows, cols }, x);
  const denseNs = Number(process.hrtime.bigint() - started);

  const started2 = process.hrtime.bigint();
  const ternary = ternaryMatMul(packed, { rows, cols }, x, { scales });
  const ternaryNs = Number(process.hrtime.bigint() - started2);

  return {
    dim,
    bitsPerWeight: Math.log2(3),
    bytesPacked: packed.length,
    bytesFloat32: weights.byteLength,
    compression: Number((weights.byteLength / packed.length).toFixed(2)),
    ternaryNs,
    denseNs,
    relativeError: Number(relativeError(ternary, dense).toFixed(6)),
  };
}

/**
 * @param {http.ServerResponse} res
 * @param {number} status
 * @param {unknown} payload
 */
function sendJson(res, status, payload) {
  const body = JSON.stringify(payload, null, 2);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
  });
  res.end(body);
}

/**
 * @param {http.ServerResponse} res
 * @param {string} event
 * @param {unknown} data
 */
function writeEvent(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

/**
 * @param {http.IncomingMessage} req
 * @returns {Promise<any>}
 */
function readJson(req) {
  return new Promise((resolve, reject) => {
    /** @type {Buffer[]} */
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8').trim();
      if (!raw) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch (error) {
        reject(new Error(`invalid JSON body: ${/** @type {Error} */ (error).message}`));
      }
    });
    req.on('error', reject);
  });
}

/**
 * Start the server.
 * @param {{port?: number, host?: string, adapter?: OsarousAdapter, installGuard?: boolean}} [options]
 * @returns {Promise<{server: http.Server, port: number, url: string}>}
 */
export function start(options = {}) {
  const server = createServer(options);
  const port = options.port ?? 8787;
  const host = options.host ?? '127.0.0.1';
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => {
      const address = server.address();
      const bound = typeof address === 'object' && address ? address.port : port;
      resolve({ server, port: bound, url: `http://${host}:${bound}` });
    });
  });
}