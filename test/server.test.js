import test from 'node:test';
import assert from 'node:assert/strict';

import { start, createServer, runBench } from '../src/server.js';

/** Boot on an ephemeral port with the egress guard left off (it is tested elsewhere). */
async function boot() {
  const { server, url } = await start({ port: 0, installGuard: false });
  return { server, url };
}

test('the UI is served from disk with no remote references', async () => {
  const { server, url } = await boot();
  try {
    const page = await fetch(`${url}/`);
    assert.equal(page.status, 200);
    const html = await page.text();

    // The whole point: nothing on this page points off-machine.
    assert.doesNotMatch(html, /https?:\/\/(?!127\.0\.0\.1|localhost)/);
    assert.match(html, /ALEXIAI/);

    const csp = page.headers.get('content-security-policy');
    assert.match(csp, /default-src 'self'/);
    assert.match(csp, /connect-src 'self'/);
  } finally {
    server.close();
  }
});

test('the field endpoint folds deterministically', async () => {
  const { server, url } = await boot();
  try {
    const a = await (await fetch(`${url}/api/gaia?prompt=hello`)).json();
    const b = await (await fetch(`${url}/api/gaia?prompt=hello`)).json();
    assert.equal(a.seed, b.seed);
    assert.ok(a.sampling.temperature > 0);
    assert.ok(a.readout.includes('gravity'));
  } finally {
    server.close();
  }
});

test('health reports honestly when no local model is running', async () => {
  const { server, url } = await boot();
  try {
    const health = await (await fetch(`${url}/api/health`)).json();
    assert.equal(health.ok, false);
    assert.equal(health.sovereignty.policy, 'inference never leaves this machine');
  } finally {
    server.close();
  }
});

test('models degrade to a catalog when the endpoint is down', async () => {
  const { server, url } = await boot();
  try {
    const res = await fetch(`${url}/api/models`);
    const body = await res.json();
    assert.equal(res.status, 503);
    assert.equal(body.catalog[0].id, 'local-model');
  } finally {
    server.close();
  }
});

test('bench is reported as a real number', async () => {
  const { server, url } = await boot();
  try {
    const bench = await (await fetch(`${url}/api/bench?dim=64`)).json();
    assert.equal(bench.dim, 64);
    assert.ok(bench.bytesPacked < bench.bytesFloat32);
    assert.ok(bench.compression > 19);
  } finally {
    server.close();
  }
});

test('path traversal out of the web root is refused', async () => {
  const { server, url } = await boot();
  try {
    const res = await fetch(`${url}/../../package.json`, { redirect: 'manual' });
    assert.ok(res.status === 403 || res.status === 404, `got ${res.status}`);
  } finally {
    server.close();
  }
});

test('unknown API routes 404 rather than hanging', async () => {
  const { server, url } = await boot();
  try {
    assert.equal((await fetch(`${url}/api/nope`)).status, 404);
  } finally {
    server.close();
  }
});

test('chat streams a GAIA frame even when the model is unreachable', async () => {
  const { server, url } = await boot();
  try {
    const res = await fetch(`${url}/api/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: 'hello' }),
    });
    assert.equal(res.status, 200);
    const text = await res.text();
    assert.match(text, /^event: gaia/m);
    assert.match(text, /"dominant"/);
  } finally {
    server.close();
  }
});

test('createServer is reusable without listening', () => {
  const server = createServer({ installGuard: false });
  assert.equal(server.listening, false);
  server.close();
});

test('runBench is callable without a server', () => {
  const bench = runBench(32);
  assert.equal(bench.dim, 32);
  assert.ok(Number.isFinite(bench.relativeError));
});