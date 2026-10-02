import test from 'node:test';
import assert from 'node:assert/strict';

import {
  OsarousAdapter,
  DEFAULT_BASE_URL,
  DONE,
  normalizeBase,
  buildBody,
  iterateSSE,
} from '../src/osarous.js';
import { SovereigntyViolation } from '../src/sovereign.js';

// ── a fake local model server ──────────────────────────────────────────────

/**
 * Build a fetch stand-in that speaks just enough of the OpenAI surface.
 * @param {{models?: unknown[], chunks?: string[], status?: number}} [opts]
 */
function fakeFetch(opts = {}) {
  const chunks = opts.chunks ?? ['hello', ' there'];
  const requested = [];
  const fetchImpl = async (url, init) => {
    requested.push({ url: String(url), init });
    if (String(url).endsWith('/v1/models')) {
      return json({ data: opts.models ?? [{ id: 'test-model', context_window: 4096 }] });
    }
    if (String(url).endsWith('/health')) {
      return json({ ok: true });
    }
    if (String(url).includes('/v1/chat/completions')) {
      const body = JSON.parse(init.body);
      if (!body.stream) return json({ model: body.model, choices: [{ message: { content: 'hello there' }, finish_reason: 'stop' }], usage: { total_tokens: 7 } });
      const sse = [
        ...chunks.map((c) => `data: ${JSON.stringify({ choices: [{ delta: { content: c } }] })}\n\n`),
        `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { total_tokens: 7 } })}\n\n`,
        `data: ${DONE}\n\n`,
      ].join('');
      return sseResponse(sse);
    }
    return json({ error: 'not found' }, 404);
  };
  return { fetchImpl, requested };
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status, headers: { 'content-type': 'application/json' },
  });
}

function sseResponse(text) {
  return new Response(text, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });
}

// ── tests ──────────────────────────────────────────────────────────────────

test('the default endpoint is loopback', () => {
  assert.match(DEFAULT_BASE_URL, /^http:\/\/127\.0\.0\.1:/);
});

test('a remote endpoint cannot even be constructed', () => {
  assert.throws(
    () => new OsarousAdapter({ baseURL: 'https://api.openai.com/v1' }),
    SovereigntyViolation
  );
  assert.throws(
    () => new OsarousAdapter({ baseURL: 'http://192.168.1.10:1337' }),
    SovereigntyViolation
  );
});

test('base URLs normalize', () => {
  assert.equal(normalizeBase('http://127.0.0.1:1337/'), 'http://127.0.0.1:1337');
  assert.equal(normalizeBase('http://localhost:9000/api/'), 'http://localhost:9000/api');
});

test('catalog discovery reads the OpenAI shape', async () => {
  const { fetchImpl } = fakeFetch();
  const adapter = new OsarousAdapter({ fetchImpl });
  const connection = await adapter.refresh();
  assert.equal(connection.catalog[0].id, 'test-model');
  assert.equal(connection.catalog[0].contextWindow, 4096);
});

test('an empty catalog falls back rather than returning nothing', async () => {
  const { fetchImpl } = fakeFetch({ models: [] });
  const adapter = new OsarousAdapter({ fetchImpl });
  const connection = await adapter.refresh();
  assert.equal(connection.catalog.length, 1);
  assert.equal(connection.catalog[0].id, 'local-model');
});

test('a down server is a fact, not an exception', async () => {
  const adapter = new OsarousAdapter({
    fetchImpl: async () => { throw new Error('ECONNREFUSED'); },
  });
  const health = await adapter.health();
  assert.equal(health.up, false);
  assert.match(health.detail, /ECONNREFUSED/);
});

test('non-streaming completion returns text and usage', async () => {
  const { fetchImpl } = fakeFetch();
  const adapter = new OsarousAdapter({ fetchImpl });
  const result = await adapter.complete({ messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(result.text, 'hello there');
  assert.equal(result.usage.total_tokens, 7);
  assert.equal(result.finishReason, 'stop');
});

test('streaming yields deltas then a terminal frame', async () => {
  const { fetchImpl } = fakeFetch({ chunks: ['a', 'b', 'c'] });
  const adapter = new OsarousAdapter({ fetchImpl });
  const frames = [];
  for await (const frame of adapter.stream({ messages: [{ role: 'user', content: 'hi' }] })) {
    frames.push(frame);
  }
  assert.deepEqual(frames.slice(0, 3), [{ delta: 'a' }, { delta: 'b' }, { delta: 'c' }]);
  assert.equal(frames.at(-1).done, true);
  assert.equal(frames.at(-1).usage.total_tokens, 7);
});

test('the request body is OpenAI-compatible and carries GAIA sampling', async () => {
  const { fetchImpl, requested } = fakeFetch();
  const adapter = new OsarousAdapter({ fetchImpl });
  await adapter.complete({
    messages: [{ role: 'user', content: 'hi' }],
    model: 'test-model',
    temperature: 0.42,
    top_p: 0.9,
    top_k: 40,
    seed: 7,
  });
  const sent = JSON.parse(requested.at(-1).init.body);
  assert.equal(sent.model, 'test-model');
  assert.equal(sent.stream, false);
  assert.equal(sent.temperature, 0.42);
  assert.equal(sent.top_k, 40);
  assert.equal(sent.seed, 7);
});

test('tool messages keep their call id', () => {
  const body = buildBody({
    messages: [{ role: 'tool', content: 'ok', tool_call_id: 'call_1' }],
  }, false);
  assert.equal(body.messages[0].tool_call_id, 'call_1');
});

test('SSE parsing survives split chunks and keepalives', async () => {
  const stream = new Response(': keepalive\n\ndata: {"a":1}\n\n: ping\n\ndata: {"b":2}\n\n').body;
  const events = [];
  for await (const event of iterateSSE(stream)) events.push(event.data);
  assert.deepEqual(events, ['{"a":1}', '{"b":2}']);
});

test('SSE parsing joins multi-line data', async () => {
  const stream = new Response('data: line1\ndata: line2\n\n').body;
  const events = [];
  for await (const event of iterateSSE(stream)) events.push(event.data);
  assert.deepEqual(events, ['line1\nline2']);
});

test('a server error surfaces with its status', async () => {
  const adapter = new OsarousAdapter({
    fetchImpl: async () => json({ error: 'nope' }, 500),
  });
  await assert.rejects(
    () => adapter.complete({ messages: [{ role: 'user', content: 'hi' }] }),
    /HTTP 500/
  );
});