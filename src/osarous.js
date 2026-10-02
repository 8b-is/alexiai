/**
 * osarous — local model support.
 *
 * An adapter for any Apple Silicon MLX server that speaks the OpenAI-compatible
 * chat-completions surface on loopback. The shape mirrors the `llm-osarous`
 * package that ships with the DeepSeek harness: an OpenAI-compatible
 * `/v1/chat/completions` endpoint, Server-Sent Events for streaming, a
 * `/v1/models` catalog, and connection facts resolved *per request* rather than
 * frozen at load — so you can point ALEXIAI at a different local port, or a
 * different local model, without restarting anything.
 *
 * What it deliberately is not: a client for anyone else's API. There is no
 * base-URL constructor path that accepts a remote host. See `sovereign.js`.
 *
 * @module alexiai/osarous
 */

import { assertLocal, SovereigntyViolation } from './sovereign.js';

/** Default sidecar endpoint. Loopback by construction, never configurable past loopback. */
export const DEFAULT_BASE_URL = 'http://127.0.0.1:1337';

/** Advertised context window when a model does not declare its own. */
export const DEFAULT_CONTEXT_WINDOW = 262144;

/** How long to wait on a non-streaming response. */
export const DEFAULT_TIMEOUT_MS = 120000;

/**
 * @typedef {object} OsarousCatalogModel
 * @property {string} id
 * @property {string} [name]
 * @property {number} [contextWindow]
 * @property {number[]} [quantBits] quantization ladder the model ships in
 */

/**
 * @typedef {object} OsarousConnection
 * @property {string} baseURL loopback base URL
 * @property {OsarousCatalogModel[]} catalog
 * @property {number} fetchedAt epoch ms when the catalog was read
 */

/**
 * @typedef {object} ChatMessage
 * @property {'system'|'user'|'assistant'|'tool'} role
 * @property {string} content
 * @property {string} [tool_call_id]
 */

/**
 * @typedef {object} ChatRequest
 * @property {ChatMessage[]} messages
 * @property {string} [model]
 * @property {boolean} [stream]
 * @property {number} [temperature]
 * @property {number} [top_p]
 * @property {number} [top_k]
 * @property {number} [max_tokens]
 * @property {number} [seed]
 */

/**
 * @typedef {object} Usage
 * @property {number} [prompt_tokens]
 * @property {number} [completion_tokens]
 * @property {number} [total_tokens]
 */

/** Sentinel emitted when a stream terminates normally (mirrors the harness). */
export const DONE = '[DONE]';

/**
 * The adapter. One instance owns one endpoint; nothing is cached across calls
 * except the catalog, which `refresh()` controls explicitly.
 */
export class OsarousAdapter {
  /**
   * @param {{baseURL?: string, defaultContextWindow?: number, timeoutMs?: number, fetchImpl?: typeof fetch}} [options]
   */
  constructor(options = {}) {
    // Throws SovereigntyViolation for any non-loopback baseURL.
    this.baseURL = normalizeBase(options.baseURL ?? DEFAULT_BASE_URL);
    this.defaultContextWindow = options.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch;
    /** @type {OsarousConnection | null} */
    this.connection = null;
  }

  /**
   * Read the model's catalog. Resolves per request, so a restart of the local
   * server on the same port is picked up without touching this object.
   * @returns {Promise<OsarousConnection>}
   */
  async refresh() {
    const url = `${this.baseURL}/v1/models`;
    const response = await this.fetchImpl(url, { method: 'GET' });
    if (!response.ok) throw new Error(`osarous: /v1/models -> HTTP ${response.status}`);

    /** @type {{data?: unknown}} */
    const payload = await response.json();
    const rows = Array.isArray(payload?.data) ? payload.data : [];

    /** @type {OsarousCatalogModel[]} */
    const catalog = rows.map((row) => {
      const r = /** @type {Record<string, unknown>} */ (row);
      return {
        id: String(r.id ?? 'local-model'),
        name: typeof r.name === 'string' ? r.name : undefined,
        contextWindow: typeof r.context_window === 'number'
          ? r.context_window
          : this.defaultContextWindow,
        quantBits: Array.isArray(r.quant_bits) ? r.quant_bits.map(Number) : undefined,
      };
    });

    this.connection = {
      baseURL: this.baseURL,
      catalog: catalog.length > 0 ? catalog : [this.fallbackModel()],
      fetchedAt: Date.now(),
    };
    return this.connection;
  }

  /**
   * @returns {OsarousCatalogModel} the model advertised when discovery is unavailable
   */
  fallbackModel() {
    return { id: 'local-model', name: 'Local Model', contextWindow: this.defaultContextWindow };
  }

  /**
   * The catalog, refreshing if it has never been read.
   * @returns {Promise<OsarousCatalogModel[]>}
   */
  async models() {
    return (await this.refresh()).catalog;
  }

  /**
   * Is the local server up? Never throws — a down server is a fact, not an error.
   * @returns {Promise<{up: boolean, baseURL: string, latencyMs: number|null, detail: string}>}
   */
  async health() {
    const started = Date.now();
    try {
      const response = await this.fetchImpl(`${this.baseURL}/health`, { method: 'GET' });
      const latencyMs = Date.now() - started;
      return {
        up: response.ok,
        baseURL: this.baseURL,
        latencyMs,
        detail: response.ok ? 'healthy' : `HTTP ${response.status}`,
      };
    } catch (error) {
      return {
        up: false,
        baseURL: this.baseURL,
        latencyMs: null,
        detail: /** @type {Error} */ (error).message,
      };
    }
  }

  /**
   * Non-streaming completion. Returns the assistant text plus usage.
   *
   * @param {ChatRequest} request
   * @returns {Promise<{text: string, model: string, usage: Usage, finishReason: string|null}>}
   */
  async complete(request) {
    const body = buildBody(request, false);
    const response = await this.send(body);
    /** @type {{choices?: Array<{message?: {content?: string}, finish_reason?: string|null}>, usage?: Usage, model?: string}} */
    const payload = await response.json();
    const choice = payload?.choices?.[0];
    return {
      text: choice?.message?.content ?? '',
      model: payload?.model ?? body.model,
      usage: payload?.usage ?? {},
      finishReason: choice?.finish_reason ?? null,
    };
  }

  /**
   * Streaming completion. Yields content deltas as they arrive, then a final
   * usage frame. The generator is lazy: nothing is requested until you pull.
   *
   * @param {ChatRequest} request
   * @returns {AsyncGenerator<{delta: string}|{done: true, usage: Usage, finishReason: string|null}, void, void>}
   */
  async *stream(request) {
    const body = buildBody(request, true);
    const response = await this.send(body);
    if (!response.body) throw new Error('osarous: streaming response had no body');

    /** @type {Usage} */
    let usage = {};
    let finishReason = null;

    for await (const event of iterateSSE(response.body)) {
      if (event.data === DONE) break;
      /** @type {{choices?: Array<{delta?: {content?: string}, finish_reason?: string|null}>, usage?: Usage}} */
      let chunk;
      try {
        chunk = JSON.parse(event.data);
      } catch {
        continue; // a server that interleaves keepalives is not a failure
      }
      const delta = chunk?.choices?.[0]?.delta?.content;
      if (delta) yield { delta };
      if (chunk?.usage) usage = chunk.usage;
      const reason = chunk?.choices?.[0]?.finish_reason;
      if (reason) finishReason = reason;
    }
    yield { done: true, usage, finishReason };
  }

  /**
   * POST with the sovereignty check applied at the last possible moment.
   * @param {Record<string, unknown>} body
   * @returns {Promise<Response>}
   */
  async send(body) {
    const url = `${this.baseURL}/v1/chat/completions`;
    assertLocal(url); // belt: the constructor already checked, braces: re-check the exact URL
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetchImpl(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(`osarous: ${url} -> HTTP ${response.status} ${text.slice(0, 200)}`);
      }
      return response;
    } catch (error) {
      if (error instanceof SovereigntyViolation) throw error;
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }
}

/**
 * Normalize and validate a base URL. Strips a trailing slash; refuses anything
 * that is not loopback.
 * @param {string} baseURL
 * @returns {string}
 */
export function normalizeBase(baseURL) {
  const parsed = assertLocal(baseURL);
  const path = parsed.pathname.replace(/\/+$/, '');
  return `${parsed.origin}${path}`;
}

/**
 * Translate a request into an OpenAI-compatible body.
 * @param {ChatRequest} request
 * @param {boolean} stream
 * @returns {Record<string, unknown>}
 */
export function buildBody(request, stream) {
  /** @type {Record<string, unknown>} */
  const body = {
    model: request.model ?? 'local-model',
    messages: request.messages.map((m) => (m.tool_call_id
      ? { role: m.role, content: m.content, tool_call_id: m.tool_call_id }
      : { role: m.role, content: m.content })),
    stream,
  };
  if (request.temperature !== undefined) body.temperature = request.temperature;
  if (request.top_p !== undefined) body.top_p = request.top_p;
  if (request.top_k !== undefined) body.top_k = request.top_k;
  if (request.max_tokens !== undefined) body.max_tokens = request.max_tokens;
  if (request.seed !== undefined) body.seed = request.seed;
  return body;
}

/**
 * Minimal SSE reader over a web ReadableStream. Yields one parsed event per
 * complete `data:` line group.
 *
 * @param {ReadableStream<Uint8Array>} body
 * @returns {AsyncGenerator<{event: string, data: string}, void, void>}
 */
export async function* iterateSSE(body) {
  const decoder = new TextDecoder();
  /** @type {string[]} */
  let buffer = [];

  for await (const chunk of body) {
    const text = decoder.decode(chunk, { stream: true });
    for (const line of text.split('\n')) {
      const trimmed = line.trim();
      if (trimmed === '') {
        if (buffer.length > 0) {
          yield parseEvent(buffer);
          buffer = [];
        }
        continue;
      }
      if (trimmed.startsWith(':')) continue; // SSE comment / keepalive
      buffer.push(trimmed);
    }
  }
  if (buffer.length > 0) yield parseEvent(buffer);
}

/**
 * @param {string[]} lines
 * @returns {{event: string, data: string}}
 */
function parseEvent(lines) {
  let event = 'message';
  /** @type {string[]} */
  const data = [];
  for (const line of lines) {
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? '' : line.slice(colon + 1).trimStart();
    if (field === 'event') event = value;
    else if (field === 'data') data.push(value);
  }
  return { event, data: data.join('\n') };
}