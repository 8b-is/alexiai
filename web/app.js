/**
 * ALEXIAI front end.
 *
 * Everything it needs is same-origin. There is no CDN, no font host, no
 * analytics beacon — the service worker precaches the whole app so the second
 * launch works with the network cable pulled.
 */

const $ = (id) => document.getElementById(id);

const transcript = $('transcript');
const composer = $('composer');
const promptBox = $('prompt');
const sendButton = $('send');

/** Conversation turns, replayed to the model each turn. */
const history = [];

/** @type {AbortController | null} */
let inFlight = null;

// ── side panels ────────────────────────────────────────────────────────────

/** Render a definition list from key/value pairs. */
function renderRows(target, rows) {
  const dl = $(target);
  dl.innerHTML = '';
  for (const [key, value, cls] of rows) {
    const wrap = document.createElement('div');
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    dd.textContent = value;
    if (cls) dd.className = cls;
    wrap.append(dt, dd);
    dl.append(wrap);
  }
}

/** Fold the field over a string and paint the chips. */
function paintField(field) {
  const box = $('field');
  box.innerHTML = '';
  const entries = [
    ['seed', field.seed, ''],
    ...Object.entries(field.layers).map(([k, v]) => [k, Number(v).toFixed(4), k === field.dominant ? 'dominant' : '']),
    ['coherence', field.coherence.toFixed(4), ''],
  ];
  for (const [key, value, cls] of entries) {
    const chip = document.createElement('span');
    chip.className = `chip ${cls}`;
    chip.innerHTML = `${key} <b></b>`;
    chip.querySelector('b').textContent = String(value);
    box.append(chip);
  }
}

async function refreshPanels() {
  try {
    const health = await (await fetch('/api/health')).json();
    const sov = health.sovereignty ?? {};
    renderRows('sovereignty', [
      ['policy', sov.policy ?? 'unknown', 'ok'],
      ['guard', sov.installed ? 'installed' : 'off', sov.installed ? 'ok' : 'bad'],
      ['endpoint', health.endpoint ?? '—', health.ok ? 'ok' : 'bad'],
      ['local model', health.ok ? `up ${health.latencyMs}ms` : 'not running', health.ok ? 'ok' : 'bad'],
    ]);
  } catch {
    renderRows('sovereignty', [['error', 'server unreachable', 'bad']]);
  }

  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const catalog = data.catalog ?? [];
    renderRows('models', catalog.length
      ? catalog.slice(0, 6).map((m) => [m.id, m.contextWindow ? `${m.contextWindow} ctx` : 'local', 'ok'])
      : [['catalog', 'empty', 'bad']]);
  } catch {
    renderRows('models', [['error', 'no catalog', 'bad']]);
  }

  try {
    const bench = await (await fetch('/api/bench?dim=256')).json();
    renderRows('substrate', [
      ['ternary', `${bench.bitsPerWeight.toFixed(3)} b/w`, 'ok'],
      ['compression', `${bench.compression}× vs f32`, 'ok'],
      ['error vs dense', `${(bench.relativeError * 100).toFixed(2)}%`, ''],
    ]);
  } catch {
    renderRows('substrate', [['error', 'bench unavailable', 'bad']]);
  }
}

// ── transcript ────────────────────────────────────────────────────────────

function addTurn(who, text) {
  const empty = transcript.querySelector('.empty');
  if (empty) empty.remove();

  const turn = document.createElement('div');
  turn.className = `turn ${who}`;
  const label = document.createElement('span');
  label.className = 'who';
  label.textContent = who;
  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = text;
  turn.append(label, body);
  transcript.append(turn);
  transcript.scrollTop = transcript.scrollHeight;
  return body;
}

function addGrain(text) {
  const last = transcript.lastElementChild;
  if (!last) return;
  const grain = document.createElement('div');
  grain.className = 'grain';
  grain.textContent = text;
  last.append(grain);
}

// ── the turn ──────────────────────────────────────────────────────────────

composer.addEventListener('submit', async (event) => {
  event.preventDefault();
  const prompt = promptBox.value.trim();
  if (!prompt || inFlight) return;

  promptBox.value = '';
  sendButton.disabled = true;
  addTurn('you', prompt);
  history.push({ role: 'user', content: prompt });
  const body = addTurn('alexiai', '');

  inFlight = new AbortController();
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt, history: history.slice(0, -1) }),
      signal: inFlight.signal,
    });

    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    let text = '';

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;

      // SSE frames are separated by a blank line.
      let split;
      while ((split = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);

        const event = frame.match(/^event: (.+)$/m)?.[1];
        const data = frame.match(/^data: (.*)$/m)?.[1];
        if (!event || !data) continue;

        if (event === 'gaia') {
          const parsed = JSON.parse(data);
          paintField(parsed.field);
          addGrain(
            `field · dominant ${parsed.field.dominant} · coherence ${parsed.field.coherence.toFixed(3)}` +
            ` · T=${parsed.sampling.temperature} · top_p=${parsed.sampling.top_p}`
          );
        } else if (event === 'delta') {
          text += JSON.parse(data).text;
          body.textContent = text;
          transcript.scrollTop = transcript.scrollHeight;
        } else if (event === 'error') {
          throw new Error(JSON.parse(data).message);
        }
      }
    }

    if (text) history.push({ role: 'assistant', content: text });
    else body.textContent = '(no output — is your local model server running?)';
  } catch (error) {
    body.textContent = `— ${error.message}`;
  } finally {
    inFlight = null;
    sendButton.disabled = false;
    promptBox.focus();
  }
});

document.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    composer.requestSubmit();
  }
  if (event.key === 'Escape' && inFlight) inFlight.abort();
});

// ── boot ──────────────────────────────────────────────────────────────────

refreshPanels();
fetch('/api/gaia').then((r) => r.json()).then((d) => paintField(d));

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {
    /* offline caching is a bonus, not a requirement */
  });
}