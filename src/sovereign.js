/**
 * Sovereignty — the zero-third-party guard.
 *
 * ALEXIAI's single non-negotiable: no model inference ever leaves this machine.
 * This module is where that promise stops being a README sentence and becomes
 * an enforcement point.
 *
 * The rule is deliberately blunt: an endpoint is usable only if it resolves to
 * a loopback address. There is no allowlist of vendors, no "but this one is
 * fine" escape hatch, and no environment variable that can widen it. If a
 * future feature needs a remote endpoint, that feature has to earn it in review,
 * not in a config file.
 *
 * @module alexiai/sovereign
 */

import { isIP } from 'node:net';

/** Thrown whenever egress is attempted toward a non-local host. */
export class SovereigntyViolation extends Error {
  /**
   * @param {string} target the offending URL or host
   * @param {string} [reason]
   */
  constructor(target, reason = 'endpoint is not loopback') {
    super(`sovereignty violation: ${target} — ${reason}`);
    this.name = 'SovereigntyViolation';
    this.target = target;
  }
}

/**
 * Hosts that count as local. Loopback by name or by literal address.
 * @type {ReadonlySet<string>}
 */
const LOOPBACK_HOSTS = new Set([
  'localhost',
  'localhost.localdomain',
  'ip6-localhost',
  'ip6-loopback',
  '::1',
  '0.0.0.0',
]);

/**
 * Is this hostname a loopback name or literal address?
 * @param {string} hostname
 * @returns {boolean}
 */
export function isLoopbackHostname(hostname) {
  const host = String(hostname || '').toLowerCase().replace(/^\[|\]$/g, '');
  if (LOOPBACK_HOSTS.has(host)) return true;
  const family = isIP(host);
  if (family === 4) return host.startsWith('127.');
  if (family === 6) return host === '::1' || host === '::ffff:127.0.0.1';
  return false;
}

/**
 * Assert that a URL points at this machine. Throws {@link SovereigntyViolation}.
 *
 * @param {string} url absolute URL
 * @returns {URL} the parsed URL, when the assertion passed
 */
export function assertLocal(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    throw new SovereigntyViolation(String(url), 'not a parseable absolute URL');
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new SovereigntyViolation(url, `unsupported protocol ${parsed.protocol}`);
  }
  if (!isLoopbackHostname(parsed.hostname)) {
    throw new SovereigntyViolation(url, `${parsed.hostname} is not loopback`);
  }
  return parsed;
}

/**
 * Non-throwing form.
 * @param {string} url
 * @returns {{ok: true, url: URL} | {ok: false, error: SovereigntyViolation}}
 */
export function checkLocal(url) {
  try {
    return { ok: true, url: assertLocal(url) };
  } catch (error) {
    return {
      ok: false,
      error: /** @type {Error} */ (error).name === 'SovereigntyViolation'
        ? /** @type {SovereigntyViolation} */ (error)
        : new SovereigntyViolation(String(url), /** @type {Error} */ (error).message),
    };
  }
}

/**
 * Install a process-wide egress guard.
 *
 * Wraps `globalThis.fetch` so that any request to a non-loopback host rejects
 * before a socket is opened. This is the backstop: even a transitive dependency
 * or a careless `fetch('https://api.openai.com/...')` cannot leave the machine.
 *
 * Returns a restore function.
 *
 * @returns {() => void} idempotent restore
 */
export function installEgressGuard() {
  const original = globalThis.fetch;
  if (original && original[GUARD]) return original[RESTORE];

  /** @type {typeof fetch} */
  const guarded = async (input, init) => {
    const url = typeof input === 'string' ? input : (input instanceof URL ? input.href : input?.url);
    if (url !== undefined) assertLocal(url);
    return original(input, init);
  };

  guarded[GUARD] = true;
  guarded[RESTORE] = () => {
    if (globalThis.fetch === guarded) globalThis.fetch = original;
  };
  globalThis.fetch = guarded;
  return guarded[RESTORE];
}

const GUARD = Symbol.for('alexiai.sovereignty.guard');
const RESTORE = Symbol.for('alexiai.sovereignty.restore');

/**
 * A short attestation, printed by `alexiai doctor` and shown in the UI so the
 * guarantee is visible rather than merely claimed.
 * @returns {{policy: string, loopbackHosts: string[], installed: boolean}}
 */
export function attestation() {
  return {
    policy: 'inference never leaves this machine',
    loopbackHosts: [...LOOPBACK_HOSTS].sort(),
    installed: Boolean(globalThis.fetch?.[GUARD]),
  };
}