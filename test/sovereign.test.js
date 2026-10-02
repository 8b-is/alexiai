import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SovereigntyViolation,
  assertLocal,
  checkLocal,
  isLoopbackHostname,
  installEgressGuard,
  attestation,
} from '../src/sovereign.js';

test('loopback is recognized by name and by literal', () => {
  for (const host of ['localhost', '127.0.0.1', '127.1.2.3', '::1', '[::1]', '0.0.0.0']) {
    assert.ok(isLoopbackHostname(host), `${host} should be loopback`);
  }
});

test('everything else is not loopback', () => {
  const remote = [
    'api.openai.com',
    'api.anthropic.com',
    'example.com',
    '10.0.0.1',
    '192.168.1.1',
    '8.8.8.8',
    '169.254.169.254', // link-local metadata endpoint
    '',
    null,
  ];
  for (const host of remote) {
    assert.equal(isLoopbackHostname(host), false, `${host} must not be loopback`);
  }
});

test('a spoof that merely starts with a loopback name is refused', () => {
  assert.throws(() => assertLocal('http://localhost.evil.com/v1'), SovereigntyViolation);
  assert.throws(() => assertLocal('http://127.0.0.1.evil.com/v1'), SovereigntyViolation);
});

test('userinfo cannot smuggle a remote host past the parser', () => {
  assert.throws(() => assertLocal('http://localhost@evil.com/v1'), SovereigntyViolation);
});

test('loopback URLs pass', () => {
  assert.equal(assertLocal('http://127.0.0.1:1337/v1').hostname, '127.0.0.1');
  assert.equal(assertLocal('http://localhost:8787/api').port, '8787');
});

test('non-http protocols are refused', () => {
  assert.throws(() => assertLocal('file:///etc/passwd'), SovereigntyViolation);
  assert.throws(() => assertLocal('ftp://127.0.0.1/'), SovereigntyViolation);
});

test('unparseable input is refused', () => {
  assert.throws(() => assertLocal('not a url'), SovereigntyViolation);
});

test('checkLocal does not throw', () => {
  assert.equal(checkLocal('http://127.0.0.1:1/').ok, true);
  const bad = checkLocal('https://api.openai.com/v1');
  assert.equal(bad.ok, false);
  assert.ok(bad.error instanceof SovereigntyViolation);
});

test('the egress guard blocks a remote fetch before a socket opens', async () => {
  const restore = installEgressGuard();
  try {
    await assert.rejects(
      () => fetch('https://api.openai.com/v1/chat/completions'),
      SovereigntyViolation
    );
  } finally {
    restore();
  }
});

test('the guard is idempotent and restorable', () => {
  const original = globalThis.fetch;
  const restoreA = installEgressGuard();
  const guarded = globalThis.fetch;
  const restoreB = installEgressGuard();
  assert.equal(globalThis.fetch, guarded, 'second install should be a no-op');

  restoreB();
  restoreA();
  assert.equal(globalThis.fetch, original);
});

test('attestation is honest about whether the guard is installed', () => {
  assert.equal(attestation().installed, false);
  const restore = installEgressGuard();
  try {
    assert.equal(attestation().installed, true);
  } finally {
    restore();
  }
});