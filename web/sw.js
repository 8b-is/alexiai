/**
 * ALEXIAI service worker — the offline half of "fully offline capable".
 *
 * Precache the whole shell on install so a cold launch works with no network
 * at all. API traffic is never cached: a stale model answer is worse than no
 * answer.
 */

const CACHE = 'alexiai-v1';

/** The entire app. Nothing else is needed to render. */
const SHELL = [
  '/',
  '/index.html',
  '/app.js',
  '/style.css',
  '/app.webmanifest',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // API and chat are live-only by design.
  if (url.pathname.startsWith('/api/')) return;

  // Shell: cache first, then fill on miss. Same-origin only, by construction.
  event.respondWith(
    caches.match(request).then((hit) => hit || fetch(request).then((response) => {
      if (response.ok && response.type === 'basic') {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(request, copy));
      }
      return response;
    }).catch(() => caches.match('/index.html')))
  );
});