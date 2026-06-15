const CACHE = 'stocker-v4';
const PRECACHE = [
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(PRECACHE))
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Only handle same-origin requests — CDN requests (jQuery, Bootstrap, etc.) pass through untouched
  if (url.origin !== self.location.origin) {
    return;
  }

  // Never intercept non-GET, API, or predict routes
  if (e.request.method !== 'GET' || url.pathname.startsWith('/api/') || url.pathname.startsWith('/predict')) {
    return;
  }

  // Never cache HTML pages — always fetch fresh from server
  const acceptHeader = e.request.headers.get('accept') || '';
  if (acceptHeader.includes('text/html')) {
    return;
  }

  // Static assets only: cache-first
  if (url.pathname.startsWith('/static/')) {
    e.respondWith(
      caches.match(e.request).then(cached => cached || fetch(e.request).then(res => {
        const clone = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
        return res;
      }))
    );
  }
});
