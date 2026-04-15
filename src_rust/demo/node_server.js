// Minimal static file server with the COOP + COEP headers required for
// SharedArrayBuffer (used by the ring buffers between AudioWorklet and Worker).
//
// Usage:
//   node demo/node_server.js [port]
//
// Then open: http://localhost:8080/demo/

const http = require('http');
const url  = require('url');
const path = require('path');
const fs   = require('fs');

const port = parseInt(process.argv[2] || '8080', 10);

const MIME = {
  html: 'text/html',
  js:   'text/javascript',
  mjs:  'text/javascript',
  wasm: 'application/wasm',
  json: 'application/json',
  onnx: 'application/octet-stream',
  css:  'text/css',
  png:  'image/png',
  ico:  'image/x-icon',
};

// Serve from the lightdfn-wasm/ project root so that /pkg/ and /demo/ both resolve.
const ROOT = path.resolve(__dirname, '..');

http.createServer((req, res) => {
  let pathname = url.parse(req.url).pathname;

  // Redirect bare root to /demo/ so the browser base URL is correct
  // (relative asset fetches like ringbuffer.js resolve to /demo/ringbuffer.js).
  if (pathname === '/') {
    res.writeHead(302, { Location: '/demo/' });
    res.end();
    return;
  }

  const filePath = path.join(ROOT, pathname);

  // Prevent directory traversal
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403); res.end('Forbidden'); return;
  }

  let resolved = filePath;
  try {
    if (fs.statSync(resolved).isDirectory()) resolved = path.join(resolved, 'index.html');
  } catch (_) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('404 Not Found: ' + pathname);
    return;
  }

  fs.readFile(resolved, (err, data) => {
    if (err) {
      res.writeHead(500); res.end('500 ' + err.message); return;
    }

    const ext      = resolved.split('.').pop().toLowerCase();
    const mimeType = MIME[ext] || 'application/octet-stream';

    res.writeHead(200, {
      'Content-Type': mimeType,
      // These two headers are REQUIRED for SharedArrayBuffer in modern browsers.
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    res.end(data);
  });
}).listen(port, () => {
  console.log(`LightDFN demo server running at:`);
  console.log(`  http://localhost:${port}/demo/`);
  console.log(`\nPress Ctrl-C to stop.`);
});
