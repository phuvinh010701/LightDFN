// ── LightDFN demo — main thread script ────────────────────────────────────────
// Pattern mirrors livekit-deepfilternet3-noise-filter:
//   1. Fetch & pre-compile WASM on main thread (structured-cloneable Module)
//   2. Fetch model bytes (ONNX + JSON)
//   3. Create AudioWorklet from blob (lightdfn_wasm.js IIFE + processor code)
//   4. Pass compiled module + model bytes via processorOptions
//   5. No Web Worker — WASM runs directly on the audio thread via initSync()

const SR = 48000;
const ENC_LAYERS = 1;
const ERB_LAYERS = 2;
const DF_LAYERS  = 2;

let audioCtx   = null;
let awpNode    = null;
let gainNode   = null;
let active     = false;
let suppEnabled = false;

// ── UI refs ────────────────────────────────────────────────────────────────────
const toggleBtn     = document.getElementById('toggle');
const gainSlider    = document.getElementById('gain');
const statusText    = document.getElementById('status-text');
const statusDot     = document.getElementById('statusDot');
const bypassCheck   = document.getElementById('bypassCheck');
const bypassLabel   = document.getElementById('bypassLabel');
const statsRow      = document.getElementById('statsRow');
const statFrames    = document.getElementById('statFrames');
const statMs        = document.getElementById('statMs');
const statCpu       = document.getElementById('statCpu');
const statUnderruns = document.getElementById('statUnderruns');

// ── Status helpers ─────────────────────────────────────────────────────────────
function setStatus(text, dotClass) {
  statusText.textContent = text;
  statusDot.className    = 'dot ' + dotClass;
}

// ── Bypass toggle ──────────────────────────────────────────────────────────────
bypassCheck.addEventListener('change', () => {
  suppEnabled = bypassCheck.checked;
  bypassLabel.textContent = suppEnabled ? 'Suppression: ON' : 'Suppression: OFF';
  if (awpNode) {
    awpNode.port.postMessage({ type: 'SET_BYPASS', value: !suppEnabled });
  }
});

// ── Main start / stop ──────────────────────────────────────────────────────────
toggleBtn.addEventListener('click', async () => {
  if (!active) {
    toggleBtn.disabled = true;
    setStatus('Loading model…', 'loading');
    try {
      await start();
      toggleBtn.textContent = '■ Stop';
      toggleBtn.classList.add('running');
      toggleBtn.disabled  = false;
      gainSlider.disabled = false;
      bypassCheck.disabled = false;
      active = true;
    } catch (err) {
      console.error('[LightDFN] start error:', err);
      setStatus('Error: ' + err.message, 'error');
      toggleBtn.textContent = '▶ Start';
      toggleBtn.disabled    = false;
    }
  } else {
    stop();
    toggleBtn.textContent = '▶ Start';
    toggleBtn.classList.remove('running');
    gainSlider.disabled  = true;
    bypassCheck.disabled = true;
    bypassCheck.checked  = false;
    bypassLabel.textContent = 'Suppression: OFF';
    statsRow.style.display  = 'none';
    setStatus('Stopped.', 'idle');
    active = false;
    if (window.__stopViz) window.__stopViz();
  }
});

// ── Gain ───────────────────────────────────────────────────────────────────────
gainSlider.addEventListener('input', () => {
  if (gainNode) gainNode.gain.value = parseFloat(gainSlider.value);
});

// ── Core pipeline ──────────────────────────────────────────────────────────────
async function start() {
  setStatus('Fetching WASM & model (~5.8 MB)…', 'loading');

  // Fetch everything in parallel
  const [wasmModule, encBytes, erbDecBytes, dfDecBytes, erbJson,
         wasmBindingsText, processorText] = await Promise.all([
    // Pre-compile on main thread — WebAssembly.Module is structured-cloneable
    fetch('/pkg/lightdfn_wasm_bg.wasm')
      .then(r => { if (!r.ok) throw new Error('WASM fetch failed: ' + r.status); return r.arrayBuffer(); })
      .then(buf => WebAssembly.compile(buf)),
    fetch('/pkg/enc.onnx').then(r => r.arrayBuffer()),
    fetch('/pkg/erb_dec.onnx').then(r => r.arrayBuffer()),
    fetch('/pkg/df_dec.onnx').then(r => r.arrayBuffer()),
    fetch('/pkg/erb_filterbank.json').then(r => r.text()),
    fetch('/pkg/lightdfn_wasm.js').then(r => r.text()),
    fetch('/demo/audio-processor.js').then(r => r.text()),
  ]);

  setStatus('Requesting microphone…', 'loading');

  audioCtx = new AudioContext({ sampleRate: SR });
  if (audioCtx.state === 'suspended') await audioCtx.resume();

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl:  false,
      sampleRate:       SR,
      channelCount:     1,   // request mono explicitly
    },
    video: false,
  });

  setStatus('Building worklet…', 'loading');

  // ── Build the AudioWorklet blob ──────────────────────────────────────────────
  // Strip ES module syntax from lightdfn_wasm.js and wrap as IIFE.
  // This mirrors the livekit-deepfilternet3 rollup IIFE technique exactly.
  const strippedBindings = wasmBindingsText
    .replace(/^\/\*\s*@ts-self-types[^*]*\*\/\s*\n?/m, '')
    .replace(/^export\s*\{[^}]*\};?\s*$/m, '')
    .replace(/^export\s+((?:async\s+)?function|class|const|let|var)\s/gm, '$1 ')
    .replace(/module_or_path\s*=\s*new URL\([^)]+import\.meta\.url\)/g,
             'module_or_path = null /* import.meta removed */');

  // TextDecoder / TextEncoder polyfill for AudioWorklet scope
  // (Chrome's AudioWorkletGlobalScope does not expose them in some versions).
  const textApiPolyfill = `
var TextDecoder = (typeof TextDecoder !== 'undefined') ? TextDecoder : (function() {
  function TextDecoder(label, opts) { this._fatal = (opts || {}).fatal || false; }
  TextDecoder.prototype.decode = function(input) {
    if (input === undefined || input === null) return '';
    var bytes = (input instanceof Uint8Array) ? input : new Uint8Array(input.buffer || input);
    var out = '', i = 0;
    while (i < bytes.length) {
      var b = bytes[i++];
      if (b < 0x80) { out += String.fromCharCode(b); }
      else if (b < 0xE0) { out += String.fromCharCode(((b & 0x1F) << 6) | (bytes[i++] & 0x3F)); }
      else if (b < 0xF0) { out += String.fromCharCode(((b & 0x0F) << 12) | ((bytes[i++] & 0x3F) << 6) | (bytes[i++] & 0x3F)); }
      else { var cp=((b&0x07)<<18)|((bytes[i++]&0x3F)<<12)|((bytes[i++]&0x3F)<<6)|(bytes[i++]&0x3F); var s=cp-0x10000; out+=String.fromCharCode(0xD800+(s>>10),0xDC00+(s&0x3FF)); }
    }
    return out;
  };
  return TextDecoder;
}());
var TextEncoder = (typeof TextEncoder !== 'undefined') ? TextEncoder : (function() {
  function TextEncoder() {}
  TextEncoder.prototype.encode = function(str) {
    var bytes = [];
    for (var i = 0; i < str.length; i++) {
      var cp = str.charCodeAt(i);
      if (cp >= 0xD800 && cp <= 0xDBFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (str.charCodeAt(++i) - 0xDC00);
      if (cp < 0x80) bytes.push(cp);
      else if (cp < 0x800) bytes.push(0xC0|(cp>>6), 0x80|(cp&0x3F));
      else if (cp < 0x10000) bytes.push(0xE0|(cp>>12), 0x80|((cp>>6)&0x3F), 0x80|(cp&0x3F));
      else bytes.push(0xF0|(cp>>18), 0x80|((cp>>12)&0x3F), 0x80|((cp>>6)&0x3F), 0x80|(cp&0x3F));
    }
    return new Uint8Array(bytes);
  };
  TextEncoder.prototype.encodeInto = function(str, view) {
    var enc = this.encode(str); view.set(enc); return { read: str.length, written: enc.length };
  };
  return TextEncoder;
}());
`;

  const iifeBindings = [
    '(function() {',
    textApiPolyfill,
    strippedBindings,
    // Expose globals needed by the processor.
    // AudioWorkletGlobalScope does not have `self` — use globalThis.
    'globalThis.initSync      = initSync;',
    'globalThis.LightDFNWasm  = LightDFNWasm;',
    '})();',
  ].join('\n');

  const workletBlob = new Blob(
    [iifeBindings, '\n\n', processorText],
    { type: 'application/javascript' }
  );

  await audioCtx.audioWorklet.addModule(URL.createObjectURL(workletBlob));

  // ── Build audio graph ────────────────────────────────────────────────────────
  const micSrc = audioCtx.createMediaStreamSource(stream);

  // Analyser on raw input (for visualizer)
  const analyserIn = audioCtx.createAnalyser();
  analyserIn.fftSize = 256;

  gainNode = audioCtx.createGain();
  gainNode.gain.value = parseFloat(gainSlider.value);

  awpNode = new AudioWorkletNode(audioCtx, 'lightdfn-processor', {
    processorOptions: {
      wasmModule,
      encBytes,
      erbDecBytes,
      dfDecBytes,
      erbJson,
      encLayers: ENC_LAYERS,
      erbLayers: ERB_LAYERS,
      dfLayers:  DF_LAYERS,
    },
    numberOfInputs:     1,
    numberOfOutputs:    1,
    outputChannelCount: [1],
  });

  // Analyser on processed output (for visualizer)
  const analyserOut = audioCtx.createAnalyser();
  analyserOut.fftSize = 256;

  // Graph: mic → analyserIn → awpNode → analyserOut → gainNode → speakers
  micSrc
    .connect(analyserIn)
    .connect(awpNode)
    .connect(analyserOut)
    .connect(gainNode)
    .connect(audioCtx.destination);

  // Register analysers for the visualizer
  if (window.__setAnalysers) window.__setAnalysers(analyserIn, analyserOut);

  // Start with suppression OFF (bypass = true) — user flips the toggle
  awpNode.port.postMessage({ type: 'SET_BYPASS', value: true });

  // ── Stats & diagnostics from worklet ────────────────────────────────────────
  awpNode.port.onmessage = (e) => {
    if (e.data.type === 'STATS') {
      const { frames, avgMs, underruns } = e.data;
      const cpuPct = (avgMs / 10 * 100).toFixed(1);
      statsRow.style.display    = 'flex';
      statFrames.textContent    = frames.toLocaleString();
      statMs.textContent        = avgMs > 0 ? `${avgMs} ms/frame` : `< 0.5 ms/frame`;
      statCpu.textContent       = cpuPct + '%';
      statUnderruns.textContent = underruns ?? 0;
    }
    if (e.data.type === 'DIAG') {
      const d = e.data;
      console.log(`[LightDFN DIAG] frame#${d.frameCount} inRms=${d.inRms} outRms=${d.outRms} viewLen=${d.viewLen} inAvail=${d.inputAvail} outAvail=${d.outputAvail} underruns=${d.underruns}`);
    }
    if (e.data.type === 'STRUCTURE') {
      console.log(`[LightDFN] Worklet input: ${e.data.inputChans} ch, blockSize=${e.data.blockSize}`);
    }
  };

  setStatus('Running ✓  —  WASM in AudioWorklet, no server', 'running');
}

function stop() {
  if (awpNode)   { try { awpNode.port.close(); } catch(_) {} awpNode.disconnect(); awpNode = null; }
  if (gainNode)  { gainNode.disconnect(); gainNode = null; }
  if (audioCtx)  { audioCtx.close(); audioCtx = null; }
}
