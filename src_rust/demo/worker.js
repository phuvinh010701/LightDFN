// Web Worker — loads LightDFN WASM (pulsed 3-model architecture with GRU state I/O),
// accumulates 480 raw samples from the AudioWorklet ring buffer, runs inference,
// writes 480 enhanced samples back.
//
// ringbuffer.js is concatenated before this file via URLFromFiles().

const FRAME = 480; // LightDFN hop size (480 samples @ 48 kHz = 10 ms)
const BLOCK = 128; // AudioWorklet block size

// If the output ring buffer is this many frames ahead of what the AWP has
// consumed, we've stalled — drain excess to let the pipeline catch up.
// At 480 samples/frame @ 48 kHz = 10 ms/frame, 20 frames = 200 ms of latency.
const MAX_OUTPUT_FRAMES = 20;

let _reader;       // AudioReader  (raw input from AWP)
let _writer;       // AudioWriter  (enhanced output to AWP)
let _rawRing;      // RingBuffer   (raw, for fill-level queries)
let _denoisedRing; // RingBuffer   (denoised, for fill-level queries)
let _model;        // LightDFNWasm instance
let _accum;        // Float32Array accumulation buffer (FRAME samples)
let _accumPos = 0;
let _interval;

// Perf tracking
let _frameCount    = 0;
let _totalInferMs  = 0;
let _lastReport    = 0;

// Overrun / stall tracking
let _outputOverrunCount = 0;
let _drainCount         = 0;

function processAvailable() {
  if (!_model) return;

  const tmp = new Float32Array(BLOCK);

  while (_reader.available_read() >= BLOCK) {
    _reader.dequeue(tmp);

    let src = 0;
    while (src < BLOCK) {
      const room = FRAME - _accumPos;
      const take = Math.min(room, BLOCK - src);
      _accum.set(tmp.subarray(src, src + take), _accumPos);
      _accumPos += take;
      src        += take;

      if (_accumPos === FRAME) {
        const t0 = performance.now();
        const view = _model.process_frame(_accum);
        // view is a Float32Array returned from WASM — copy before next call.
        const enhanced = view.slice();
        const inferMs = performance.now() - t0;

        _frameCount++;
        _totalInferMs += inferMs;

        // Check if the output ring buffer is getting too full (AWP not consuming
        // fast enough, or a stall). Drain oldest frames to avoid a deadlock where
        // the worker stops writing because the buffer is full and the AWP keeps
        // reading silence because no new data arrives.
        const outputSamples = _denoisedRing.available_read();
        const outputFrames  = Math.floor(outputSamples / FRAME);
        if (outputFrames >= MAX_OUTPUT_FRAMES) {
          // Drain down to half MAX so we get a clean recovery
          const toDrain = Math.floor((outputFrames - MAX_OUTPUT_FRAMES / 2) * FRAME);
          const drainBuf = new Float32Array(toDrain);
          _denoisedRing.pop(drainBuf);
          _drainCount++;
          console.warn(`[worker] output buffer stall — drained ${toDrain} samples (${outputFrames} frames backed up). drain #${_drainCount}`);
        }

        const written = _writer.enqueue(enhanced);
        if (written !== FRAME) {
          _outputOverrunCount++;
          console.warn(`[worker] output overrun #${_outputOverrunCount} — enqueued ${written}/${FRAME} samples`);
        }

        _accumPos = 0;
      }
    }
  }

  // Periodic stats report
  if (_frameCount > 0 && _frameCount - _lastReport >= 100) {
    const avg         = (_totalInferMs / _frameCount).toFixed(2);
    const rawFill     = _rawRing.available_read();
    const denoisedFill = _denoisedRing.available_read();
    postMessage({
      type: 'STATS',
      frames: _frameCount,
      avgMs: parseFloat(avg),
      rawFill,
      denoisedFill,
      outputOverruns: _outputOverrunCount,
      drains: _drainCount,
    });
    console.log(
      `[worker] ${_frameCount} frames | avg ${avg} ms/frame | ` +
      `raw=${rawFill} denoised=${denoisedFill} | overruns=${_outputOverrunCount} drains=${_drainCount}`
    );
    _lastReport = _frameCount;
  }
}

onmessage = async (e) => {
  const { command } = e.data;

  switch (command) {
    case 'init': {
      const {
        wasmBytes,
        encBytes, erbDecBytes, dfDecBytes,
        erbJson,
        encLayers, erbLayers, dfLayers,
        rawSab, denoisedSab, pkgBaseUrl,
      } = e.data;

      _rawRing      = new RingBuffer(rawSab,      Float32Array);
      _denoisedRing = new RingBuffer(denoisedSab, Float32Array);
      _reader = new AudioReader(_rawRing);
      _writer = new AudioWriter(_denoisedRing);
      _accum  = new Float32Array(FRAME);

      const wasmModule = await WebAssembly.compile(wasmBytes);
      const { default: init, LightDFNWasm } = await import(pkgBaseUrl + 'lightdfn_wasm.js');
      await init({ module_or_path: wasmModule });

      const t0 = performance.now();
      // Pulsed ONNX API: three models + ERB filterbank JSON + layer counts.
      // GRU hidden states are managed internally as explicit tensor I/O each frame.
      _model = new LightDFNWasm(
        new Uint8Array(encBytes),
        new Uint8Array(erbDecBytes),
        new Uint8Array(dfDecBytes),
        erbJson,
        encLayers || 1,
        erbLayers || 2,
        dfLayers  || 2,
      );
      const initMs = (performance.now() - t0).toFixed(0);
      console.log(`[worker] LightDFN ready in ${initMs}ms`);
      postMessage({ type: 'INIT_TIME', ms: parseInt(initMs) });

      // Warmup: prime V8 JIT + tract lazy init
      const silent = new Float32Array(FRAME);
      for (let i = 0; i < 10; i++) _model.process_frame(silent);
      _frameCount = 0; _totalInferMs = 0; _lastReport = 0;
      _outputOverrunCount = 0; _drainCount = 0;
      console.log('[worker] warmup done');

      _interval = setInterval(processAvailable, 0);
      postMessage({ type: 'READY' });
      break;
    }

    case 'stop': {
      clearInterval(_interval);
      if (_model) {
        const avg = _frameCount > 0 ? (_totalInferMs / _frameCount).toFixed(2) : '—';
        console.log(`[worker] stopped. ${_frameCount} frames, avg ${avg} ms/frame | overruns=${_outputOverrunCount} drains=${_drainCount}`);
        _model.free();
        _model = null;
      }
      postMessage({ type: 'STOPPED' });
      break;
    }

    default:
      console.warn('[worker] unknown command', command);
  }
};
