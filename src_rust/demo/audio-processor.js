// AudioWorklet processor — runs LightDFN WASM directly on the audio thread.
//
// CRITICAL buffer sizing: use 8192 (matches reference), NOT FRAME*4=1920.
// A small buffer causes write-laps-read corruption → gradual fade to silence.
//
// lightdfn_wasm.js is inlined as an IIFE before this file (see script.js).

const _now = (typeof performance !== 'undefined' && typeof performance.now === 'function')
  ? () => performance.now()
  : () => currentTime * 1000;

const FRAME = 480;
const BUF   = 8192;

class LightDFNProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    const { wasmModule, encBytes, erbDecBytes, dfDecBytes, erbJson,
            encLayers, erbLayers, dfLayers } = options.processorOptions;

    initSync({ module: wasmModule });

    this._model = new LightDFNWasm(
      new Uint8Array(encBytes),
      new Uint8Array(erbDecBytes),
      new Uint8Array(dfDecBytes),
      erbJson,
      encLayers || 1,
      erbLayers || 2,
      dfLayers  || 2,
    );

    this._inputBuf       = new Float32Array(BUF);
    this._inputWritePos  = 0;
    this._inputReadPos   = 0;

    this._outputBuf      = new Float32Array(BUF);
    this._outputWritePos = 0;
    this._outputReadPos  = 0;

    this._tempFrame = new Float32Array(FRAME);
    this._bypass    = true;

    this._frameCount  = 0;
    this._totalMs     = 0;
    this._lastReport  = 0;
    this._underruns   = 0;

    // Diagnostics: track model output RMS to detect stuck-silent state
    this._outRmsSmooth = 0;   // exponential moving average of model output RMS
    this._inRmsSmooth  = 0;
    this._diagTick     = 0;

    this.port.onmessage = (e) => {
      if (e.data.type === 'SET_BYPASS') {
        this._bypass = Boolean(e.data.value);
      }
    };

    console.log('[LightDFN AWP] ready ✓  BUF=' + BUF);
  }

  _inputAvailable() {
    return (this._inputWritePos - this._inputReadPos + BUF) % BUF;
  }

  _outputAvailable() {
    return (this._outputWritePos - this._outputReadPos + BUF) % BUF;
  }

  process(inputList, outputList) {
    const inChans  = inputList[0]  ?? [];
    const outChans = outputList[0] ?? [];
    if (outChans.length === 0) return true;

    const blockSize = outChans[0].length;

    // Mix to mono
    let mono;
    if (inChans.length === 0) {
      mono = new Float32Array(blockSize);
    } else if (inChans.length === 1) {
      mono = inChans[0];
    } else {
      mono = new Float32Array(blockSize);
      for (let ch = 0; ch < inChans.length; ch++)
        for (let i = 0; i < blockSize; i++) mono[i] += inChans[ch][i];
      const inv = 1 / inChans.length;
      for (let i = 0; i < blockSize; i++) mono[i] *= inv;
    }

    if (this._bypass) {
      for (let ch = 0; ch < outChans.length; ch++) outChans[ch].set(mono);
      return true;
    }

    // Write mono into input ring-buffer
    for (let i = 0; i < blockSize; i++) {
      this._inputBuf[this._inputWritePos] = mono[i];
      this._inputWritePos = (this._inputWritePos + 1) % BUF;
    }

    // Drain full frames → WASM → output ring-buffer
    while (this._inputAvailable() >= FRAME) {
      for (let i = 0; i < FRAME; i++) {
        this._tempFrame[i] = this._inputBuf[this._inputReadPos];
        this._inputReadPos = (this._inputReadPos + 1) % BUF;
      }

      const t0   = _now();
      const view = this._model.process_frame(this._tempFrame);
      this._totalMs += _now() - t0;
      this._frameCount++;

      // Track RMS of model input and output for diagnostics
      let inSq = 0, outSq = 0;
      for (let k = 0; k < FRAME; k++) inSq  += this._tempFrame[k] * this._tempFrame[k];
      for (let k = 0; k < view.length; k++) outSq += view[k] * view[k];
      const inRms  = Math.sqrt(inSq  / FRAME);
      const outRms = Math.sqrt(outSq / (view.length || 1));

      // Exponential moving average (α=0.1)
      this._inRmsSmooth  = 0.9 * this._inRmsSmooth  + 0.1 * inRms;
      this._outRmsSmooth = 0.9 * this._outRmsSmooth + 0.1 * outRms;

      this._diagTick++;
      // Report diagnostics every 50 frames (~0.5 second)
      if (this._diagTick % 50 === 0) {
        this.port.postMessage({
          type:       'DIAG',
          frameCount: this._frameCount,
          inRms:      parseFloat(this._inRmsSmooth.toFixed(5)),
          outRms:     parseFloat(this._outRmsSmooth.toFixed(5)),
          viewLen:    view.length,
          inputAvail: this._inputAvailable(),
          outputAvail:this._outputAvailable(),
          underruns:  this._underruns,
        });
      }

      for (let i = 0; i < view.length; i++) {
        this._outputBuf[this._outputWritePos] = view[i];
        this._outputWritePos = (this._outputWritePos + 1) % BUF;
      }

      if (this._frameCount - this._lastReport >= 200) {
        this.port.postMessage({
          type:      'STATS',
          frames:    this._frameCount,
          avgMs:     parseFloat((this._totalMs / this._frameCount).toFixed(3)),
          underruns: this._underruns,
        });
        this._lastReport = this._frameCount;
      }
    }

    // Drain blockSize samples from output ring-buffer → speakers
    if (this._outputAvailable() >= blockSize) {
      for (let ch = 0; ch < outChans.length; ch++) {
        const out = outChans[ch];
        let rp = this._outputReadPos;
        for (let i = 0; i < blockSize; i++) {
          out[i] = this._outputBuf[rp];
          rp = (rp + 1) % BUF;
        }
      }
      this._outputReadPos = (this._outputReadPos + blockSize) % BUF;
    } else {
      // Underrun: output passthrough (dry signal) so user always hears audio.
      // This only happens transiently at startup or if WASM is too slow.
      this._underruns++;
      for (let ch = 0; ch < outChans.length; ch++) outChans[ch].set(mono);
    }

    return true;
  }
}

registerProcessor('lightdfn-processor', LightDFNProcessor);
