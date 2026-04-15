/**
 * Node.js WASM smoke test — verifies LightDFNWasm processes audio frames correctly.
 *
 * Run from lightdfn-wasm/:
 *   node test_wasm.mjs
 */

import { readFileSync } from "fs";
import { performance } from "perf_hooks";

// wasm-pack --target web generates an ESM module that requires initSync or default init.
// In Node we use createRequire + dynamic import with ?url trick, or just read+instantiate manually.
// Simplest: use the _bg.wasm directly via WebAssembly + the generated bindings.

// Read the generated JS glue (target=web) and patch it to work in Node.
// We'll use the node target approach instead — rebuild for node target first.
// Actually easier: use wasm-pack --target nodejs output if present, else use
// the universal approach of reading wasm bytes + init().

const PKG = "./pkg";

// Dynamic import of the ESM module
const { default: init, LightDFNWasm } = await import(`${PKG}/lightdfn_wasm.js`);

// init() for --target web needs a URL or Response — in Node, pass the wasm buffer directly
const wasmBytes = readFileSync(`${PKG}/lightdfn_wasm_bg.wasm`);
await init({ module_or_path: wasmBytes });

console.log("✓ WASM module initialized");

// Load model files
const encBytes  = readFileSync(`${PKG}/enc.onnx`);
const erbBytes  = readFileSync(`${PKG}/erb_dec.onnx`);
const dfBytes   = readFileSync(`${PKG}/df_dec.onnx`);
const erbFbJson = readFileSync(`${PKG}/erb_filterbank.json`, "utf8");

console.log(`  enc=${(encBytes.length/1024).toFixed(0)}KB  erb_dec=${(erbBytes.length/1024).toFixed(0)}KB  df_dec=${(dfBytes.length/1024).toFixed(0)}KB`);

// Create processor
console.log("\nCreating LightDFNWasm...");
const t0 = performance.now();
const proc = new LightDFNWasm(
  encBytes, erbBytes, dfBytes,
  erbFbJson,
  1, // enc_layers
  2, // erb_layers
  2  // df_layers
);
const initMs = performance.now() - t0;
console.log(`  Init: ${initMs.toFixed(1)}ms`);

// Generate test audio: 440 Hz sine, 1 second
const SR = 48000;
const HOP = 480;
const n = SR; // 1 second
const audio = new Float32Array(n);
for (let i = 0; i < n; i++) {
  audio[i] = 0.1 * Math.sin(2 * Math.PI * 440 * i / SR);
}

// Process frame by frame
console.log(`\nProcessing ${Math.floor(n/HOP)} frames...`);
const t1 = performance.now();
let rmsSum = 0;
let nanCount = 0;
const nFrames = Math.floor(n / HOP);

for (let i = 0; i < nFrames; i++) {
  const frame = audio.slice(i * HOP, (i + 1) * HOP);
  const out = proc.process_frame(frame);

  // Check for NaN/Inf
  for (let k = 0; k < out.length; k++) {
    if (!isFinite(out[k])) nanCount++;
    rmsSum += out[k] * out[k];
  }
}

const procMs = performance.now() - t1;
const audioDurMs = (nFrames * HOP / SR) * 1000;
const rtf = audioDurMs / procMs;
const rms = Math.sqrt(rmsSum / (nFrames * HOP));

console.log(`  Processed in ${procMs.toFixed(1)}ms  (audio=${audioDurMs.toFixed(0)}ms  RTF=${rtf.toFixed(1)}x)`);
console.log(`  Output RMS: ${rms.toFixed(5)}  NaN/Inf: ${nanCount}`);
console.log(`  Frame count: ${proc.get_frame_count()}`);

// Correctness checks
let ok = true;
if (nanCount > 0)         { console.error(`  ✗ NaN/Inf in output`); ok = false; }
if (rms < 1e-6)           { console.error(`  ✗ Output is silent (rms=${rms})`); ok = false; }
if (rms > 10)             { console.error(`  ✗ Output is clipped/exploded (rms=${rms})`); ok = false; }
if (rtf < 1)              { console.error(`  ✗ RTF < 1 (too slow: ${rtf.toFixed(2)}x)`); ok = false; }
if (proc.get_frame_count() !== nFrames) { console.error(`  ✗ Frame count mismatch`); ok = false; }

if (ok) {
  console.log(`\n✓ All checks passed — WASM works correctly!`);
} else {
  console.error(`\n✗ Some checks failed`);
  process.exit(1);
}

// Test reset
proc.reset();
console.log("✓ reset() OK");

// Process a few more frames after reset
for (let i = 0; i < 5; i++) {
  const frame = audio.slice(i * HOP, (i + 1) * HOP);
  proc.process_frame(frame);
}
console.log(`✓ post-reset processing OK (frame_count=${proc.get_frame_count()})`);
