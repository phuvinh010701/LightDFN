/**
 * WASM real-audio test — process actual noisy WAV and check output quality.
 * Run: node test_wasm_audio.mjs
 */

import { readFileSync, writeFileSync } from "fs";
import { performance } from "perf_hooks";

// Minimal WAV parser (PCM int16 only — enough for our test files)
function readWavMono(path) {
  const buf = readFileSync(path);
  const sr   = buf.readUInt32LE(24);
  const ch   = buf.readUInt16LE(22);
  const bits = buf.readUInt16LE(34);
  // Find 'data' chunk
  let offset = 12;
  while (offset < buf.length - 8) {
    const tag = buf.toString("ascii", offset, offset + 4);
    const sz  = buf.readUInt32LE(offset + 4);
    if (tag === "data") {
      offset += 8;
      break;
    }
    offset += 8 + sz;
  }
  const n = (buf.length - offset) / (bits / 8) / ch;
  const samples = new Float32Array(n);
  const scale = bits === 16 ? 32768 : 2147483648;
  for (let i = 0; i < n; i++) {
    const raw = bits === 16 ? buf.readInt16LE(offset + i * ch * 2) : buf.readInt32LE(offset + i * ch * 4);
    samples[i] = raw / scale;
  }
  return { samples, sr };
}

function writeWavF32(path, samples, sr) {
  const n = samples.length;
  const buf = Buffer.alloc(44 + n * 4);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + n * 4, 4);
  buf.write("WAVE", 8); buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(3, 20); // PCM float32
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * 4, 28); buf.writeUInt16LE(4, 32); buf.writeUInt16LE(32, 34);
  buf.write("data", 36); buf.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) buf.writeFloatLE(samples[i], 44 + i * 4);
  writeFileSync(path, buf);
}

const PKG = "./pkg";
const { default: init, LightDFNWasm } = await import(`${PKG}/lightdfn_wasm.js`);
const wasmBytes = readFileSync(`${PKG}/lightdfn_wasm_bg.wasm`);
await init({ module_or_path: wasmBytes });

const encBytes  = readFileSync(`${PKG}/enc.onnx`);
const erbBytes  = readFileSync(`${PKG}/erb_dec.onnx`);
const dfBytes   = readFileSync(`${PKG}/df_dec.onnx`);
const erbFbJson = readFileSync(`${PKG}/erb_filterbank.json`, "utf8");

const proc = new LightDFNWasm(encBytes, erbBytes, dfBytes, erbFbJson, 1, 2, 2);

// Load real noisy audio
const { samples: audio, sr } = readWavMono("../datasets/demo/test.wav");
console.log(`Audio: ${audio.length} samples @ ${sr}Hz  (${(audio.length/sr).toFixed(2)}s)`);

const HOP = 480;
const nFrames = Math.floor(audio.length / HOP);
const enhanced = new Float32Array(nFrames * HOP);

const t0 = performance.now();
for (let i = 0; i < nFrames; i++) {
  const frame = audio.slice(i * HOP, (i + 1) * HOP);
  const out = proc.process_frame(frame);
  enhanced.set(out, i * HOP);
}
const elapsed = performance.now() - t0;

const audioDur = (nFrames * HOP / sr) * 1000;
const rtf = audioDur / elapsed;

// Stats
let rmsIn = 0, rmsOut = 0, nanCount = 0;
for (let i = 0; i < enhanced.length; i++) {
  rmsIn  += audio[i] * audio[i];
  rmsOut += enhanced[i] * enhanced[i];
  if (!isFinite(enhanced[i])) nanCount++;
}
rmsIn  = Math.sqrt(rmsIn  / enhanced.length);
rmsOut = Math.sqrt(rmsOut / enhanced.length);

// Write output
const TRIM = 480;
const outSamples = enhanced.slice(TRIM);
writeWavF32("/tmp/lightdfn_wasm_out.wav", outSamples, sr);

console.log(`\n=== WASM Real-Audio Test Results ===`);
console.log(`  Frames:   ${nFrames}`);
console.log(`  Time:     ${elapsed.toFixed(1)}ms  (audio=${audioDur.toFixed(0)}ms)`);
console.log(`  RTF:      ${rtf.toFixed(1)}x`);
console.log(`  RMS in:   ${rmsIn.toFixed(5)}`);
console.log(`  RMS out:  ${rmsOut.toFixed(5)}`);
console.log(`  NaN/Inf:  ${nanCount}`);
console.log(`  Saved:    /tmp/lightdfn_wasm_out.wav`);

const ok = nanCount === 0 && rmsOut > 1e-5 && rtf > 1;
console.log(ok ? "\n✓ WASM real-audio test PASSED" : "\n✗ WASM real-audio test FAILED");
if (!ok) process.exit(1);
