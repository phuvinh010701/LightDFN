/// End-to-end inference test: load demo audio → denoise → save output.
///
/// Usage (run from lightdfn-wasm/):
///   cargo run --release --example infer_audio -- [input.wav] [output.wav]
///
/// Defaults:
///   input : ../datasets/demo/test.wav
///   output: /tmp/lightdfn_out.wav
///
/// Layer counts default to (enc=1, erb=2, df=2); override via env vars:
///   ENC_LAYERS=1 ERB_LAYERS=2 DF_LAYERS=2 cargo run --release --example infer_audio
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use lightdfn_wasm::streaming::StreamingProcessor;
use ndarray::Array1;
use std::path::Path;
use std::time::Instant;

const HOP_SIZE: usize = 480;
const SAMPLE_RATE: u32 = 48_000;

fn load_wav_mono_f32(path: &Path) -> (Vec<f32>, u32) {
    let mut reader = WavReader::open(path).expect("Cannot open WAV");
    let spec = reader.spec();
    let sr = spec.sample_rate;

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / 2147483648.0)
            .collect(),
        _ => panic!(
            "Unsupported WAV format: {:?} {}bit",
            spec.sample_format, spec.bits_per_sample
        ),
    };

    // If stereo, take left channel
    let mono: Vec<f32> = if spec.channels == 2 {
        samples.iter().step_by(2).copied().collect()
    } else {
        samples
    };

    (mono, sr)
}

fn save_wav_f32(path: &Path, samples: &[f32], sr: u32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec).expect("Cannot create output WAV");
    for &s in samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("../datasets/demo/test.wav");
    let output_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("/tmp/lightdfn_out.wav");

    let enc_layers = env_usize("ENC_LAYERS", 1);
    let erb_layers = env_usize("ERB_LAYERS", 2);
    let df_layers = env_usize("DF_LAYERS", 2);

    println!("=== LightDFN infer_audio (pulsed ONNX) ===");
    println!("Input : {input_path}");
    println!("Output: {output_path}");
    println!("Layers: enc={enc_layers} erb={erb_layers} df={df_layers}");

    // ── Load models ─────────────────────────────────────────────────────────
    println!("\nLoading models…");
    let enc_bytes = std::fs::read("pkg/enc.onnx").expect("pkg/enc.onnx not found");
    let erb_dec_bytes =
        std::fs::read("pkg/erb_dec.onnx").expect("pkg/erb_dec.onnx not found");
    let df_dec_bytes =
        std::fs::read("pkg/df_dec.onnx").expect("pkg/df_dec.onnx not found");
    println!(
        "  enc={:.0}KB  erb_dec={:.0}KB  df_dec={:.0}KB",
        enc_bytes.len() as f64 / 1024.0,
        erb_dec_bytes.len() as f64 / 1024.0,
        df_dec_bytes.len() as f64 / 1024.0,
    );

    // ── Load ERB filterbank ──────────────────────────────────────────────────
    let erb_json =
        std::fs::read_to_string("pkg/erb_filterbank.json").expect("pkg/erb_filterbank.json not found");
    let erb_data: serde_json::Value = serde_json::from_str(&erb_json).unwrap();
    let erb_fb_vec: Vec<Vec<f32>> =
        serde_json::from_value(erb_data["erb_fb"].clone()).unwrap();
    let n_freqs = erb_fb_vec.len();
    let nb_erb = erb_fb_vec[0].len();
    let mut erb_fb = ndarray::Array2::<f32>::zeros((n_freqs, nb_erb));
    for (i, row) in erb_fb_vec.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            erb_fb[[i, j]] = v;
        }
    }

    let erb_inv_fb_vec: Vec<Vec<f32>> =
        serde_json::from_value(erb_data["erb_inv_fb"].clone()).unwrap();
    let nb_erb2 = erb_inv_fb_vec.len();
    let n_freqs2 = erb_inv_fb_vec[0].len();
    let mut erb_inv_fb = ndarray::Array2::<f32>::zeros((nb_erb2, n_freqs2));
    for (i, row) in erb_inv_fb_vec.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            erb_inv_fb[[i, j]] = v;
        }
    }
    println!("  ERB fb: [{n_freqs}, {nb_erb}]  inv_fb: [{nb_erb2}, {n_freqs2}]");

    // ── Init processor ────────────────────────────────────────────────────────
    let t_init = Instant::now();
    let mut proc = StreamingProcessor::new(
        &enc_bytes,
        &erb_dec_bytes,
        &df_dec_bytes,
        erb_fb,
        erb_inv_fb,
        enc_layers,
        erb_layers,
        df_layers,
    )
    .unwrap();
    println!("  Init: {:.1}ms", t_init.elapsed().as_secs_f64() * 1000.0);

    // ── Load audio ────────────────────────────────────────────────────────────
    println!("\nLoading audio…");
    let (audio, sr) = load_wav_mono_f32(Path::new(input_path));
    let duration_s = audio.len() as f64 / sr as f64;
    println!("  {:.2}s @ {}Hz ({} samples)", duration_s, sr, audio.len());
    if sr != SAMPLE_RATE {
        eprintln!("WARNING: sample rate {sr} != expected {SAMPLE_RATE}. Results may be incorrect.");
    }

    // ── Process frame by frame ────────────────────────────────────────────────
    println!("\nProcessing…");
    let mut enhanced: Vec<f32> = Vec::with_capacity(audio.len());
    let t_proc = Instant::now();
    let n_frames = audio.len() / HOP_SIZE;

    for i in 0..n_frames {
        let frame = &audio[i * HOP_SIZE..(i + 1) * HOP_SIZE];
        let arr = Array1::from_vec(frame.to_vec());
        let out = proc.process_frame(arr.view()).unwrap();
        enhanced.extend_from_slice(out.as_slice().unwrap());
    }

    // Handle any remaining samples (pad to hop_size)
    let remainder = audio.len() % HOP_SIZE;
    if remainder > 0 {
        let mut pad = audio[n_frames * HOP_SIZE..].to_vec();
        pad.resize(HOP_SIZE, 0.0);
        let arr = Array1::from_vec(pad);
        let out = proc.process_frame(arr.view()).unwrap();
        enhanced.extend_from_slice(&out.as_slice().unwrap()[..remainder]);
    }

    let proc_ms = t_proc.elapsed().as_secs_f64() * 1000.0;
    let rtf = (duration_s * 1000.0) / proc_ms;
    println!("  Processed {n_frames} frames in {proc_ms:.1}ms  →  RTF {rtf:.1}x");

    // DFN3 convention: trim fft_size - hop_size = 480 samples of latency at start
    let trim = 480usize;
    let trimmed = if enhanced.len() > trim {
        &enhanced[trim..]
    } else {
        &enhanced
    };

    // ── Save output ───────────────────────────────────────────────────────────
    save_wav_f32(Path::new(output_path), trimmed, SAMPLE_RATE);
    println!("\nSaved: {output_path}  ({} samples)", trimmed.len());

    // ── Quick sanity: check output is not silent or NaN ───────────────────────
    let n_nan = trimmed
        .iter()
        .filter(|s| s.is_nan() || s.is_infinite())
        .count();
    let rms: f32 =
        (trimmed.iter().map(|s| s * s).sum::<f32>() / trimmed.len() as f32).sqrt();
    println!("  RMS={rms:.4}  NaN/Inf={n_nan}");
    if n_nan > 0 {
        eprintln!("ERROR: output contains {n_nan} NaN/Inf samples!");
        std::process::exit(1);
    }
    if rms < 1e-6 {
        eprintln!("WARNING: output is nearly silent (RMS={rms:.2e})");
    } else {
        println!("  ✓ Output looks good");
    }
}
