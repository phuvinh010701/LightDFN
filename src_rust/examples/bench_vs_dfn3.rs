/// bench_vs_dfn3 — compare LightDFN (pulsed + native) vs DFN3 per-frame timing.
///
/// Three contenders:
///   LightDFN pulsed  — Li-GRU expanded into ONNX primitive ops, explicit state I/O
///   LightDFN native  — conv/linear in ONNX, Li-GRU runs in pure Rust (ligru.rs)
///   DFN3             — measured separately via bench-dfn3 in the DeepFilterNet workspace
///                      (DFN3 uses tract 0.21.4 PulsedModel; we use 0.21.15 — version conflict)
///
/// Run from lightdfn-wasm/:
///
///   # 1. Run DFN3 bench in its own workspace first:
///   cd ~/Documents/personal/DeepFilterNet
///   cargo run --release --bin bench-dfn3 --features tract,transforms,default-model
///
///   # 2. Then run this bench, passing DFN3 mean µs:
///   cd lightdfn-wasm
///   cargo run --release --example bench_vs_dfn3 -- --dfn3-mean 1234.5
///
use lightdfn_wasm::model_native::LightDFNNative;
use ndarray::{Array2, Array4, Array5};
use std::time::Instant;
use tract_onnx::prelude::*;

type TractModel = TypedSimpleState<TypedModel, TypedSimplePlan<TypedModel>>;

const N_WARMUP: usize = 50;
const N_BENCH: usize = 1000;
const HOP_SIZE: usize = 480; // 10 ms @ 48 kHz

const NB_ERB: usize = 32;
const NB_SPEC: usize = 96;
const EMB_DIM: usize = 256;
const CONV_CH: usize = 64;
const KT_INP: usize = 3;
const KT_DFP: usize = 5;
const N_FREQS: usize = 481; // FFT_SIZE/2+1 = 960/2+1

fn stats(mut v: Vec<f64>) -> (f64, f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let p50 = v[v.len() / 2];
    let p95 = v[(v.len() as f64 * 0.95) as usize];
    let p99 = v[(v.len() as f64 * 0.99) as usize];
    (mean, p50, p95, p99)
}

fn load_model(bytes: &[u8]) -> TractModel {
    let mut m = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .expect("parse")
        .into_typed()
        .expect("into_typed");
    m.declutter().expect("declutter");
    let opt = m.into_optimized().expect("optimize");
    TypedSimpleState::new(SimplePlan::new(opt).expect("plan")).expect("state")
}

// ─── LightDFN pulsed (ONNX Li-GRU, explicit state I/O) ───────────────────────

fn bench_lightdfn_pulsed(enc_bytes: &[u8], erb_bytes: &[u8], df_bytes: &[u8]) -> Vec<f64> {
    let mut enc = load_model(enc_bytes);
    let mut erb = load_model(erb_bytes);
    let mut df  = load_model(df_bytes);

    let h_enc    = Tensor::zero::<f32>(&[1, 1, EMB_DIM]).unwrap();
    let h_erb    = Tensor::zero::<f32>(&[2, 1, EMB_DIM]).unwrap();
    let h_df     = Tensor::zero::<f32>(&[2, 1, EMB_DIM]).unwrap();
    let erb_ctx  = Tensor::zero::<f32>(&[1, 1, KT_INP, NB_ERB]).unwrap();
    let spec_ctx = Tensor::zero::<f32>(&[1, 2, KT_INP, NB_SPEC]).unwrap();
    let c0_ctx   = Tensor::zero::<f32>(&[1, CONV_CH, KT_DFP, NB_SPEC]).unwrap();

    let mut h_enc_t = h_enc.clone();
    let mut h_erb_t = h_erb.clone();
    let mut h_df_t  = h_df.clone();

    macro_rules! run_pulsed_frame {
        () => {{
            let enc_out = enc.run(tvec![
                TValue::from(erb_ctx.clone()),
                TValue::from(spec_ctx.clone()),
                TValue::from(h_enc_t.clone()),
            ]).unwrap();
            h_enc_t = enc_out[7].clone().into_tensor();
            let erb_out = erb.run(tvec![
                enc_out[4].clone(), enc_out[3].clone(), enc_out[2].clone(),
                enc_out[1].clone(), enc_out[0].clone(),
                TValue::from(h_erb_t.clone()),
            ]).unwrap();
            h_erb_t = erb_out[1].clone().into_tensor();
            let df_out = df.run(tvec![
                enc_out[4].clone(),
                TValue::from(c0_ctx.clone()),
                TValue::from(h_df_t.clone()),
            ]).unwrap();
            h_df_t = df_out[1].clone().into_tensor();
        }};
    }

    for _ in 0..N_WARMUP { run_pulsed_frame!(); }
    h_enc_t = h_enc.clone();
    h_erb_t = h_erb.clone();
    h_df_t  = h_df.clone();

    let mut times = Vec::with_capacity(N_BENCH);
    for _ in 0..N_BENCH {
        let t0 = Instant::now();
        run_pulsed_frame!();
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times
}

// ─── LightDFN native (conv ONNX + pure-Rust Li-GRU) ─────────────────────────

fn bench_lightdfn_native(enc_ff: &[u8], erb_ff: &[u8], df_ff: &[u8], ligru_json: &str) -> Vec<f64> {
    let erb_inv_fb = Array2::<f32>::zeros((NB_ERB, N_FREQS));
    let mut model  = LightDFNNative::from_bytes(enc_ff, erb_ff, df_ff, ligru_json, erb_inv_fb)
        .expect("native model load");

    let spec      = Array5::<f32>::zeros((1, 1, 1, N_FREQS, 2));
    let feat_erb  = Array4::<f32>::zeros((1, 1, 1, NB_ERB));
    let feat_spec = Array5::<f32>::zeros((1, 1, 1, NB_SPEC, 2));

    for _ in 0..N_WARMUP {
        model.process_frame(spec.clone(), feat_erb.clone(), feat_spec.clone()).unwrap();
    }
    model.reset();

    let mut times = Vec::with_capacity(N_BENCH);
    for _ in 0..N_BENCH {
        let t0 = Instant::now();
        model.process_frame(spec.clone(), feat_erb.clone(), feat_spec.clone()).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times
}

// ─── main ────────────────────────────────────────────────────────────────────

fn print_row(label: &str, mean: f64, p50: f64, p95: f64, p99: f64, rtf: f64) {
    println!("  {label:<32} mean={mean:7.1}µs  p50={p50:7.1}µs  p95={p95:7.1}µs  p99={p99:7.1}µs  RTF={rtf:5.1}x");
}

fn main() {
    // Parse --dfn3-mean <f64> from argv (optional)
    let dfn3_mean: Option<f64> = {
        let args: Vec<String> = std::env::args().collect();
        args.windows(2)
            .find(|w| w[0] == "--dfn3-mean")
            .and_then(|w| w[1].parse().ok())
    };

    let sep = "=".repeat(80);
    println!("{sep}");
    println!(" LightDFN (pulsed + native) vs DFN3  —  tract per-frame benchmark");
    println!(" {N_BENCH} frames, {N_WARMUP} warmup, 10 ms/frame @ 48 kHz");
    println!("{sep}\n");

    let pulsed_dir = std::path::Path::new("pkg");
    let native_dir = std::path::Path::new("pkg/native");

    let ld_enc = std::fs::read(pulsed_dir.join("enc.onnx")).expect("pulsed enc.onnx");
    let ld_erb = std::fs::read(pulsed_dir.join("erb_dec.onnx")).expect("pulsed erb_dec.onnx");
    let ld_df  = std::fs::read(pulsed_dir.join("df_dec.onnx")).expect("pulsed df_dec.onnx");

    let nat_enc    = std::fs::read(native_dir.join("enc_ff.onnx")).expect("native enc_ff.onnx");
    let nat_erb    = std::fs::read(native_dir.join("erb_dec_ff.onnx")).expect("native erb_dec_ff.onnx");
    let nat_df     = std::fs::read(native_dir.join("df_dec_ff.onnx")).expect("native df_dec_ff.onnx");
    let ligru_json = std::fs::read_to_string(native_dir.join("ligru_weights.json"))
        .expect("native ligru_weights.json");

    let frame_ms = HOP_SIZE as f64 / 48.0; // 10.0 ms

    println!("Benchmarking LightDFN pulsed…");
    let t = bench_lightdfn_pulsed(&ld_enc, &ld_erb, &ld_df);
    let (mean, p50, p95, p99) = stats(t);
    let pulsed_mean = mean;
    print_row("LightDFN pulsed (Li-GRU ONNX)", mean, p50, p95, p99, frame_ms * 1e3 / mean);
    println!();

    println!("Benchmarking LightDFN native…");
    let t = bench_lightdfn_native(&nat_enc, &nat_erb, &nat_df, &ligru_json);
    let (mean, p50, p95, p99) = stats(t);
    let native_mean = mean;
    print_row("LightDFN native (Rust Li-GRU)", mean, p50, p95, p99, frame_ms * 1e3 / mean);
    println!();

    if let Some(dfn3) = dfn3_mean {
        let rtf = frame_ms * 1e3 / dfn3;
        println!("DFN3 (from bench-dfn3, same machine):");
        print_row("DFN3 (PulsedModel, int8 GRU)", dfn3, 0.0, 0.0, 0.0, rtf);
        println!("  (p50/p95/p99 not available — run bench-dfn3 for full stats)\n");
    } else {
        println!("  [DFN3] Run bench-dfn3 first, then pass --dfn3-mean <µs>:");
        println!("    cd ~/Documents/personal/DeepFilterNet");
        println!("    cargo run --release --bin bench-dfn3 --features tract,transforms,default-model");
        println!();
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("{sep}");
    println!(" RESULTS");
    println!("{sep}");

    let speedup_vs_pulsed = pulsed_mean / native_mean;
    if speedup_vs_pulsed >= 1.0 {
        println!("  native is {speedup_vs_pulsed:.2}x FASTER than pulsed");
    } else {
        println!("  native is {:.2}x SLOWER than pulsed", 1.0 / speedup_vs_pulsed);
    }

    if let Some(dfn3) = dfn3_mean {
        let ratio_native = dfn3 / native_mean;
        if ratio_native >= 1.0 {
            println!("  native is {ratio_native:.2}x FASTER than DFN3");
        } else {
            println!("  native is {:.2}x SLOWER than DFN3", 1.0 / ratio_native);
        }
        let ratio_pulsed = dfn3 / pulsed_mean;
        if ratio_pulsed >= 1.0 {
            println!("  pulsed is {ratio_pulsed:.2}x FASTER than DFN3");
        } else {
            println!("  pulsed is {:.2}x SLOWER than DFN3", 1.0 / ratio_pulsed);
        }
    }

    println!("{sep}");
}
