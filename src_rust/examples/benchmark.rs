use lightdfn_wasm::streaming;
use ndarray::Array1;
use std::time::Instant;

// Default layer counts — override with CLI args if needed
const ENC_LAYERS: usize = 1;
const ERB_LAYERS: usize = 2;
const DF_LAYERS: usize = 2;

fn main() {
    let sep = "=".repeat(60);
    println!("{}", sep);
    println!("LightDFN Rust Pulsed Benchmark");
    println!("{}", sep);
    println!();

    // 1. Load models (pulsed ONNX — GRU states as explicit tensor I/O)
    println!("1. Loading models…");
    let enc_bytes = std::fs::read("pkg/enc.onnx").expect("pkg/enc.onnx not found");
    let erb_dec_bytes =
        std::fs::read("pkg/erb_dec.onnx").expect("pkg/erb_dec.onnx not found");
    let df_dec_bytes =
        std::fs::read("pkg/df_dec.onnx").expect("pkg/df_dec.onnx not found");
    println!(
        "   enc={:.0}KB  erb_dec={:.0}KB  df_dec={:.0}KB",
        enc_bytes.len() as f64 / 1024.0,
        erb_dec_bytes.len() as f64 / 1024.0,
        df_dec_bytes.len() as f64 / 1024.0,
    );

    // 2. Load ERB filterbank
    println!("2. Loading ERB…");
    let erb_json =
        std::fs::read_to_string("pkg/erb_filterbank.json").expect("Failed to load ERB");
    let erb_data: serde_json::Value = serde_json::from_str(&erb_json).unwrap();
    let erb_fb_vec: Vec<Vec<f32>> =
        serde_json::from_value(erb_data["erb_fb"].clone()).unwrap();

    let n_freqs = erb_fb_vec.len();
    let nb_erb = erb_fb_vec[0].len();
    let mut erb_fb = ndarray::Array2::zeros((n_freqs, nb_erb));
    for (i, row) in erb_fb_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            erb_fb[[i, j]] = val;
        }
    }
    println!("   Shape: [{}, {}]", n_freqs, nb_erb);

    let erb_inv_fb_vec: Vec<Vec<f32>> =
        serde_json::from_value(erb_data["erb_inv_fb"].clone()).unwrap();
    let nb_erb2 = erb_inv_fb_vec.len();
    let n_freqs2 = erb_inv_fb_vec[0].len();
    let mut erb_inv_fb = ndarray::Array2::<f32>::zeros((nb_erb2, n_freqs2));
    for (i, row) in erb_inv_fb_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            erb_inv_fb[[i, j]] = val;
        }
    }

    // 3. Init processor
    println!("3. Initializing (enc_layers={ENC_LAYERS} erb_layers={ERB_LAYERS} df_layers={DF_LAYERS})…");
    let t0 = Instant::now();
    let mut proc = streaming::StreamingProcessor::new(
        &enc_bytes,
        &erb_dec_bytes,
        &df_dec_bytes,
        erb_fb,
        erb_inv_fb,
        ENC_LAYERS,
        ERB_LAYERS,
        DF_LAYERS,
    )
    .unwrap();
    println!("   Time: {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);
    println!();

    // 4. Warmup
    println!("4. Warmup (10 frames)…");
    let audio = Array1::zeros(480);
    for _ in 0..10 {
        let _ = proc.process_frame(audio.view());
    }
    println!("   Done");
    println!();

    // 5. Benchmark
    println!("5. Benchmark (1000 frames)…");
    let mut times = Vec::new();

    for i in 0..1000 {
        let t = Instant::now();
        let _ = proc.process_frame(audio.view()).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);

        if (i + 1) % 250 == 0 {
            println!("   {} / 1000", i + 1);
        }
    }
    println!();

    // 6. Stats
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[500];
    let p95 = times[950];
    let rtf = 10.0 / mean;

    println!("{}", sep);
    println!("RESULTS");
    println!("{}", sep);
    println!("Mean:    {:.3} ms", mean);
    println!("Median:  {:.3} ms", median);
    println!("P95:     {:.3} ms", p95);
    println!("RTF:     {:.1}x", rtf);
    println!();

    let dfn3 = 2.5;
    println!(
        "vs DFN3: {:.2}x {}",
        dfn3 / mean,
        if mean < dfn3 { "faster" } else { "slower" }
    );
}
