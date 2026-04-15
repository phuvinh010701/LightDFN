use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lightdfn_wasm::*;
use ndarray::Array1;
use std::time::Duration;

const ENC_LAYERS: usize = 1;
const ERB_LAYERS: usize = 2;
const DF_LAYERS: usize = 2;

fn load_erb_fb() -> (ndarray::Array2<f32>, ndarray::Array2<f32>) {
    let erb_json =
        std::fs::read_to_string("pkg/erb_filterbank.json").expect("Failed to load ERB JSON");
    let erb_data: serde_json::Value = serde_json::from_str(&erb_json).unwrap();

    let erb_fb_vec: Vec<Vec<f32>> =
        serde_json::from_value(erb_data["erb_fb"].clone()).unwrap();
    let n_freqs = erb_fb_vec.len();
    let nb_erb = erb_fb_vec[0].len();
    let mut erb_fb = ndarray::Array2::<f32>::zeros((n_freqs, nb_erb));
    for (i, row) in erb_fb_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            erb_fb[[i, j]] = val;
        }
    }

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

    (erb_fb, erb_inv_fb)
}

fn load_processor() -> streaming::StreamingProcessor {
    let enc_bytes =
        std::fs::read("pkg/enc.onnx").expect("pkg/enc.onnx not found");
    let erb_dec_bytes =
        std::fs::read("pkg/erb_dec.onnx").expect("pkg/erb_dec.onnx not found");
    let df_dec_bytes =
        std::fs::read("pkg/df_dec.onnx").expect("pkg/df_dec.onnx not found");
    let (erb_fb, erb_inv_fb) = load_erb_fb();
    streaming::StreamingProcessor::new(
        &enc_bytes,
        &erb_dec_bytes,
        &df_dec_bytes,
        erb_fb,
        erb_inv_fb,
        ENC_LAYERS,
        ERB_LAYERS,
        DF_LAYERS,
    )
    .expect("Failed to init StreamingProcessor")
}

fn benchmark_model_loading(c: &mut Criterion) {
    let enc_bytes =
        std::fs::read("pkg/enc.onnx").expect("pkg/enc.onnx not found");
    let erb_dec_bytes =
        std::fs::read("pkg/erb_dec.onnx").expect("pkg/erb_dec.onnx not found");
    let df_dec_bytes =
        std::fs::read("pkg/df_dec.onnx").expect("pkg/df_dec.onnx not found");
    let (erb_fb, erb_inv_fb) = load_erb_fb();

    c.bench_function("model_loading", |b| {
        b.iter(|| {
            streaming::StreamingProcessor::new(
                black_box(&enc_bytes),
                black_box(&erb_dec_bytes),
                black_box(&df_dec_bytes),
                black_box(erb_fb.clone()),
                black_box(erb_inv_fb.clone()),
                black_box(ENC_LAYERS),
                black_box(ERB_LAYERS),
                black_box(DF_LAYERS),
            )
            .unwrap()
        })
    });
}

fn benchmark_single_frame(c: &mut Criterion) {
    let mut processor = load_processor();
    let audio_frame = Array1::<f32>::zeros(480);

    c.bench_function("process_single_frame", |b| {
        b.iter(|| processor.process_frame(black_box(audio_frame.view())))
    });
}

fn benchmark_1000_frames(c: &mut Criterion) {
    let mut processor = load_processor();
    let audio_frame = Array1::<f32>::zeros(480);

    c.bench_function("process_1000_frames", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let _ = processor.process_frame(black_box(audio_frame.view()));
            }
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets = benchmark_model_loading, benchmark_single_frame, benchmark_1000_frames
);

criterion_main!(benches);
