use anyhow::{bail, Result};
use std::io::Cursor;
use std::path::Path;
use tract_onnx::prelude::TVec;
use tract_onnx::prelude::*;
pub use tract_pulse::model::PulsedModel;
pub use tract_pulse::model::PulsedModelExt;

use crate::erb::{ErbFilterbank, NB_ERB, NB_FREQS};
use crate::stft::{StreamingStft, HOP_SIZE};
pub type Complex32 = num_complex::Complex32;

// ── Constants ─────────────────────────────────────────────────────────────────
const NB_SPEC: usize = 96;
const ALPHA: f32 = 0.990_049_834_f32;
const ONE_MINUS_ALPHA: f32 = 1.0 - ALPHA;

pub type TractModel = TypedSimpleState<TypedModel, TypedSimplePlan<TypedModel>>;

/// Core Processor (Pure Rust, Native compatible)
pub struct LightDFNProcessor {
    // Split models (Pulsed + Optimized)
    enc: TractModel,
    erb_dec: TractModel,
    df_dec: TractModel,

    // Internal components for DSP
    stft: StreamingStft,
    erb_fb: ErbFilterbank,
    erb_norm: [f32; NB_ERB],
    unit_norm: [f32; NB_SPEC],

    // Pre-allocated TValue buffers for calling models
    erb_in: Tensor,
    spec_in: Tensor,
}

impl LightDFNProcessor {
    pub fn from_onnx_bytes(
        enc_bytes: &[u8],
        erb_dec_bytes: &[u8],
        df_dec_bytes: &[u8],
    ) -> Result<Self> {
        let enc_probe = tract_onnx::onnx().model_for_read(&mut Cursor::new(enc_bytes))?;
        let s = enc_probe.symbols.sym("S");
        let s_dim = s.to_dim();

        let enc = Self::load_pulsed_model_from_bytes(
            enc_bytes,
            "S",
            &["feat_erb", "feat_spec"],
            &["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
            vec![
                tvec![1.to_dim(), 1.to_dim(), s_dim.clone(), NB_ERB.into()],
                tvec![1.to_dim(), 2.to_dim(), s_dim.clone(), NB_SPEC.into()],
            ],
        )?;

        let erb_dec = Self::load_pulsed_model_from_bytes(
            erb_dec_bytes,
            "S",
            &["emb", "e3", "e2", "e1", "e0"],
            &["mask"],
            vec![
                tvec![1.to_dim(), 512.to_dim(), s_dim.clone(), 1.to_dim()],
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 8.to_dim()],
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 8.to_dim()],
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 16.to_dim()],
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 32.to_dim()],
            ],
        )?;

        let df_dec = Self::load_pulsed_model_from_bytes(
            df_dec_bytes,
            "S",
            &["emb", "c0"],
            &["coefs"],
            vec![
                tvec![1.to_dim(), 512.to_dim(), s_dim.clone(), 1.to_dim()],
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 96.to_dim()],
            ],
        )?;

        Ok(Self {
            enc,
            erb_dec,
            df_dec,
            stft: StreamingStft::new(),
            erb_fb: ErbFilterbank::new(),
            erb_norm: init_erb_norm(),
            unit_norm: init_unit_norm(),
            erb_in: Tensor::zero::<f32>(&[1, 1, 1, NB_ERB])?,
            spec_in: Tensor::zero::<f32>(&[1, 2, 1, NB_SPEC])?,
        })
    }

    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let dir = model_dir.as_ref();

        // Load one model briefly just to get the symbol S
        let enc_model = tract_onnx::onnx().model_for_path(dir.join("enc.onnx"))?;
        let s = enc_model.symbols.sym("S");
        let s_dim = s.to_dim();
        let enc = Self::load_pulsed_model(
            dir.join("enc.onnx"),
            "S",
            &["feat_erb", "feat_spec"],
            &["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
            vec![
                tvec![1.to_dim(), 1.to_dim(), s_dim.clone(), NB_ERB.into()],
                tvec![1.to_dim(), 2.to_dim(), s_dim.clone(), NB_SPEC.into()],
            ],
        )?;
        let erb_dec = Self::load_pulsed_model(
            dir.join("erb_dec.onnx"),
            "S",
            &["emb", "e3", "e2", "e1", "e0"],
            &["mask"],
            vec![
                tvec![1.to_dim(), 512.to_dim(), s_dim.clone(), 1.to_dim()], // emb  (4D: B, H, S, 1)
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 8.to_dim()],  // e3   (4D: B, C, S, F)
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 8.to_dim()],  // e2
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 16.to_dim()], // e1
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 32.to_dim()], // e0
            ],
        )?;
        let df_dec = Self::load_pulsed_model(
            dir.join("df_dec.onnx"),
            "S",
            &["emb", "c0"],
            &["coefs"],
            vec![
                tvec![1.to_dim(), 512.to_dim(), s_dim.clone(), 1.to_dim()], // emb  (4D)
                tvec![1.to_dim(), 64.to_dim(), s_dim.clone(), 96.to_dim()], // c0   (4D)
            ],
        )?;

        // 2. Initialize DSP components
        Ok(Self {
            enc,
            erb_dec,
            df_dec,
            stft: StreamingStft::new(),
            erb_fb: ErbFilterbank::new(),
            erb_norm: init_erb_norm(),
            unit_norm: init_unit_norm(),
            erb_in: Tensor::zero::<f32>(&[1, 1, 1, NB_ERB])?,
            spec_in: Tensor::zero::<f32>(&[1, 2, 1, NB_SPEC])?,
        })
    }

    fn load_pulsed_model<P: AsRef<Path>>(
        path: P,
        symbol: &str,
        input_names: &[&str],
        output_names: &[&str],
        input_shapes: Vec<TVec<TDim>>,
    ) -> Result<TractModel> {
        let mut model = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .model_for_path(path)?;

        let s = model.symbols.sym(symbol);

        // Pin shapes explicitly
        model.set_input_names(input_names)?;
        for (i, shape) in input_shapes.into_iter().enumerate() {
            model.set_input_fact(i, f32::datum_type().fact(shape).into())?;
        }
        model.set_output_names(output_names)?;

        // DeepFilterNet way: typed -> declutter -> pulse
        let mut typed = model.into_typed()?;
        typed.declutter()?;

        // Pulse -> Optimized flow
        let pulsed = PulsedModel::new(&typed, s, &1.to_dim())?;

        let optimized = pulsed.into_typed()?.into_optimized()?;

        let runnable = optimized.into_runnable()?;
        SimpleState::new(runnable)
    }

    fn load_pulsed_model_from_bytes(
        bytes: &[u8],
        symbol: &str,
        input_names: &[&str],
        output_names: &[&str],
        input_shapes: Vec<TVec<TDim>>,
    ) -> Result<TractModel> {
        let mut model = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .model_for_read(&mut Cursor::new(bytes))?;

        let s = model.symbols.sym(symbol);

        model.set_input_names(input_names)?;
        for (i, shape) in input_shapes.into_iter().enumerate() {
            model.set_input_fact(i, f32::datum_type().fact(shape).into())?;
        }
        model.set_output_names(output_names)?;

        let mut typed = model.into_typed()?;
        typed.declutter()?;

        let pulsed = PulsedModel::new(&typed, s, &1.to_dim())?;
        let optimized = pulsed.into_typed()?.into_optimized()?;
        let runnable = optimized.into_runnable()?;
        SimpleState::new(runnable)
    }

    pub fn process_frame(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != HOP_SIZE {
            bail!("Invalid input length");
        }

        let hop_arr: &[f32; HOP_SIZE] = input.try_into().unwrap();
        let spectrum = *self.stft.forward(hop_arr); // Dereference/Copy to avoid borrow issues

        // 1. Feature Extraction (Into internal buffers)
        self.update_features(&spectrum);

        // 2. Run Encoder
        let inputs: TVec<TValue> =
            vec![self.erb_in.clone().into(), self.spec_in.clone().into()].into();
        let mut enc_out = self.enc.run(inputs)?;

        // Order: ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]
        let _lsnr = enc_out.pop().unwrap();
        let c0 = enc_out.pop().unwrap();
        let emb = enc_out.pop().unwrap();
        let e3 = enc_out.pop().unwrap();
        let e2 = enc_out.pop().unwrap();
        let e1 = enc_out.pop().unwrap();
        let e0 = enc_out.pop().unwrap();

        // 3. Run ERB Decoder
        let inputs: TVec<TValue> = vec![
            emb.clone().into(),
            e3.into(),
            e2.into(),
            e1.into(),
            e0.into(),
        ]
        .into();
        let mut erb_out = self.erb_dec.run(inputs)?;
        let mask = erb_out.pop().unwrap().into_tensor();
        let mask_vals = mask.as_slice::<f32>()?;
        if mask_vals.len() < NB_ERB {
            bail!("Unexpected ERB mask size: {}", mask_vals.len());
        }

        // 4. Run DF Decoder
        let inputs: TVec<TValue> = vec![emb.into(), c0.into()].into();
        let mut df_out = self.df_dec.run(inputs)?;
        let _coefs = df_out.pop().unwrap();

        // 5. Apply ERB mask on the complex spectrum.
        // Each FFT bin maps to exactly one ERB band in this configuration.
        let mut enhanced_spec = spectrum;
        for f in 0..NB_FREQS {
            let b = self.erb_fb.band[f] as usize;
            let g = mask_vals[b];
            enhanced_spec[f].re *= g;
            enhanced_spec[f].im *= g;
        }

        // 6. Synthesis
        let mut output = vec![0.0f32; HOP_SIZE];
        self.stft
            .inverse(&enhanced_spec, output.as_mut_slice().try_into().unwrap());

        Ok(output)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.stft = StreamingStft::new();
        self.erb_norm = init_erb_norm();
        self.unit_norm = init_unit_norm();
        self.erb_in = Tensor::zero::<f32>(&[1, 1, 1, NB_ERB])?;
        self.spec_in = Tensor::zero::<f32>(&[1, 2, 1, NB_SPEC])?;
        Ok(())
    }

    fn update_features(&mut self, spectrum: &[Complex32; NB_FREQS]) {
        let mut erb_feat = vec![0.0f32; NB_ERB];
        let mut power = [0.0f32; NB_FREQS];
        for f in 0..NB_FREQS {
            power[f] = spectrum[f].re * spectrum[f].re + spectrum[f].im * spectrum[f].im;
        }
        let mut erb_power = [0.0f32; NB_ERB];
        self.erb_fb.apply(&power, &mut erb_power);

        for b in 0..NB_ERB {
            let erb_db = (erb_power[b] + 1e-10).log10() * 10.0;
            let ns = erb_db * ONE_MINUS_ALPHA + self.erb_norm[b] * ALPHA;
            self.erb_norm[b] = ns;
            erb_feat[b] = (erb_db - ns) / 40.0;
        }
        self.erb_in = tract_ndarray::Array4::from_shape_vec((1, 1, 1, NB_ERB), erb_feat)
            .unwrap()
            .into_tensor();

        let mut spec_feat = vec![0.0f32; NB_SPEC * 2];
        for f in 0..NB_SPEC {
            let abs = (spectrum[f].re * spectrum[f].re + spectrum[f].im * spectrum[f].im).sqrt();
            let ns = abs * ONE_MINUS_ALPHA + self.unit_norm[f] * ALPHA;
            self.unit_norm[f] = ns;
            let inv_sqrt = 1.0 / (ns + 1e-10).sqrt();
            // Layout must be planar for [B, 2, T, F]:
            // channel 0 = real[0..F), channel 1 = imag[0..F).
            spec_feat[f] = spectrum[f].re * inv_sqrt;
            spec_feat[NB_SPEC + f] = spectrum[f].im * inv_sqrt;
        }
        self.spec_in = tract_ndarray::Array4::from_shape_vec((1, 2, 1, NB_SPEC), spec_feat)
            .unwrap()
            .into_tensor();
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn init_erb_norm() -> [f32; NB_ERB] {
    let mut arr = [0.0; NB_ERB];
    for (i, v) in arr.iter_mut().enumerate() {
        *v = -60.0 - (30.0 * i as f32) / (NB_ERB - 1) as f32;
    }
    arr
}

fn init_unit_norm() -> [f32; NB_SPEC] {
    let mut arr = [0.0; NB_SPEC];
    for (i, v) in arr.iter_mut().enumerate() {
        *v = 10f32.powf(-3.0 - (i as f32 / (NB_SPEC - 1) as f32));
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_native_perf() {
        // Points to our new Split Models directory (relative to src_rust/)
        let model_dir = "../checkpoints/dfn_style";
        if !std::path::Path::new(model_dir).exists() {
            println!("Skipping test: model directory {} not found", model_dir);
            return;
        }

        let mut processor = LightDFNProcessor::new(model_dir).expect("Failed to init processor");

        println!("\nPerformance Benchmark (Native, Release) - DFN-Style Pulsed Architecture:");

        // Warmup
        for _ in 0..10 {
            let _ = processor.process_frame(&[0.0; HOP_SIZE]);
        }

        let niters = 2000;
        let mut t_stft = 0.0f64;
        let mut t_feat = 0.0f64;
        let mut t_inf = 0.0f64;
        let mut t_istft = 0.0f64;

        let hop = [0.0f32; HOP_SIZE];
        for _ in 0..niters {
            let now = std::time::Instant::now();
            let spectrum = *processor.stft.forward(&hop);
            t_stft += now.elapsed().as_secs_f64() * 1000.0;

            let now = std::time::Instant::now();
            processor.update_features(&spectrum);
            t_feat += now.elapsed().as_secs_f64() * 1000.0;

            let now = std::time::Instant::now();
            let inputs: TVec<TValue> = vec![
                processor.erb_in.clone().into(),
                processor.spec_in.clone().into(),
            ]
            .into();
            let mut enc_out = processor.enc.run(inputs).unwrap();

            let _lsnr = enc_out.pop().unwrap();
            let c0 = enc_out.pop().unwrap();
            let emb = enc_out.pop().unwrap();
            let e3 = enc_out.pop().unwrap();
            let e2 = enc_out.pop().unwrap();
            let e1 = enc_out.pop().unwrap();
            let e0 = enc_out.pop().unwrap();

            let inputs: TVec<TValue> = vec![
                emb.clone().into(),
                e3.into(),
                e2.into(),
                e1.into(),
                e0.into(),
            ]
            .into();
            let mut erb_out = processor.erb_dec.run(inputs).unwrap();
            let _mask = erb_out.pop().unwrap();

            let inputs: TVec<TValue> = vec![emb.into(), c0.into()].into();
            let mut df_out = processor.df_dec.run(inputs).unwrap();
            let _coefs = df_out.pop().unwrap();
            t_inf += now.elapsed().as_secs_f64() * 1000.0;

            let now = std::time::Instant::now();
            let mut output = [0.0f32; HOP_SIZE];
            processor.stft.inverse(&spectrum, &mut output);
            t_istft += now.elapsed().as_secs_f64() * 1000.0;
        }

        println!("  - Avg. STFT Forward:  {:>8.3} ms", t_stft / niters as f64);
        println!("  - Avg. ERB Features:  {:>8.3} ms", t_feat / niters as f64);
        println!("  - Avg. Model Infr:    {:>8.3} ms", t_inf / niters as f64);
        println!(
            "  - Avg. STFT Inverse:  {:>8.3} ms",
            t_istft / niters as f64
        );

        let total_avg = (t_stft + t_feat + t_inf + t_istft) / niters as f64;
        println!("  - Total per frame:    {:>8.3} ms", total_avg);
        println!("  - Real-time factor:   {:>8.2}X", 10.0 / total_avg);
    }
}
