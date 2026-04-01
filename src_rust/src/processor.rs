//! LightDFN WASM Processor - Matching DFN3's architecture exactly
//!
//! Uses tract-pulse TypedSimpleState with manual state management.

use crate::erb::{ErbFilterbank, NB_ERB, NB_FREQS};
use crate::stft::{Complex32, StreamingStft, HOP_SIZE};
use wasm_bindgen::prelude::*;

use tract_onnx::prelude::*;
#[allow(unused_imports)]
use tract_nnef::prelude::*;

// ── Constants ─────────────────────────────────────────────────────────────────
const NB_SPEC: usize = 96;
const ALPHA: f32 = 0.990_049_834_f32;
const ONE_MINUS_ALPHA: f32 = 1.0 - ALPHA;
const GRU_RESET_FRAMES: u32 = 512;

// ── Output indices ────────────────────────────────────────────────────────────
const OUT_ENHANCED_SPEC: usize = 0;
const OUT_H_ENC: usize = 1;
const OUT_H_ERB: usize = 2;
const OUT_H_DF: usize = 3;
const OUT_BUF_ERB0: usize = 4;
const OUT_BUF_DF0: usize = 5;
const OUT_BUF_DFP: usize = 6;
const OUT_BUF_SPEC: usize = 7;

// Use TypedSimpleState like DFN3
pub type TractModel = TypedSimpleState<TypedModel, TypedSimplePlan<TypedModel>>;

/// LightDFN Processor matching DFN3's architecture
#[wasm_bindgen]
pub struct LightDFNProcessor {
    model: TractModel,
    stft: StreamingStft,
    erb_fb: ErbFilterbank,

    // Normalization states
    erb_norm: [f32; NB_ERB],
    unit_norm: [f32; NB_SPEC],

    // State tensors (pre-allocated, reused each frame)
    h_enc: Tensor,
    h_erb: Tensor,
    h_df: Tensor,
    buf_erb0: Tensor,
    buf_df0: Tensor,
    buf_dfp: Tensor,
    buf_spec: Tensor,

    // Scratch buffers
    power: [f32; NB_FREQS],
    erb_power: [f32; NB_ERB],

    frame_count: u32,
}

#[wasm_bindgen]
impl LightDFNProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(onnx_bytes: &[u8]) -> Result<LightDFNProcessor, JsValue> {
        console_error_panic_hook::set_once();

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Loading model...".into());

        let model = Self::load_model(onnx_bytes).map_err(|e| {
            let err_msg = format!("Failed to load model: {:?}", e);
            #[cfg(target_arch = "wasm32")]
            web_sys::console::error_1(&err_msg.clone().into());
            JsValue::from_str(&err_msg)
        })?;

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Model loaded, allocating tensors...".into());

        // Pre-allocate all state tensors
        let h_enc = Tensor::zero::<f32>(&[1, 1, 256])
            .map_err(|e| JsValue::from_str(&format!("Failed to allocate h_enc: {}", e)))?;
        let h_erb = Tensor::zero::<f32>(&[2, 1, 256])
            .map_err(|e| JsValue::from_str(&format!("Failed to allocate h_erb: {}", e)))?;
        let h_df = Tensor::zero::<f32>(&[2, 1, 256])
            .map_err(|e| JsValue::from_str(&format!("Failed to allocate h_df: {}", e)))?;
        let buf_erb0 = Tensor::zero::<f32>(&[1, 1, 2, 32])
            .map_err(|e| JsValue::from_str(&format!("Failed to allocate buf_erb0: {}", e)))?;
        let buf_df0 = Tensor::zero::<f32>(&[1, 2, 2, 96])
            .map_err(|e| JsValue::from_str(&format!("Failed to allocate buf_df0: {}", e)))?;
        let buf_dfp = Tensor::zero::<f32>(&[1, 64, 4, 96])
            .map_err(|e| JsValue::from_str(&format!("Failed to allocate buf_dfp: {}", e)))?;
        let buf_spec = Tensor::zero::<f32>(&[1, 1, 2, 481, 2])
            .map_err(|e| JsValue::from_str(&format!("Failed to allocate buf_spec: {}", e)))?;

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"All tensors allocated successfully!".into());

        Ok(LightDFNProcessor {
            model,
            stft: StreamingStft::new(),
            erb_fb: ErbFilterbank::new(),
            erb_norm: init_erb_norm(),
            unit_norm: init_unit_norm(),
            h_enc,
            h_erb,
            h_df,
            buf_erb0,
            buf_df0,
            buf_dfp,
            buf_spec,
            power: [0.0; NB_FREQS],
            erb_power: [0.0; NB_ERB],
            frame_count: 0,
        })
    }

    pub fn process_frame(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), HOP_SIZE);

        // Reset GRU states every 512 frames (like DFN3)
        if self.frame_count > 0 && self.frame_count % GRU_RESET_FRAMES == 0 {
            self.h_enc.as_slice_mut::<f32>().unwrap().fill(0.0);
            self.h_erb.as_slice_mut::<f32>().unwrap().fill(0.0);
            self.h_df.as_slice_mut::<f32>().unwrap().fill(0.0);
        }
        self.frame_count += 1;

        // ── STFT ──────────────────────────────────────────────────────────
        let hop_arr: &[f32; HOP_SIZE] = input.try_into().unwrap();
        let spectrum: &[Complex32; NB_FREQS] = self.stft.forward(hop_arr);

        // ── Build input tensors ───────────────────────────────────────────

        // Spec tensor [1, 1, 1, 481, 2]
        let mut spec_data = vec![0.0f32; NB_FREQS * 2];
        for (f, c) in spectrum.iter().enumerate() {
            spec_data[f * 2] = c.re;
            spec_data[f * 2 + 1] = c.im;
            self.power[f] = c.re * c.re + c.im * c.im;
        }
        let spec_tensor = Tensor::from_shape(&[1, 1, 1, NB_FREQS, 2], &spec_data)
            .expect("Failed to create spec tensor");

        // ERB features [1, 1, 1, 32]
        self.erb_fb.apply(&self.power, &mut self.erb_power);
        let mut erb_feat = vec![0.0f32; NB_ERB];
        for b in 0..NB_ERB {
            let erb_db = (self.erb_power[b] + 1e-10).log10() * 10.0;
            let new_state = erb_db * ONE_MINUS_ALPHA + self.erb_norm[b] * ALPHA;
            self.erb_norm[b] = new_state;
            erb_feat[b] = (erb_db - new_state) / 40.0;
        }
        let feat_erb_tensor = Tensor::from_shape(&[1, 1, 1, NB_ERB], &erb_feat)
            .expect("Failed to create feat_erb tensor");

        // Spec features [1, 1, 1, 96, 2]
        let mut spec_feat = vec![0.0f32; NB_SPEC * 2];
        for f in 0..NB_SPEC {
            let re = spectrum[f].re;
            let im = spectrum[f].im;
            let abs = (re * re + im * im).sqrt();
            let new_state = abs * ONE_MINUS_ALPHA + self.unit_norm[f] * ALPHA;
            self.unit_norm[f] = new_state;
            let inv_sqrt = 1.0 / (new_state + 1e-10).sqrt();
            spec_feat[f * 2] = re * inv_sqrt;
            spec_feat[f * 2 + 1] = im * inv_sqrt;
        }
        let feat_spec_tensor = Tensor::from_shape(&[1, 1, 1, NB_SPEC, 2], &spec_feat)
            .expect("Failed to create feat_spec tensor");

        // ── Run model with all 10 inputs ──────────────────────────────────
        let inputs = tvec![
            spec_tensor.into(),
            feat_erb_tensor.into(),
            feat_spec_tensor.into(),
            self.h_enc.clone().into(),
            self.h_erb.clone().into(),
            self.h_df.clone().into(),
            self.buf_erb0.clone().into(),
            self.buf_df0.clone().into(),
            self.buf_dfp.clone().into(),
            self.buf_spec.clone().into(),
        ];

        let outputs = self.model.run(inputs).expect("Model inference failed");

        // ── Extract outputs and update states ─────────────────────────────

        // Enhanced spectrum
        let enh_tensor = &outputs[OUT_ENHANCED_SPEC];
        let enh_view = enh_tensor.to_array_view::<f32>().unwrap();
        let enh_data = enh_view.as_slice().unwrap();

        // Update state tensors for next frame
        copy_tensor(&outputs[OUT_H_ENC], &mut self.h_enc);
        copy_tensor(&outputs[OUT_H_ERB], &mut self.h_erb);
        copy_tensor(&outputs[OUT_H_DF], &mut self.h_df);
        copy_tensor(&outputs[OUT_BUF_ERB0], &mut self.buf_erb0);
        copy_tensor(&outputs[OUT_BUF_DF0], &mut self.buf_df0);
        copy_tensor(&outputs[OUT_BUF_DFP], &mut self.buf_dfp);
        copy_tensor(&outputs[OUT_BUF_SPEC], &mut self.buf_spec);

        // ── ISTFT ─────────────────────────────────────────────────────────
        let mut enh_spec = [Complex32::default(); NB_FREQS];
        for (f, c) in enh_spec.iter_mut().enumerate() {
            c.re = enh_data[f * 2];
            c.im = enh_data[f * 2 + 1];
        }
        enh_spec[0].im = 0.0;
        enh_spec[NB_FREQS - 1].im = 0.0;

        let mut output = vec![0.0f32; HOP_SIZE];
        let out_arr: &mut [f32; HOP_SIZE] = output.as_mut_slice().try_into().unwrap();
        self.stft.inverse(&enh_spec, out_arr);

        output
    }

    pub fn reset(&mut self) {
        self.stft.reset();
        self.erb_norm = init_erb_norm();
        self.unit_norm = init_unit_norm();

        // Reset all state tensors
        self.h_enc.as_slice_mut::<f32>().unwrap().fill(0.0);
        self.h_erb.as_slice_mut::<f32>().unwrap().fill(0.0);
        self.h_df.as_slice_mut::<f32>().unwrap().fill(0.0);
        self.buf_erb0.as_slice_mut::<f32>().unwrap().fill(0.0);
        self.buf_df0.as_slice_mut::<f32>().unwrap().fill(0.0);
        self.buf_dfp.as_slice_mut::<f32>().unwrap().fill(0.0);
        self.buf_spec.as_slice_mut::<f32>().unwrap().fill(0.0);

        self.frame_count = 0;
    }

    pub fn sample_rate(&self) -> u32 {
        48_000
    }
}

impl LightDFNProcessor {
    /// Load a pre-typed NNEF model from bytes (no into_typed() at runtime).
    /// WASM uses this path — avoids the recursive call-stack overflow that
    /// tract's ONNX graph type-inference triggers in the browser JS engine.
    fn load_model(bytes: &[u8]) -> TractResult<TractModel> {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Loading NNEF model...".into());

        let mut cursor = std::io::Cursor::new(bytes);
        let typed_model = tract_nnef::nnef()
            .with_tract_core()
            .model_for_read(&mut cursor)?;

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Creating runnable plan...".into());

        let plan = typed_model.into_runnable()?;

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Model ready!".into());

        SimpleState::new(plan)
    }

    /// Load model from ONNX path — native only, for tests and conversion.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_model_from_onnx_path(path: &str) -> TractResult<TractModel> {
        use tract_onnx::prelude::InferenceModelExt;
        let typed_model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_typed()?;
        let plan = typed_model.into_runnable()?;
        SimpleState::new(plan)
    }

    /// Load model from a pre-typed NNEF file path — native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_model_from_nnef_path(path: &str) -> TractResult<TractModel> {
        let typed_model = tract_nnef::nnef()
            .with_tract_core()
            .model_for_path(path)?;
        let plan = typed_model.into_runnable()?;
        SimpleState::new(plan)
    }
}

/// Copy tensor data from source to destination
#[inline]
fn copy_tensor(src: &TValue, dst: &mut Tensor) {
    let src_view = src.to_array_view::<f32>().unwrap();
    let src_slice = src_view.as_slice().unwrap();
    let dst_slice = dst.as_slice_mut::<f32>().unwrap();
    dst_slice.copy_from_slice(src_slice);
}

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
        let log_val = -3.0 - (i as f32 / (NB_SPEC - 1) as f32);
        *v = 10f32.powf(log_val);
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::HOP_SIZE;

    fn make_processor(model: TractModel) -> LightDFNProcessor {
        LightDFNProcessor {
            model,
            stft: crate::stft::StreamingStft::new(),
            erb_fb: crate::erb::ErbFilterbank::new(),
            erb_norm: init_erb_norm(),
            unit_norm: init_unit_norm(),
            h_enc: Tensor::zero::<f32>(&[1, 1, 256]).unwrap(),
            h_erb: Tensor::zero::<f32>(&[2, 1, 256]).unwrap(),
            h_df: Tensor::zero::<f32>(&[2, 1, 256]).unwrap(),
            buf_erb0: Tensor::zero::<f32>(&[1, 1, 2, 32]).unwrap(),
            buf_df0: Tensor::zero::<f32>(&[1, 2, 2, 96]).unwrap(),
            buf_dfp: Tensor::zero::<f32>(&[1, 64, 4, 96]).unwrap(),
            buf_spec: Tensor::zero::<f32>(&[1, 1, 2, 481, 2]).unwrap(),
            power: [0.0; NB_FREQS],
            erb_power: [0.0; NB_ERB],
            frame_count: 0,
        }
    }

    /// Load from ONNX and run one silent frame end-to-end.
    #[test]
    fn test_processor_from_onnx() {
        let model = LightDFNProcessor::load_model_from_onnx_path(
            "../checkpoints/streaming_best_inline.onnx",
        )
        .expect("load_model_from_onnx_path failed");

        let mut processor = make_processor(model);
        let silent = vec![0.0f32; HOP_SIZE];
        let out = processor.process_frame(&silent);
        assert_eq!(out.len(), HOP_SIZE);
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        println!("test_processor_from_onnx OK — RMS: {rms:.6}");
    }

    /// Load from pre-typed NNEF archive and run one silent frame.
    /// Run `cargo run --bin convert_nnef --release` first to generate the archive.
    #[test]
    fn test_processor_from_nnef() {
        let nnef_path = "../checkpoints/streaming_best.nnef.tar";
        if !std::path::Path::new(nnef_path).exists() {
            println!("SKIP: {nnef_path} not found — run `cargo run --bin convert_nnef --release`");
            return;
        }
        let model = LightDFNProcessor::load_model_from_nnef_path(nnef_path)
            .expect("load_model_from_nnef_path failed");

        let mut processor = make_processor(model);
        let silent = vec![0.0f32; HOP_SIZE];
        let out = processor.process_frame(&silent);
        assert_eq!(out.len(), HOP_SIZE);
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        println!("test_processor_from_nnef OK — RMS: {rms:.6}");
    }
}


