//! LightDFN DSP Module - STFT, ERB, and feature extraction only
//!
//! This module handles audio signal processing (DSP) in Rust/WASM.
//! ONNX model inference is handled separately in JavaScript with ONNX Runtime Web.
//!
//! Architecture:
//! - Rust/WASM: STFT → ERB features → Spectral features (fast DSP)
//! - JavaScript: ONNX Runtime Web inference (runs in Web Worker)
//! - Rust/WASM: iSTFT ← Apply enhanced spectrum (post-processing)
//!
//! Feature normalisation exactly matches the Python training pipeline in
//! `src/dataloader/fft.py` (_running_mean_norm_erb / _running_unit_norm).

use crate::erb::{ErbFilterbank, NB_ERB, NB_FREQS};
use crate::stft::{Complex32, StreamingStft, HOP_SIZE};
use wasm_bindgen::prelude::*;

const NB_SPEC: usize = 96;
/// α = exp(−1 / (frames_per_sec × τ)) = exp(−1 / (100 × 1)) ≈ 0.99004983
const ALPHA: f32 = 0.990_049_834_f32;
const ONE_MINUS_ALPHA: f32 = 1.0 - ALPHA;

// ── Initialisation helpers ──────────────────────────────────────────────────

/// Initialise ERB running-mean state from −60 dB (band 0) to −90 dB (band E−1).
/// Matches Python: `torch.linspace(−60., −90., nb_erb)`.
fn init_erb_norm() -> [f32; NB_ERB] {
    let mut arr = [0.0f32; NB_ERB];
    for (i, v) in arr.iter_mut().enumerate() {
        *v = -60.0 - 30.0 * (i as f32) / (NB_ERB - 1) as f32;
    }
    arr
}

/// Initialise per-bin unit-norm state from 0.001 (bin 0) to 0.0001 (bin F−1).
/// Matches Python: `torch.linspace(0.001, 0.0001, nb_spec)`.
fn init_unit_norm() -> [f32; NB_SPEC] {
    let mut arr = [0.0f32; NB_SPEC];
    for (i, v) in arr.iter_mut().enumerate() {
        let exp = -3.0 - (i as f32) / (NB_SPEC - 1) as f32;
        *v = 10f32.powf(exp);
    }
    arr
}

// ── Exported WASM struct ────────────────────────────────────────────────────

/// DSP processor for LightDFN - handles STFT and feature extraction.
#[wasm_bindgen]
pub struct LightDFNDSP {
    stft: StreamingStft,
    erb_fb: ErbFilterbank,

    /// Running-mean state in dB per ERB band (matches Python erb_state).
    erb_norm: [f32; NB_ERB],
    /// Running amplitude-scale state per spectral bin (matches Python unit_state).
    unit_norm: [f32; NB_SPEC],

    // Scratch buffers – reused every frame
    power: [f32; NB_FREQS],
    erb_power: [f32; NB_ERB],
}

#[wasm_bindgen]
impl LightDFNDSP {
    /// Create a new DSP processor.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<LightDFNDSP, JsValue> {
        console_error_panic_hook::set_once();
        Ok(Self {
            stft: StreamingStft::new(),
            erb_fb: ErbFilterbank::new(),
            erb_norm: init_erb_norm(),
            unit_norm: init_unit_norm(),
            power: [0.0; NB_FREQS],
            erb_power: [0.0; NB_ERB],
        })
    }

    /// Process one audio frame: compute STFT and extract model features.
    ///
    /// **Input:** 480 audio samples (10 ms at 48 kHz).
    ///
    /// **Output:** flat `Float32Array` of 1186 values packed as:
    /// ```text
    /// spec_real [481]  — raw spectrum real parts  (for model input `spec`)
    /// spec_imag [481]  — raw spectrum imag parts
    /// feat_erb  [ 32]  — ERB features (running-mean normalised, ÷40)
    /// feat_spec_re [96] — unit-normalised complex spectrum, real parts
    /// feat_spec_im [96] — unit-normalised complex spectrum, imag parts
    /// ```
    /// The JavaScript caller must interleave spec and feat_spec into shape
    /// `[1,1,1,F,2]` / `[1,1,1,96,2]` before feeding the ONNX model.
    #[wasm_bindgen]
    pub fn extract_features(&mut self, audio_frame: &[f32]) -> Result<Vec<f32>, JsValue> {
        if audio_frame.len() != HOP_SIZE {
            return Err(JsValue::from_str(&format!(
                "Expected {} samples, got {}",
                HOP_SIZE,
                audio_frame.len()
            )));
        }

        // 1. STFT
        let hop: &[f32; HOP_SIZE] = audio_frame
            .try_into()
            .map_err(|_| JsValue::from_str("Invalid frame size"))?;
        let spectrum = self.stft.forward(hop);

        // 2. Power spectrum
        for (i, &c) in spectrum.iter().enumerate() {
            self.power[i] = c.re * c.re + c.im * c.im;
        }

        // 3. ERB features — running-mean normalisation in dB domain
        //    Matches Python _running_mean_norm_erb in fft.py:
        //      state = x*(1−α) + state*α
        //      out   = (x − state) / 40
        self.erb_fb.apply(&self.power, &mut self.erb_power);
        for i in 0..NB_ERB {
            let erb_db = (self.erb_power[i] + 1e-10).log10() * 10.0;
            let new_state = erb_db * ONE_MINUS_ALPHA + self.erb_norm[i] * ALPHA;
            self.erb_norm[i] = new_state;
            self.erb_power[i] = (erb_db - new_state) / 40.0;
        }

        // 4. Spectral features — per-frequency unit normalisation
        //    Matches Python _running_unit_norm in fft.py:
        //      state = (|x|*(1−α) + state*α).clamp(1e-10)
        //      out   = x / sqrt(state)
        let mut spec_feat_re = [0.0f32; NB_SPEC];
        let mut spec_feat_im = [0.0f32; NB_SPEC];
        for i in 0..NB_SPEC {
            let abs_val = self.power[i].sqrt();
            let new_state = (abs_val * ONE_MINUS_ALPHA + self.unit_norm[i] * ALPHA).max(1e-10);
            self.unit_norm[i] = new_state;
            let inv_sqrt = 1.0 / new_state.sqrt();
            spec_feat_re[i] = spectrum[i].re * inv_sqrt;
            spec_feat_im[i] = spectrum[i].im * inv_sqrt;
        }

        // 5. Pack output (planar layout; JS caller interleaves for ONNX tensors)
        let mut output = Vec::with_capacity(NB_FREQS + NB_FREQS + NB_ERB + NB_SPEC + NB_SPEC);

        for c in spectrum.iter() {
            output.push(c.re);
        } // spec_real [481]
        for c in spectrum.iter() {
            output.push(c.im);
        } // spec_imag [481]
        output.extend_from_slice(&self.erb_power); // feat_erb  [ 32]
        output.extend_from_slice(&spec_feat_re); // feat_spec_re [96]
        output.extend_from_slice(&spec_feat_im); // feat_spec_im [96]

        Ok(output)
    }

    /// Apply enhanced spectrum and reconstruct audio via iSTFT.
    ///
    /// **Input:** flat `Float32Array` of 962 values: `[real[481], imag[481]]`.
    /// **Output:** 480 audio samples (10 ms at 48 kHz).
    #[wasm_bindgen]
    pub fn apply_enhancement(&mut self, enhanced_spec: &[f32]) -> Result<Vec<f32>, JsValue> {
        if enhanced_spec.len() != NB_FREQS * 2 {
            return Err(JsValue::from_str(&format!(
                "Expected {} floats, got {}",
                NB_FREQS * 2,
                enhanced_spec.len()
            )));
        }

        // The enhanced_spec from the ONNX model is interleaved [re0,im0, re1,im1, ...]
        // because the model output tensor has shape [1,1,1,481,2] in C-order.
        let mut spectrum = [Complex32::new(0.0, 0.0); NB_FREQS];
        for i in 0..NB_FREQS {
            spectrum[i] = Complex32::new(enhanced_spec[i * 2], enhanced_spec[i * 2 + 1]);
        }

        // Zero DC and Nyquist imaginary parts (conjugate symmetry)
        spectrum[0].im = 0.0;
        spectrum[NB_FREQS - 1].im = 0.0;

        let mut output = [0.0f32; HOP_SIZE];
        self.stft.inverse(&spectrum, &mut output);
        Ok(output.to_vec())
    }

    /// Reset all internal state (normalisers + STFT buffers).
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.stft = StreamingStft::new();
        self.erb_norm = init_erb_norm();
        self.unit_norm = init_unit_norm();
        self.power.fill(0.0);
        self.erb_power.fill(0.0);
    }

    /// Sample rate (always 48 000 Hz).
    #[wasm_bindgen]
    pub fn sample_rate(&self) -> u32 {
        48_000
    }

    /// Hop size — number of samples expected per `extract_features` call.
    #[wasm_bindgen]
    pub fn hop_size(&self) -> usize {
        HOP_SIZE
    }
}

impl Default for LightDFNDSP {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
