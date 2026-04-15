use crate::features::FeatureExtractor;
use crate::model::{LightDFNModel, Result};
use ndarray::{Array1, Array2, ArrayView1};

/// High-level streaming processor — mirrors DFN3's DfTract.process().
pub struct StreamingProcessor {
    model: LightDFNModel,
    features: FeatureExtractor,
}

impl StreamingProcessor {
    /// Create from three ONNX model bytes (pulsed variant — GRU states as explicit tensor I/O),
    /// the ERB analysis filterbank, its inverse, and the layer counts.
    ///
    /// `erb_fb`:     shape [n_freqs, nb_erb] — forward ERB filterbank (for feature extraction).
    /// `erb_inv_fb`: shape [nb_erb, n_freqs] — inverse filterbank (for ERB→freq gain mapping).
    /// `enc_layers`, `erb_layers`, `df_layers`: Li-GRU layer counts (must match exported ONNX).
    pub fn new(
        enc_bytes: &[u8],
        erb_dec_bytes: &[u8],
        df_dec_bytes: &[u8],
        erb_fb: Array2<f32>,
        erb_inv_fb: Array2<f32>,
        enc_layers: usize,
        erb_layers: usize,
        df_layers: usize,
    ) -> Result<Self> {
        let model = LightDFNModel::from_bytes(
            enc_bytes,
            erb_dec_bytes,
            df_dec_bytes,
            erb_inv_fb,
            enc_layers,
            erb_layers,
            df_layers,
        )?;

        let features = FeatureExtractor::new(
            model.fft_size,
            model.hop_size,
            48_000,
            model.nb_erb,
            model.nb_spec,
            erb_fb,
            1.0,
        );

        Ok(Self { model, features })
    }

    /// Process one audio frame (`hop_size` samples @ 48 kHz) and return the LSNR
    /// for the last processed frame (for diagnostics).
    pub fn last_lsnr(&self) -> f32 {
        self.model.last_lsnr
    }

    /// Process one audio frame (`hop_size` samples @ 48 kHz).
    ///
    /// Pipeline (mirrors DFN3 / infer_streaming_onnx.py):
    ///   1. Silence gate — if frame power < 1e-7, return zeros (skip norm updates)
    ///   2. STFT analysis (Vorbis window, wnorm, overlap-add)
    ///   3. ERB log-power features + running-mean normalisation
    ///   4. Complex-spectrum features + running unit-norm normalisation
    ///   5. Encoder (ONNX) → LSNR → conditional ERB decoder → conditional DF decoder
    ///   6. iSTFT synthesis (Vorbis window, OLA)
    pub fn process_frame(&mut self, audio_frame: ArrayView1<f32>) -> Result<Array1<f32>> {
        // ── Silence gate (mirrors Python DFState.process_frame) ───────────────
        // Skip inference AND normalization state updates for silent frames.
        // Without this, unit_state converges toward zero → feat_spec explodes → silence.
        const SILENCE_THRESH: f32 = 1e-7;
        let frame_power = audio_frame.iter().map(|&x| x * x).sum::<f32>()
            / audio_frame.len() as f32;
        if frame_power < SILENCE_THRESH {
            return Ok(Array1::zeros(self.features.hop_size));
        }

        // ── STFT ─────────────────────────────────────────────────────────────
        let (spec_real, spec_imag) = self.features.compute_stft(audio_frame);

        // ── Feature extraction ────────────────────────────────────────────────
        let feat_erb = self.features.extract_erb_features(&spec_real, &spec_imag);
        let (feat_spec_real, feat_spec_imag) =
            self.features.extract_spec_features(&spec_real, &spec_imag);

        // ── Run encoder + conditional erb_dec + conditional df_dec ────────────
        // Pass flat arrays — model.rs builds tensors and handles all ONNX I/O
        let (enh_real, enh_imag) = self.model.process_frame(
            &spec_real,
            &spec_imag,
            &feat_erb,
            &feat_spec_real,
            &feat_spec_imag,
        )?;

        // ── Inverse STFT (OLA) ────────────────────────────────────────────────
        let enhanced_audio = self.features.inverse_stft(&enh_real, &enh_imag);

        Ok(enhanced_audio)
    }

    pub fn reset(&mut self) {
        self.model.reset_states();
        self.features.reset();
    }
}
