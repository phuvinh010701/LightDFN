//! Native Li-GRU inference path: feedforward ONNX (conv/linear) + Rust GRU.
//!
//! ## Why this is faster than the pulsed ONNX path
//!
//! The pulsed ONNX path (`model.rs`) expands each Li-GRU step into ~20–80 primitive
//! ONNX ops (matmul, LayerNorm decomposed into 10 ops, sigmoid, relu, …) and runs
//! them through tract's op dispatcher.  That dispatcher adds non-trivial overhead
//! per op, especially in WASM where there is no JIT.
//!
//! This module moves the entire GRU to Rust (`ligru.rs`) and keeps only the
//! convolutional / linear feed-forward layers in ONNX:
//!
//! ```text
//! enc_ff.onnx      conv encoder (no GRU) → emb_raw, e0..e3, c0
//! ──────────────────────────────────────────────────────────────
//! native Rust      enc Li-GRU step       → emb [emb_out], lsnr
//! native Rust      erb_dec Li-GRU step   → emb_d [emb_out]
//! erb_dec_ff.onnx  ERB decoder convs     → mask m
//! native Rust      df_dec Li-GRU step    → c_gru [df_hid]
//! df_dec_ff.onnx   DF decoder convs      → coefs
//! ```
//!
//! Additional gains when the model is trained with **BatchNorm** (recommended):
//! - `export_native_onnx.py` folds running stats → per-element `scale/offset`
//! - Rust just does `x * scale + offset` per gate — essentially free
//! - No LayerNorm mean/variance computation at all
//!
//! The GRU always runs (even for bypassed LSNR frames) so hidden-state evolution
//! is identical to the Python streaming model.  Only the expensive conv ONNX
//! models are conditionally skipped.

use crate::features::FeatureExtractor;
use crate::ligru::{zero_hidden, NativeGrus};
use crate::model::{ModelError, Result, CONV_CH, DF_ORDER, DF_PAD_BEFORE, FFT_SIZE, HOP_SIZE,
                   N_FREQS, NB_ERB, NB_SPEC, KT_DFP, KT_ENC};
use ndarray::{s, Array1, Array2, Array4, Array5, Axis};
use tract_core::prelude::*;
use tract_onnx::prelude::*;

const MIN_DB_THRESH: f32 = -10.0;
const MAX_DB_ERB_THRESH: f32 = 30.0;
const MAX_DB_DF_THRESH: f32 = 20.0;

type TractModel = TypedSimpleState<TypedModel, TypedSimplePlan<TypedModel>>;

fn load_ff_model(bytes: &[u8]) -> Result<TractModel> {
    let mut m = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .map_err(|e| ModelError::LoadError(format!("parse: {e}")))?
        .into_typed()
        .map_err(|e| ModelError::LoadError(format!("into_typed: {e}")))?;
    m.declutter()
        .map_err(|e| ModelError::LoadError(format!("declutter: {e}")))?;
    let opt = m
        .into_optimized()
        .map_err(|e| ModelError::LoadError(format!("optimize: {e}")))?;
    TypedSimpleState::new(
        SimplePlan::new(opt).map_err(|e| ModelError::LoadError(format!("plan: {e}")))?,
    )
    .map_err(|e| ModelError::LoadError(format!("state: {e}")))
}

// ---------------------------------------------------------------------------
// LightDFNNative
// ---------------------------------------------------------------------------

/// Native-GRU inference model.
///
/// Load with `LightDFNNative::from_bytes`; call `process_frame` each hop.
pub struct LightDFNNative {
    // Feedforward-only ONNX models (no GRU ops inside)
    enc_ff: TractModel,
    erb_dec_ff: TractModel,
    df_dec_ff: TractModel,

    // All three Li-GRU modules (weights loaded from ligru_weights.json)
    grus: NativeGrus,

    // GRU hidden states: Vec<Array1<f32>> — one Array1 per layer
    h_enc: Vec<Array1<f32>>,
    h_erb: Vec<Array1<f32>>,
    h_df: Vec<Array1<f32>>,

    // Causal conv ring-buffers (same layout as model.rs)
    buf_erb0: Array4<f32>, // [1, 1,       KT_ENC-1, NB_ERB]
    buf_df0: Array4<f32>,  // [1, 2,       KT_ENC-1, NB_SPEC]
    buf_dfp: Array4<f32>,  // [1, CONV_CH, KT_DFP-1, NB_SPEC]

    // DF spec context buffer for deep filtering
    buf_spec_re: Array2<f32>, // [DF_PAD_BEFORE, N_FREQS]
    buf_spec_im: Array2<f32>,

    // ERB inverse filterbank [NB_ERB, N_FREQS]
    erb_inv_fb: Array2<f32>,

    m_zeros: Array2<f32>,

    pub fft_size: usize,
    pub hop_size: usize,
    pub nb_erb: usize,
    pub nb_spec: usize,
    pub df_order: usize,
}

impl LightDFNNative {
    /// Construct from feedforward ONNX bytes + Li-GRU weight JSON.
    ///
    /// `ligru_json` is the content of `ligru_weights.json` produced by
    /// `export_native_onnx.py`.  The file embeds all weight matrices for
    /// `enc.emb_gru`, `erb_dec.emb_gru`, and `df_dec.df_gru`.
    pub fn from_bytes(
        enc_ff_bytes: &[u8],
        erb_dec_ff_bytes: &[u8],
        df_dec_ff_bytes: &[u8],
        ligru_json: &str,
        erb_inv_fb: Array2<f32>,
    ) -> Result<Self> {
        let enc_ff = load_ff_model(enc_ff_bytes)?;
        let erb_dec_ff = load_ff_model(erb_dec_ff_bytes)?;
        let df_dec_ff = load_ff_model(df_dec_ff_bytes)?;

        let grus = NativeGrus::from_json(ligru_json)
            .map_err(|e| ModelError::LoadError(format!("ligru JSON: {e}")))?;

        // Initialise GRU hidden states to zero (shapes come from the loaded weights)
        let h_enc = zero_hidden(grus.enc.gru.num_layers, grus.enc.gru.hidden_size);
        let h_erb = zero_hidden(grus.erb_dec.num_layers, grus.erb_dec.hidden_size);
        let h_df = zero_hidden(grus.df_dec.gru.num_layers, grus.df_dec.gru.hidden_size);

        Ok(Self {
            enc_ff,
            erb_dec_ff,
            df_dec_ff,
            grus,
            h_enc,
            h_erb,
            h_df,
            buf_erb0: Array4::zeros((1, 1, KT_ENC - 1, NB_ERB)),
            buf_df0: Array4::zeros((1, 2, KT_ENC - 1, NB_SPEC)),
            buf_dfp: Array4::zeros((1, CONV_CH, KT_DFP - 1, NB_SPEC)),
            buf_spec_re: Array2::zeros((DF_PAD_BEFORE, N_FREQS)),
            buf_spec_im: Array2::zeros((DF_PAD_BEFORE, N_FREQS)),
            erb_inv_fb,
            m_zeros: Array2::zeros((1, NB_ERB)),
            fft_size: FFT_SIZE,
            hop_size: HOP_SIZE,
            nb_erb: NB_ERB,
            nb_spec: NB_SPEC,
            df_order: DF_ORDER,
        })
    }

    // ── Ring buffer helpers ──────────────────────────────────────────────────

    /// Shift `buf` left along axis 2, append `new_frame` at the last slot.
    fn slide_buf(buf: &mut Array4<f32>, new_frame: &Array4<f32>) {
        let kt = buf.shape()[2];
        for t in 0..kt - 1 {
            let src = buf.slice(s![.., .., t + 1, ..]).to_owned();
            buf.slice_mut(s![.., .., t, ..]).assign(&src);
        }
        buf.slice_mut(s![.., .., kt - 1, ..])
            .assign(&new_frame.slice(s![.., .., 0, ..]));
    }

    // ── Tensor helpers ───────────────────────────────────────────────────────

    /// Wrap a flat `Array1<f32>` as a `[1, 1, dim]` tract `TValue`.
    fn arr1_to_tv(arr: &Array1<f32>) -> TValue {
        let dim = arr.len();
        let mut t3 = ndarray::Array3::<f32>::zeros((1, 1, dim));
        t3.slice_mut(s![0, 0, ..]).assign(arr);
        TValue::from(Tensor::from(t3.into_dyn()))
    }

    // ── LSNR gating ─────────────────────────────────────────────────────────

    /// Returns `(apply_gains, apply_gain_zeros, apply_df)`.
    fn lsnr_gates(lsnr: f32) -> (bool, bool, bool) {
        if lsnr < MIN_DB_THRESH {
            (false, true, false) // pure noise → zero mask, skip DF
        } else if lsnr > MAX_DB_ERB_THRESH {
            (false, false, false) // clean speech → ones mask (pass-through)
        } else if lsnr > MAX_DB_DF_THRESH {
            (true, false, false) // ERB only
        } else {
            (true, false, true) // ERB + DF
        }
    }

    // ── Core frame processing ────────────────────────────────────────────────

    /// Process one STFT frame through the noise-suppression pipeline.
    ///
    /// All three Li-GRU modules always run (regardless of LSNR) so that hidden
    /// states track identically to the Python streaming model.  Only the
    /// expensive convolutional ONNX models are conditionally skipped.
    ///
    /// Inputs:
    ///   `spec`      `[1, 1, 1, N_FREQS, 2]`   noisy complex spectrum
    ///   `feat_erb`  `[1, 1, 1, NB_ERB]`        ERB log-power (normalised)
    ///   `feat_spec` `[1, 1, 1, NB_SPEC, 2]`   complex spec (unit-normalised)
    pub fn process_frame(
        &mut self,
        spec: Array5<f32>,
        feat_erb: Array4<f32>,
        feat_spec: Array5<f32>,
    ) -> Result<Array5<f32>> {
        // ── Pack feat_spec [1,1,1,NB_SPEC,2] → real/imag [1,2,1,NB_SPEC] ──
        let mut cplx = Array4::<f32>::zeros((1, 2, 1, NB_SPEC));
        for i in 0..NB_SPEC {
            cplx[[0, 0, 0, i]] = feat_spec[[0, 0, 0, i, 0]];
            cplx[[0, 1, 0, i]] = feat_spec[[0, 0, 0, i, 1]];
        }

        // ── Build causal context windows ────────────────────────────────────
        let erb_ctx = ndarray::concatenate(Axis(2), &[self.buf_erb0.view(), feat_erb.view()])
            .map_err(|e| ModelError::InvalidShape(format!("erb ctx: {e}")))?;
        let cplx_ctx = ndarray::concatenate(Axis(2), &[self.buf_df0.view(), cplx.view()])
            .map_err(|e| ModelError::InvalidShape(format!("cplx ctx: {e}")))?;

        // Advance ring buffers (must happen before enc_ff so they hold correct past)
        Self::slide_buf(&mut self.buf_erb0, &feat_erb);
        Self::slide_buf(&mut self.buf_df0, &cplx);

        // ── enc_ff ONNX: conv encoder without GRU ───────────────────────────
        // Outputs: e0[0] e1[1] e2[2] e3[3] emb_raw[4] c0[5]
        let enc_out = self
            .enc_ff
            .run(tvec![
                TValue::from(Tensor::from(erb_ctx.into_dyn())),
                TValue::from(Tensor::from(cplx_ctx.into_dyn())),
            ])
            .map_err(|e| ModelError::InferenceError(format!("enc_ff: {e}")))?;

        // emb_raw: flatten [1,1,emb_in] → Array1 for native GRU
        let emb_raw_1d: Array1<f32> = enc_out[4]
            .to_array_view::<f32>()
            .map_err(|e| ModelError::InferenceError(e.to_string()))?
            .iter()
            .copied()
            .collect();

        // ── Native encoder GRU (always runs) ────────────────────────────────
        let (emb, lsnr, new_h_enc) = self.grus.enc.step(&emb_raw_1d, &self.h_enc);
        self.h_enc = new_h_enc;

        let (apply_gains, apply_gain_zeros, apply_df) = Self::lsnr_gates(lsnr);

        // ── c0 context for df_convp ─────────────────────────────────────────
        let c0_arr: Array4<f32> = enc_out[5]
            .to_array_view::<f32>()
            .map_err(|e| ModelError::InferenceError(e.to_string()))?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| ModelError::InvalidShape(e.to_string()))?
            .to_owned();
        let c0_ctx = ndarray::concatenate(Axis(2), &[self.buf_dfp.view(), c0_arr.view()])
            .map_err(|e| ModelError::InvalidShape(format!("dfp ctx: {e}")))?;
        Self::slide_buf(&mut self.buf_dfp, &c0_arr);

        // ── Native ERB decoder GRU (always runs) ────────────────────────────
        // Running always keeps state in sync with Python — only the conv ONNX
        // model (expensive) is skipped when LSNR is outside the ERB range.
        let (emb_d, new_h_erb) = self.grus.erb_dec.step(&emb, &self.h_erb);
        self.h_erb = new_h_erb;

        // Select ERB mask based on LSNR regime
        let mask_erb: Array2<f32> = if apply_gains {
            // Run ERB decoder conv stack with post-GRU emb_d
            let erb_out = self
                .erb_dec_ff
                .run(tvec![
                    Self::arr1_to_tv(&emb_d), // emb_d [1,1,emb_out]
                    enc_out[3].clone(),        // e3
                    enc_out[2].clone(),        // e2
                    enc_out[1].clone(),        // e1
                    enc_out[0].clone(),        // e0
                ])
                .map_err(|e| ModelError::InferenceError(format!("erb_dec_ff: {e}")))?;

            erb_out[0]
                .to_array_view::<f32>()
                .map_err(|e| ModelError::InferenceError(e.to_string()))?
                .to_owned()
                .into_shape_with_order((1, NB_ERB))
                .map_err(|e| ModelError::InvalidShape(e.to_string()))?
        } else if apply_gain_zeros {
            self.m_zeros.clone() // pure noise → zero mask
        } else {
            Array2::ones((1, NB_ERB)) // clean speech → pass-through
        };

        // Apply ERB mask → per-frequency gains
        let gains = mask_erb.dot(&self.erb_inv_fb); // [1, N_FREQS]
        let mut spec_m = spec.clone();
        for f in 0..N_FREQS {
            let g = gains[[0, f]];
            spec_m[[0, 0, 0, f, 0]] *= g;
            spec_m[[0, 0, 0, f, 1]] *= g;
        }

        // ── Native DF decoder GRU (always runs) ─────────────────────────────
        let (c_gru, new_h_df) = self.grus.df_dec.step(&emb, &self.h_df);
        self.h_df = new_h_df;

        // ── Deep filtering (conditional on LSNR) ────────────────────────────
        if apply_df {
            let df_out = self
                .df_dec_ff
                .run(tvec![
                    Self::arr1_to_tv(&c_gru),
                    TValue::from(Tensor::from(c0_ctx.into_dyn())),
                ])
                .map_err(|e| ModelError::InferenceError(format!("df_dec_ff: {e}")))?;

            // coefs: [1, DF_ORDER, 1, NB_SPEC, 2]
            let coefs: Array5<f32> = df_out[0]
                .to_array_view::<f32>()
                .map_err(|e| ModelError::InferenceError(e.to_string()))?
                .into_dimensionality::<ndarray::Ix5>()
                .map_err(|e| ModelError::InvalidShape(e.to_string()))?
                .to_owned();

            // Complex FIR over DF_ORDER past frames
            for f in 0..NB_SPEC {
                let mut r_acc = 0.0_f32;
                let mut i_acc = 0.0_f32;
                for o in 0..self.df_order {
                    let (sr, si) = if o < DF_PAD_BEFORE {
                        (self.buf_spec_re[[o, f]], self.buf_spec_im[[o, f]])
                    } else if o == DF_PAD_BEFORE {
                        (spec[[0, 0, 0, f, 0]], spec[[0, 0, 0, f, 1]])
                    } else {
                        (0.0, 0.0) // future frames zero-padded (causal streaming)
                    };
                    let cr = coefs[[0, o, 0, f, 0]];
                    let ci = coefs[[0, o, 0, f, 1]];
                    r_acc += sr * cr - si * ci;
                    i_acc += sr * ci + si * cr;
                }
                spec_m[[0, 0, 0, f, 0]] = r_acc;
                spec_m[[0, 0, 0, f, 1]] = i_acc;
            }
        }

        // Advance spec ring buffer (rotate left, then write current frame at end)
        if DF_PAD_BEFORE > 1 {
            for t in 0..DF_PAD_BEFORE - 1 {
                let src_re = self.buf_spec_re.slice(s![t + 1, ..]).to_owned();
                let src_im = self.buf_spec_im.slice(s![t + 1, ..]).to_owned();
                self.buf_spec_re.slice_mut(s![t, ..]).assign(&src_re);
                self.buf_spec_im.slice_mut(s![t, ..]).assign(&src_im);
            }
        }
        if DF_PAD_BEFORE > 0 {
            let last = DF_PAD_BEFORE - 1;
            for f in 0..N_FREQS {
                self.buf_spec_re[[last, f]] = spec[[0, 0, 0, f, 0]];
                self.buf_spec_im[[last, f]] = spec[[0, 0, 0, f, 1]];
            }
        }

        Ok(spec_m)
    }

    /// Reset all GRU states and ring buffers (equivalent to starting a new stream).
    pub fn reset(&mut self) {
        self.h_enc = zero_hidden(self.grus.enc.gru.num_layers, self.grus.enc.gru.hidden_size);
        self.h_erb = zero_hidden(self.grus.erb_dec.num_layers, self.grus.erb_dec.hidden_size);
        self.h_df = zero_hidden(
            self.grus.df_dec.gru.num_layers,
            self.grus.df_dec.gru.hidden_size,
        );
        self.buf_erb0.fill(0.0);
        self.buf_df0.fill(0.0);
        self.buf_dfp.fill(0.0);
        self.buf_spec_re.fill(0.0);
        self.buf_spec_im.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// StreamingProcessorNative — wraps LightDFNNative + FeatureExtractor
// ---------------------------------------------------------------------------

/// High-level streaming processor using native Rust Li-GRU.
///
/// Mirrors `StreamingProcessor` (streaming.rs) but routes GRU through Rust
/// instead of tract ONNX.
pub struct StreamingProcessorNative {
    model: LightDFNNative,
    features: FeatureExtractor,
}

impl StreamingProcessorNative {
    /// Create from feedforward ONNX bytes + GRU weight JSON + ERB filterbanks.
    ///
    /// `erb_fb`:     `[n_freqs, nb_erb]` — forward ERB filterbank (feature extraction)
    /// `erb_inv_fb`: `[nb_erb, n_freqs]` — inverse filterbank (ERB mask → per-freq gains)
    pub fn new(
        enc_ff_bytes: &[u8],
        erb_dec_ff_bytes: &[u8],
        df_dec_ff_bytes: &[u8],
        ligru_json: &str,
        erb_fb: Array2<f32>,
        erb_inv_fb: Array2<f32>,
    ) -> Result<Self> {
        let model = LightDFNNative::from_bytes(
            enc_ff_bytes,
            erb_dec_ff_bytes,
            df_dec_ff_bytes,
            ligru_json,
            erb_inv_fb,
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

    /// Process one audio hop (`hop_size` samples @ 48 kHz).
    pub fn process_frame(
        &mut self,
        audio: ndarray::ArrayView1<f32>,
    ) -> Result<Array1<f32>> {
        let (spec_r, spec_i) = self.features.compute_stft(audio);
        let feat_erb = self.features.extract_erb_features(&spec_r, &spec_i);
        let (feat_spec_r, feat_spec_i) = self.features.extract_spec_features(&spec_r, &spec_i);
        let (spec, feat_erb_4d, feat_spec) = self.features.pack_features(
            &spec_r,
            &spec_i,
            &feat_erb,
            &feat_spec_r,
            &feat_spec_i,
        );

        let enhanced = self.model.process_frame(spec, feat_erb_4d, feat_spec)?;

        let n_freqs = self.model.fft_size / 2 + 1;
        let mut enh_r = Array1::<f32>::zeros(n_freqs);
        let mut enh_i = Array1::<f32>::zeros(n_freqs);
        for i in 0..n_freqs {
            enh_r[i] = enhanced[[0, 0, 0, i, 0]];
            enh_i[i] = enhanced[[0, 0, 0, i, 1]];
        }

        Ok(self.features.inverse_stft(&enh_r, &enh_i))
    }

    pub fn reset(&mut self) {
        self.model.reset();
        self.features.reset();
    }
}
