use ndarray::{s, Array1, Array2};
use thiserror::Error;
use tract_core::prelude::*;
use tract_onnx::prelude::*;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Failed to load model: {0}")]
    LoadError(String),
    #[error("Inference failed: {0}")]
    InferenceError(String),
    #[error("Invalid input shape: {0}")]
    InvalidShape(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;

type TractModel = TypedSimpleState<TypedModel, TypedSimplePlan<TypedModel>>;

// ── LSNR thresholds (matching DFN3 defaults) ─────────────────────────────────
const MIN_DB_THRESH: f32 = -10.0;
const MAX_DB_ERB_THRESH: f32 = 30.0;
const MAX_DB_DF_THRESH: f32 = 20.0;

// ── Model parameters (must match exported ONNX) ───────────────────────────────
pub const FFT_SIZE: usize = 960;
pub const HOP_SIZE: usize = 480;
pub const NB_ERB: usize = 32;
pub const NB_SPEC: usize = 96; // nb_df
pub const DF_ORDER: usize = 5;
pub const DF_LOOKAHEAD: usize = 2;
pub const DF_PAD_BEFORE: usize = DF_ORDER - 1 - DF_LOOKAHEAD; // = 2
pub const N_FREQS: usize = FFT_SIZE / 2 + 1; // 481
pub const EMB_DIM: usize = 256; // Li-GRU hidden state size (h_enc, h_erb)
pub const DF_DIM: usize = 256; // DF decoder Li-GRU hidden size
pub const CONV_CH: usize = 64;

// Context window sizes for each model input (from ONNX shape inspection):
//   enc  feat_erb_ctx:  [1, 1, KT_ENC, NB_ERB]  KT_ENC=3
//   enc  feat_spec_ctx: [1, 2, KT_ENC, NB_SPEC]  KT_ENC=3
//   df_dec c0_ctx:      [1, CONV_CH, KT_DFP, NB_SPEC]  KT_DFP=5
pub const KT_ENC: usize = 3; // temporal context window for encoder conv0
pub const KT_DFP: usize = 5; // temporal context window for df_dec convp

/// Load a pulsed ONNX model.
fn load_pulsed_model(bytes: &[u8]) -> Result<TractModel> {
    let m = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .map_err(|e| ModelError::LoadError(format!("parse: {e}")))?
        .into_typed()
        .map_err(|e| ModelError::LoadError(format!("into_typed: {e}")))?;

    let mut typed = m;
    typed
        .declutter()
        .map_err(|e| ModelError::LoadError(format!("declutter: {e}")))?;

    let optimized = typed
        .into_optimized()
        .map_err(|e| ModelError::LoadError(format!("optimize: {e}")))?;

    let plan = SimplePlan::new(optimized)
        .map_err(|e| ModelError::LoadError(format!("plan: {e}")))?;
    TypedSimpleState::new(plan)
        .map_err(|e| ModelError::LoadError(format!("state: {e}")))
}

/// LightDFN inference engine — pulsed ONNX variant.
///
/// ## Architecture (mirrors DeepFilterNet3 / infer_streaming_onnx.py)
///
/// Three ONNX models. The caller maintains ring buffers for temporal context
/// and pre-concatenates them before each ONNX call (_ctx suffix inputs).
///
///   enc.onnx:
///     inputs:  feat_erb_ctx [1,1,KT_ENC,NB_ERB], feat_spec_ctx [1,2,KT_ENC,NB_SPEC],
///              h_enc [1,1,EMB_DIM]
///     outputs: e0, e1, e2, e3, emb, c0, lsnr, h_enc_new
///
///   erb_dec.onnx:
///     inputs:  emb, e3, e2, e1, e0, h_erb [2,1,EMB_DIM]
///     outputs: m [1,1,1,NB_ERB], h_erb_new [2,1,EMB_DIM]
///
///   df_dec.onnx:
///     inputs:  emb, c0_ctx [1,CONV_CH,KT_DFP,NB_SPEC], h_df [2,1,DF_DIM]
///     outputs: coefs [1,DF_ORDER,1,NB_SPEC,2], h_df_new [2,1,DF_DIM]
pub struct LightDFNModel {
    // ── ONNX sub-models ─────────────────────────────────────────────────────
    enc: TractModel,
    erb_dec: TractModel,
    df_dec: TractModel,

    // ── GRU hidden states (tract Tensors, passed as inputs each call) ───────
    h_enc: Tensor, // [1, 1, EMB_DIM]
    h_erb: Tensor, // [erb_layers, 1, EMB_DIM]
    h_df: Tensor,  // [df_layers,  1, DF_DIM]

    pub fft_size: usize,
    pub hop_size: usize,
    pub nb_erb: usize,
    pub nb_spec: usize,
    pub df_order: usize,

    // ── Caller-managed ring buffers for context concatenation ────────────────
    // enc:  feat_erb ring: last KT_ENC-1 frames [KT_ENC-1, NB_ERB]
    erb_ctx: Vec<f32>, // (KT_ENC-1) * NB_ERB
    // enc:  feat_spec ring: last KT_ENC-1 frames, re+im interleaved [KT_ENC-1, 2, NB_SPEC]
    spec_ctx_re: Vec<f32>, // (KT_ENC-1) * NB_SPEC
    spec_ctx_im: Vec<f32>, // (KT_ENC-1) * NB_SPEC
    // df_dec: c0 ring: last KT_DFP-1 frames [KT_DFP-1, CONV_CH, NB_SPEC]
    c0_ctx: Vec<f32>, // (KT_DFP-1) * CONV_CH * NB_SPEC

    // ── DF spec context buffer (past noisy spec frames for deep filtering) ───
    buf_spec_re: Array2<f32>, // [DF_PAD_BEFORE, N_FREQS]
    buf_spec_im: Array2<f32>, // [DF_PAD_BEFORE, N_FREQS]

    // ── ERB inverse filterbank ──────────────────────────────────────────────
    erb_inv_fb: Array2<f32>, // [NB_ERB, N_FREQS]

    m_zeros: Array2<f32>, // zeroed mask for pure-noise frames

    // ── Layer counts (for state init/reset) ─────────────────────────────────
    enc_layers: usize,
    erb_layers: usize,
    df_layers: usize,

    /// Last LSNR value (dB) from the encoder — for diagnostics/gating.
    pub last_lsnr: f32,
}

impl LightDFNModel {
    /// Construct from three ONNX byte slices.
    pub fn from_bytes(
        enc_bytes: &[u8],
        erb_dec_bytes: &[u8],
        df_dec_bytes: &[u8],
        erb_inv_fb: Array2<f32>,
        enc_layers: usize,
        erb_layers: usize,
        df_layers: usize,
    ) -> Result<Self> {
        let enc = load_pulsed_model(enc_bytes)?;
        let erb_dec = load_pulsed_model(erb_dec_bytes)?;
        let df_dec = load_pulsed_model(df_dec_bytes)?;

        // GRU hidden states — zero-initialised
        let h_enc = Tensor::zero::<f32>(&[enc_layers, 1, EMB_DIM])
            .map_err(|e| ModelError::LoadError(format!("h_enc zero: {e}")))?;
        let h_erb = Tensor::zero::<f32>(&[erb_layers, 1, EMB_DIM])
            .map_err(|e| ModelError::LoadError(format!("h_erb zero: {e}")))?;
        let h_df = Tensor::zero::<f32>(&[df_layers, 1, DF_DIM])
            .map_err(|e| ModelError::LoadError(format!("h_df zero: {e}")))?;

        Ok(Self {
            enc,
            erb_dec,
            df_dec,
            h_enc,
            h_erb,
            h_df,
            fft_size: FFT_SIZE,
            hop_size: HOP_SIZE,
            nb_erb: NB_ERB,
            nb_spec: NB_SPEC,
            df_order: DF_ORDER,
            // Ring buffers: zero-initialised, capacity = (window-1) frames
            erb_ctx: vec![0.0_f32; (KT_ENC - 1) * NB_ERB],
            spec_ctx_re: vec![0.0_f32; (KT_ENC - 1) * NB_SPEC],
            spec_ctx_im: vec![0.0_f32; (KT_ENC - 1) * NB_SPEC],
            c0_ctx: vec![0.0_f32; (KT_DFP - 1) * CONV_CH * NB_SPEC],
            buf_spec_re: Array2::zeros((DF_PAD_BEFORE, N_FREQS)),
            buf_spec_im: Array2::zeros((DF_PAD_BEFORE, N_FREQS)),
            erb_inv_fb,
            m_zeros: Array2::zeros((1, NB_ERB)),
            enc_layers,
            erb_layers,
            df_layers,
            last_lsnr: 0.0,
        })
    }

    fn apply_stages(&self, lsnr: f32) -> (bool, bool, bool) {
        if lsnr < MIN_DB_THRESH {
            (false, true, false)
        } else if lsnr > MAX_DB_ERB_THRESH {
            (false, false, false)
        } else if lsnr > MAX_DB_DF_THRESH {
            (true, false, false)
        } else {
            (true, false, true)
        }
    }

    /// Process one STFT frame through the full noise-suppression pipeline.
    ///
    /// Matches `infer_streaming_onnx.py::DFState.process_frame()` exactly:
    ///   - Caller concatenates context ring buffers to build _ctx tensors.
    ///   - GRU hidden states passed as inputs, returned updated as outputs.
    ///
    /// Inputs:
    ///   spec_re/im      [N_FREQS]   — raw noisy complex spectrum
    ///   feat_erb        [NB_ERB]    — ERB log-power (normalised)
    ///   feat_spec_re/im [NB_SPEC]   — complex spectrum (unit-normalised)
    ///
    /// Returns:
    ///   (enhanced_re, enhanced_im)  — enhanced complex spectrum [N_FREQS]
    pub fn process_frame(
        &mut self,
        spec_re: &Array1<f32>,
        spec_im: &Array1<f32>,
        feat_erb: &Array1<f32>,
        feat_spec_re: &Array1<f32>,
        feat_spec_im: &Array1<f32>,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        // ── Build encoder context inputs ─────────────────────────────────────
        // feat_erb_ctx: [1, 1, KT_ENC, NB_ERB]
        //   Layout: time-major, past first then current.
        //   Flat layout: [t=0..KT_ENC-2 (past), t=KT_ENC-1 (current)] × NB_ERB
        let mut feat_erb_ctx = vec![0.0_f32; KT_ENC * NB_ERB];
        // past KT_ENC-1 frames from ring buffer
        feat_erb_ctx[..(KT_ENC - 1) * NB_ERB].copy_from_slice(&self.erb_ctx);
        // current frame
        feat_erb_ctx[(KT_ENC - 1) * NB_ERB..].copy_from_slice(
            feat_erb.as_slice().unwrap(),
        );
        let feat_erb_t =
            Tensor::from_shape(&[1, 1, KT_ENC, NB_ERB], &feat_erb_ctx)
                .map_err(|e| ModelError::InvalidShape(format!("feat_erb_ctx: {e}")))?;

        // feat_spec_ctx: [1, 2, KT_ENC, NB_SPEC]
        //   Channel 0 = real, channel 1 = imag.
        //   Flat layout (row-major): [ch, t, freq]
        let mut feat_spec_ctx = vec![0.0_f32; 2 * KT_ENC * NB_SPEC];
        // channel 0 (real): past then current
        let ch0 = &mut feat_spec_ctx[..KT_ENC * NB_SPEC];
        ch0[..(KT_ENC - 1) * NB_SPEC].copy_from_slice(&self.spec_ctx_re);
        ch0[(KT_ENC - 1) * NB_SPEC..].copy_from_slice(feat_spec_re.as_slice().unwrap());
        // channel 1 (imag): past then current
        let ch1 = &mut feat_spec_ctx[KT_ENC * NB_SPEC..];
        ch1[..(KT_ENC - 1) * NB_SPEC].copy_from_slice(&self.spec_ctx_im);
        ch1[(KT_ENC - 1) * NB_SPEC..].copy_from_slice(feat_spec_im.as_slice().unwrap());
        let feat_spec_t =
            Tensor::from_shape(&[1, 2, KT_ENC, NB_SPEC], &feat_spec_ctx)
                .map_err(|e| ModelError::InvalidShape(format!("feat_spec_ctx: {e}")))?;

        // ── Run encoder ──────────────────────────────────────────────────────
        // Inputs:  feat_erb_ctx [1,1,KT_ENC,NB_ERB],
        //          feat_spec_ctx [1,2,KT_ENC,NB_SPEC],
        //          h_enc [enc_layers,1,EMB_DIM]
        // Outputs: e0[0], e1[1], e2[2], e3[3], emb[4], c0[5], lsnr[6], h_enc_new[7]
        let enc_out = self
            .enc
            .run(tvec![
                TValue::from(feat_erb_t),
                TValue::from(feat_spec_t),
                TValue::from(self.h_enc.clone()),
            ])
            .map_err(|e| ModelError::InferenceError(format!("enc: {e}")))?;

        self.h_enc = enc_out[7].clone().into_tensor();

        let lsnr = *enc_out[6]
            .to_scalar::<f32>()
            .map_err(|e| ModelError::InferenceError(format!("lsnr scalar: {e}")))?;
        self.last_lsnr = lsnr;

        // ── Update encoder context ring buffers ──────────────────────────────
        // Slide past-window forward: drop oldest frame, append current.
        if KT_ENC > 1 {
            let step = NB_ERB;
            self.erb_ctx.copy_within(step.., 0);
            let last = (KT_ENC - 2) * NB_ERB;
            self.erb_ctx[last..].copy_from_slice(feat_erb.as_slice().unwrap());

            let step_s = NB_SPEC;
            self.spec_ctx_re.copy_within(step_s.., 0);
            let last_s = (KT_ENC - 2) * NB_SPEC;
            self.spec_ctx_re[last_s..].copy_from_slice(feat_spec_re.as_slice().unwrap());
            self.spec_ctx_im.copy_within(step_s.., 0);
            self.spec_ctx_im[last_s..].copy_from_slice(feat_spec_im.as_slice().unwrap());
        }

        let (apply_gains, apply_gain_zeros, apply_df) = self.apply_stages(lsnr);

        // ── ERB decoder ──────────────────────────────────────────────────────
        // Inputs:  emb, e3, e2, e1, e0, h_erb
        // Outputs: m [1,1,1,NB_ERB], h_erb_new
        let mask_erb: Array2<f32> = if apply_gains {
            let erb_out = self
                .erb_dec
                .run(tvec![
                    enc_out[4].clone(), // emb
                    enc_out[3].clone(), // e3
                    enc_out[2].clone(), // e2
                    enc_out[1].clone(), // e1
                    enc_out[0].clone(), // e0
                    TValue::from(self.h_erb.clone()),
                ])
                .map_err(|e| ModelError::InferenceError(format!("erb_dec: {e}")))?;

            self.h_erb = erb_out[1].clone().into_tensor();

            erb_out[0]
                .to_array_view::<f32>()
                .map_err(|e| ModelError::InferenceError(e.to_string()))?
                .to_owned()
                .into_shape_with_order((1, self.nb_erb))
                .map_err(|e| ModelError::InvalidShape(e.to_string()))?
        } else if apply_gain_zeros {
            self.m_zeros.clone()
        } else {
            Array2::ones((1, self.nb_erb))
        };

        // ── Apply ERB mask → per-frequency gains ────────────────────────────
        let gains = mask_erb.dot(&self.erb_inv_fb); // [1, N_FREQS]
        let mut enh_re = spec_re.clone();
        let mut enh_im = spec_im.clone();
        for f in 0..N_FREQS {
            let g = gains[[0, f]];
            enh_re[f] *= g;
            enh_im[f] *= g;
        }

        // ── DF decoder ───────────────────────────────────────────────────────
        if apply_df {
            // c0_ctx: [1, CONV_CH, KT_DFP, NB_SPEC]
            // Layout: [ch, t, freq] — past KT_DFP-1 frames + current c0.
            // c0 from encoder: [1, CONV_CH, 1, NB_SPEC]
            let c0_slice = enc_out[5]
                .to_array_view::<f32>()
                .map_err(|e| ModelError::InferenceError(e.to_string()))?;
            // c0_slice has shape [1, CONV_CH, 1, NB_SPEC]; flatten to [CONV_CH * NB_SPEC]
            let c0_flat: Vec<f32> = c0_slice.iter().copied().collect();

            // Build c0_ctx: [1, CONV_CH, KT_DFP, NB_SPEC]
            // For each channel ch: [past (KT_DFP-1) frames | current frame]
            // c0_ctx flat layout: [ch=0..CONV_CH, t=0..KT_DFP, freq=0..NB_SPEC]
            let mut c0_ctx_full = vec![0.0_f32; CONV_CH * KT_DFP * NB_SPEC];
            for ch in 0..CONV_CH {
                let dst_ch = ch * KT_DFP * NB_SPEC;
                // past frames from ring buffer
                let past_len = (KT_DFP - 1) * NB_SPEC;
                let src_past = ch * (KT_DFP - 1) * NB_SPEC;
                c0_ctx_full[dst_ch..dst_ch + past_len]
                    .copy_from_slice(&self.c0_ctx[src_past..src_past + past_len]);
                // current frame from c0_flat
                let src_cur = ch * NB_SPEC;
                let dst_cur = dst_ch + past_len;
                c0_ctx_full[dst_cur..dst_cur + NB_SPEC]
                    .copy_from_slice(&c0_flat[src_cur..src_cur + NB_SPEC]);
            }
            let c0_ctx_t =
                Tensor::from_shape(&[1, CONV_CH, KT_DFP, NB_SPEC], &c0_ctx_full)
                    .map_err(|e| ModelError::InvalidShape(format!("c0_ctx: {e}")))?;

            let df_out = self
                .df_dec
                .run(tvec![
                    enc_out[4].clone(), // emb
                    TValue::from(c0_ctx_t),
                    TValue::from(self.h_df.clone()),
                ])
                .map_err(|e| ModelError::InferenceError(format!("df_dec: {e}")))?;

            self.h_df = df_out[1].clone().into_tensor();

            // Update c0 ring buffer: slide forward, store current c0
            if KT_DFP > 1 {
                for ch in 0..CONV_CH {
                    let base = ch * (KT_DFP - 1) * NB_SPEC;
                    self.c0_ctx.copy_within(base + NB_SPEC..base + (KT_DFP - 1) * NB_SPEC, base);
                    let last_pos = base + (KT_DFP - 2) * NB_SPEC;
                    let src = ch * NB_SPEC;
                    self.c0_ctx[last_pos..last_pos + NB_SPEC]
                        .copy_from_slice(&c0_flat[src..src + NB_SPEC]);
                }
            }

            let coefs = df_out[0]
                .to_array_view::<f32>()
                .map_err(|e| ModelError::InferenceError(e.to_string()))?
                .into_dimensionality::<ndarray::Ix5>()
                .map_err(|e| ModelError::InvalidShape(e.to_string()))?;
            // coefs: [1, DF_ORDER, 1, NB_SPEC, 2]

            // ── Deep filtering: complex FIR over DF_ORDER past frames ────────
            // Context alignment:
            //   o=0..DF_PAD_BEFORE-1 → oldest..newest past frames in buf_spec
            //   o=DF_PAD_BEFORE      → current noisy frame
            //   o>DF_PAD_BEFORE      → future frames → zero (lookahead=0 causal)
            let nb_df = self.nb_spec;
            for f in 0..nb_df {
                let mut r_acc = 0.0_f32;
                let mut i_acc = 0.0_f32;
                for o in 0..self.df_order {
                    let (sr, si) = if o < DF_PAD_BEFORE {
                        (self.buf_spec_re[[o, f]], self.buf_spec_im[[o, f]])
                    } else if o == DF_PAD_BEFORE {
                        (spec_re[f], spec_im[f])
                    } else {
                        (0.0_f32, 0.0_f32)
                    };
                    let cr = coefs[[0, o, 0, f, 0]];
                    let ci = coefs[[0, o, 0, f, 1]];
                    r_acc += sr * cr - si * ci;
                    i_acc += sr * ci + si * cr;
                }
                enh_re[f] = r_acc;
                enh_im[f] = i_acc;
            }
        } else {
            // When not applying DF: still update c0 ring buffer with current c0
            let c0_slice = enc_out[5]
                .to_array_view::<f32>()
                .map_err(|e| ModelError::InferenceError(e.to_string()))?;
            let c0_flat: Vec<f32> = c0_slice.iter().copied().collect();
            if KT_DFP > 1 {
                for ch in 0..CONV_CH {
                    let base = ch * (KT_DFP - 1) * NB_SPEC;
                    self.c0_ctx.copy_within(base + NB_SPEC..base + (KT_DFP - 1) * NB_SPEC, base);
                    let last_pos = base + (KT_DFP - 2) * NB_SPEC;
                    let src = ch * NB_SPEC;
                    self.c0_ctx[last_pos..last_pos + NB_SPEC]
                        .copy_from_slice(&c0_flat[src..src + NB_SPEC]);
                }
            }
        }

        // ── Advance noisy-spec ring buffer (for deep filtering) ──────────────
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
                self.buf_spec_re[[last, f]] = spec_re[f];
                self.buf_spec_im[[last, f]] = spec_im[f];
            }
        }

        Ok((enh_re, enh_im))
    }

    pub fn reset_states(&mut self) {
        self.h_enc = Tensor::zero::<f32>(&[self.enc_layers, 1, EMB_DIM]).unwrap();
        self.h_erb = Tensor::zero::<f32>(&[self.erb_layers, 1, EMB_DIM]).unwrap();
        self.h_df = Tensor::zero::<f32>(&[self.df_layers, 1, DF_DIM]).unwrap();
        self.erb_ctx.iter_mut().for_each(|x| *x = 0.0);
        self.spec_ctx_re.iter_mut().for_each(|x| *x = 0.0);
        self.spec_ctx_im.iter_mut().for_each(|x| *x = 0.0);
        self.c0_ctx.iter_mut().for_each(|x| *x = 0.0);
        self.buf_spec_re.fill(0.0);
        self.buf_spec_im.fill(0.0);
    }
}
