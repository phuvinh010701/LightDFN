use ndarray::{s, Array1, Array2, Array4, Array5, ArrayView1};
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use std::f32::consts::PI;
use std::sync::Arc;

/// Audio feature extraction for LightDFN — matches libDF's DFState.
///
/// Window: Vorbis/sine window (same as DFN3/libDF) which satisfies the
/// constant-overlap-add (COLA) constraint at 50% overlap.
///
/// STFT convention:
///   - Analysis: overlap-add of past `fft_size - hop_size` samples + new `hop_size`.
///   - Synthesis: proper OLA with the Vorbis window applied to both analysis
///     and synthesis halves.
pub struct FeatureExtractor {
    pub fft_size: usize,
    pub hop_size: usize,
    pub sr: usize,
    pub nb_erb: usize,
    pub nb_spec: usize,

    // Vorbis/sine window — matches DFN3/libDF.
    window: Array1<f32>,

    // STFT analysis normalisation: wnorm = 1 / (fft_size² / (2 * hop_size))
    // Matches DFN3 lib.rs and Python streaming.py exactly.
    wnorm: f32,

    // ERB filterbank [n_freqs, nb_erb]
    erb_fb: Array2<f32>,

    // Normalisation states
    erb_state: Array1<f32>,  // [nb_erb]  — running mean for ERB log-power
    unit_state: Array1<f32>, // [nb_spec] — running unit norm for complex spec

    // Smoothing factor α = exp(-hop_size / (sr * tau))
    alpha: f32,
    one_minus_alpha: f32,

    // Analysis overlap buffer: last `fft_size - hop_size` input samples
    analysis_mem: Array1<f32>,

    // Synthesis OLA buffer: last `fft_size - hop_size` output samples waiting
    // to be added to the next output block
    synthesis_mem: Array1<f32>,

    // FFT plans (pre-computed, reused every frame)
    fft_forward: Arc<dyn realfft::RealToComplex<f32>>,
    fft_inverse: Arc<dyn realfft::ComplexToReal<f32>>,
    fft_scratch_fwd: Vec<Complex<f32>>,
    fft_scratch_inv: Vec<Complex<f32>>,

    // Pre-allocated scratch buffers (reused every frame)
    stft_input: Vec<f32>,
    stft_spectrum: Vec<Complex<f32>>,
}

impl FeatureExtractor {
    pub fn new(
        fft_size: usize,
        hop_size: usize,
        sr: usize,
        nb_erb: usize,
        nb_spec: usize,
        erb_fb: Array2<f32>,
        norm_tau: f32,
    ) -> Self {
        // ── Vorbis/sine window ─────────────────────────────────────────────
        // w[i] = sin(0.5·π·sin²(0.5·π·(i+0.5)/H))  where H = fft_size/2
        // Identical to Python streaming.py and DFN3/libDF.
        let h = (fft_size / 2) as f32;
        let mut window = Array1::<f32>::zeros(fft_size);
        for i in 0..fft_size {
            let t = (i as f32 + 0.5) / h;
            window[i] = (0.5 * PI * (0.5 * PI * t).sin().powi(2)).sin();
        }

        // ── STFT analysis normalisation: wnorm ────────────────────────────
        // Matches DFN3 lib.rs and Python streaming.py:
        //   wnorm = 1.0 / (fft_size² / (2 * hop_size))
        // For fft_size=960, hop_size=480: wnorm = 1.0 / 960 ≈ 0.001042
        let wnorm = 1.0 / (fft_size as f32 * fft_size as f32 / (2.0 * hop_size as f32));

        // ── Normalisation factor α ─────────────────────────────────────────
        let alpha = (-(hop_size as f32) / (sr as f32 * norm_tau)).exp();
        let one_minus_alpha = 1.0 - alpha;

        // ── Running-mean ERB state seed (libDF MEAN_NORM_INIT) ─────────────
        // Linear from -60 dB to -90 dB across nb_erb bands.
        let mut erb_state = Array1::<f32>::zeros(nb_erb);
        for i in 0..nb_erb {
            erb_state[i] = -60.0 - (30.0 * i as f32) / (nb_erb.saturating_sub(1).max(1)) as f32;
        }

        // ── Unit-norm state seed (libDF UNIT_NORM_INIT) ────────────────────
        // Linear from 0.001 to 0.0001.
        let mut unit_state = Array1::<f32>::zeros(nb_spec);
        for i in 0..nb_spec {
            unit_state[i] = 0.001 - (0.0009 * i as f32) / (nb_spec.saturating_sub(1).max(1)) as f32;
        }

        // Overlap = fft_size - hop_size samples
        let overlap = fft_size - hop_size;
        let analysis_mem = Array1::<f32>::zeros(overlap);
        let synthesis_mem = Array1::<f32>::zeros(overlap);

        // Pre-plan FFTs (expensive, done once at init)
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(fft_size);
        let fft_inverse = planner.plan_fft_inverse(fft_size);
        let fft_scratch_fwd = fft_forward.make_scratch_vec();
        let fft_scratch_inv = fft_inverse.make_scratch_vec();

        // Pre-allocated scratch buffers (reused every frame, avoid per-frame alloc)
        let stft_input = vec![0.0_f32; fft_size];
        let stft_spectrum = fft_forward.make_output_vec();

        Self {
            fft_size,
            hop_size,
            sr,
            nb_erb,
            nb_spec,
            window,
            wnorm,
            erb_fb,
            erb_state,
            unit_state,
            alpha,
            one_minus_alpha,
            analysis_mem,
            synthesis_mem,
            fft_forward,
            fft_inverse,
            fft_scratch_fwd,
            fft_scratch_inv,
            stft_input,
            stft_spectrum,
        }
    }

    // ── STFT analysis ──────────────────────────────────────────────────────────

    /// Compute STFT of one audio hop.
    ///
    /// Builds a full `fft_size` frame by prepending the analysis overlap buffer,
    /// applies the Vorbis window, runs the real FFT, and returns the complex
    /// spectrum as two separate real/imag arrays of length `fft_size/2 + 1`.
    ///
    /// The analysis_mem is updated to hold the latest `fft_size - hop_size`
    /// input samples, ready for the next call.
    pub fn compute_stft(&mut self, audio_frame: ArrayView1<f32>) -> (Array1<f32>, Array1<f32>) {
        assert_eq!(audio_frame.len(), self.hop_size);
        let overlap = self.fft_size - self.hop_size;

        // Build full frame in-place: [overlap | hop] — reuse stft_input buffer
        for i in 0..overlap {
            self.stft_input[i] = self.analysis_mem[i];
        }
        for i in 0..self.hop_size {
            self.stft_input[overlap + i] = audio_frame[i];
        }

        // Advance analysis overlap buffer
        if overlap <= self.hop_size {
            let start = self.hop_size - overlap;
            for i in 0..overlap {
                self.analysis_mem[i] = audio_frame[start + i];
            }
        } else {
            // overlap > hop_size: shift existing mem left by hop_size, append new frame
            let keep = overlap - self.hop_size;
            for i in 0..keep {
                self.analysis_mem[i] = self.analysis_mem[i + self.hop_size];
            }
            for i in 0..self.hop_size {
                self.analysis_mem[keep + i] = audio_frame[i];
            }
        }

        // Apply Vorbis window
        for i in 0..self.fft_size {
            self.stft_input[i] *= self.window[i];
        }

        // Forward FFT — reuse pre-allocated spectrum buffer
        let n_freqs = self.fft_size / 2 + 1;
        self.fft_forward
            .process_with_scratch(&mut self.stft_input, &mut self.stft_spectrum, &mut self.fft_scratch_fwd)
            .expect("FFT failed");

        // Apply wnorm to match DFN3/Python: wnorm = 1/(fft_size²/(2*hop_size))
        let mut spec_real = Array1::<f32>::zeros(n_freqs);
        let mut spec_imag = Array1::<f32>::zeros(n_freqs);
        for k in 0..n_freqs {
            spec_real[k] = self.stft_spectrum[k].re * self.wnorm;
            spec_imag[k] = self.stft_spectrum[k].im * self.wnorm;
        }

        (spec_real, spec_imag)
    }

    // ── STFT synthesis (OLA) ───────────────────────────────────────────────────

    /// Inverse STFT with overlap-add synthesis.
    ///
    /// Applies the Vorbis window to the reconstructed time-domain frame,
    /// then performs overlap-add with `synthesis_mem`.
    ///
    /// Returns the first `hop_size` samples of the reconstructed signal
    /// (the portion that is complete).  The remaining `fft_size - hop_size`
    /// samples are stored in `synthesis_mem` for the next call.
    ///
    /// The normalisation chain (with wnorm at analysis):
    ///   Analysis: FFT output × wnorm (wnorm = 2*hop/fft_size²)
    ///   Synthesis: realfft IFFT output = fft_size × conventional IFFT.
    ///             Apply Vorbis window (COLA=1 at 50% overlap).
    ///             No extra 1/fft_size scale — wnorm × fft_size provides
    ///             the correct round-trip gain with Vorbis OLA.
    ///
    /// This matches DFN3 lib.rs synthesis and Python build_irfft_matrix
    /// where irfft = fft_size * window × irfft_basis.
    pub fn inverse_stft(
        &mut self,
        spec_real: &Array1<f32>,
        spec_imag: &Array1<f32>,
    ) -> Array1<f32> {
        let n_freqs = self.fft_size / 2 + 1;
        let overlap = self.fft_size - self.hop_size;

        // Build complex spectrum for realfft (expects DC and Nyquist to be real)
        let mut spectrum: Vec<Complex<f32>> = (0..n_freqs)
            .map(|k| Complex::new(spec_real[k], spec_imag[k]))
            .collect();
        spectrum[0].im = 0.0;
        spectrum[n_freqs - 1].im = 0.0;

        let mut output = self.fft_inverse.make_output_vec();
        self.fft_inverse
            .process_with_scratch(&mut spectrum, &mut output, &mut self.fft_scratch_inv)
            .expect("IFFT failed");

        // realfft IFFT gives fft_size × conventional IFFT output.
        // Apply Vorbis window only — NO 1/fft_size division.
        // The wnorm applied at analysis (wnorm = 2*hop/N²) combined with the
        // IFFT factor of N and Vorbis COLA window gives correct round-trip gain.
        // This matches DFN3 and Python irfft_matrix = fft_size * window × basis.
        let mut frame = Array1::<f32>::zeros(self.fft_size);
        for i in 0..self.fft_size {
            frame[i] = output[i] * self.window[i];
        }

        // Overlap-add: first `overlap` samples add to the synthesis buffer,
        // then emit the first `hop_size` samples.
        let mut out = Array1::<f32>::zeros(self.hop_size);
        for i in 0..overlap {
            out[i] = self.synthesis_mem[i] + frame[i];
        }
        // If hop_size > overlap, fill the rest directly from frame
        for i in overlap..self.hop_size {
            out[i] = frame[i];
        }

        // Update synthesis_mem: the tail of the current frame
        for i in 0..overlap {
            self.synthesis_mem[i] = frame[self.hop_size + i];
        }

        out
    }

    // ── Feature extraction ─────────────────────────────────────────────────────

    /// ERB log-power features with running-mean normalisation.
    ///
    /// Matches `DFState::feat_erb` in libDF:
    ///   1. Power spectrum: |X[k]|² per bin.
    ///   2. ERB-band energy: matmul with analysis filterbank.
    ///   3. Log-power: 10·log10(energy + ε).
    ///   4. Exponential running-mean normalisation, divided by 40.
    pub fn extract_erb_features(
        &mut self,
        spec_real: &Array1<f32>,
        spec_imag: &Array1<f32>,
    ) -> Array1<f32> {
        // Power per frequency bin
        let power = spec_real.mapv(|r| r * r) + spec_imag.mapv(|i| i * i);
        // ERB-band energy: [n_freqs] · [n_freqs, nb_erb] → [nb_erb]
        let erb_power = self.erb_fb.t().dot(&power);
        // Log-power in dB
        let erb_db = erb_power.mapv(|p| 10.0 * (p + 1e-10).log10());

        // Running-mean normalisation (exponential smoothing)
        // DFN3 and Python: update state FIRST, then subtract NEW state.
        let new_erb_state = &erb_db * self.one_minus_alpha + &self.erb_state * self.alpha;
        let feat = (&erb_db - &new_erb_state) / 40.0;
        self.erb_state = new_erb_state;

        feat
    }

    /// Complex-spectrum unit-norm features.
    ///
    /// Matches `DFState::feat_cplx` in libDF:
    ///   1. Take first `nb_spec` bins.
    ///   2. Running exponential mean of |X[k]| (amplitude, not power).
    ///   3. Divide complex spectrum by sqrt(state) → normalised amplitude ≈ 1.
    pub fn extract_spec_features(
        &mut self,
        spec_real: &Array1<f32>,
        spec_imag: &Array1<f32>,
    ) -> (Array1<f32>, Array1<f32>) {
        let spec_r = spec_real.slice(s![..self.nb_spec]).to_owned();
        let spec_i = spec_imag.slice(s![..self.nb_spec]).to_owned();

        // Amplitude per bin
        let amp = (&spec_r * &spec_r + &spec_i * &spec_i).mapv(|x| x.sqrt());

        // Running mean of amplitude — clamp state (matches training's .clamp(min=1e-10))
        self.unit_state = (&amp * self.one_minus_alpha + &self.unit_state * self.alpha)
            .mapv(|x| x.max(1e-10));

        // Normalise: divide by sqrt(clamped state)
        let norm = self.unit_state.mapv(|x| x.sqrt());
        let feat_r = &spec_r / &norm;
        let feat_i = &spec_i / &norm;

        (feat_r, feat_i)
    }

    // ── Tensor packing ─────────────────────────────────────────────────────────

    /// Pack features into model input tensors.
    ///
    /// Returns:
    ///   `spec`         [1, 1, 1, n_freqs, 2]   — full noisy complex spectrum
    ///   `feat_erb_4d`  [1, 1, 1, nb_erb]        — ERB features
    ///   `feat_spec_5d` [1, 1, 1, nb_spec, 2]   — complex spec features
    pub fn pack_features(
        &self,
        spec_real: &Array1<f32>,
        spec_imag: &Array1<f32>,
        feat_erb: &Array1<f32>,
        feat_spec_real: &Array1<f32>,
        feat_spec_imag: &Array1<f32>,
    ) -> (Array5<f32>, Array4<f32>, Array5<f32>) {
        let n_freqs = self.fft_size / 2 + 1;

        let mut spec = Array5::<f32>::zeros((1, 1, 1, n_freqs, 2));
        for i in 0..n_freqs {
            spec[[0, 0, 0, i, 0]] = spec_real[i];
            spec[[0, 0, 0, i, 1]] = spec_imag[i];
        }

        let mut feat_erb_4d = Array4::<f32>::zeros((1, 1, 1, self.nb_erb));
        for i in 0..self.nb_erb {
            feat_erb_4d[[0, 0, 0, i]] = feat_erb[i];
        }

        let mut feat_spec_5d = Array5::<f32>::zeros((1, 1, 1, self.nb_spec, 2));
        for i in 0..self.nb_spec {
            feat_spec_5d[[0, 0, 0, i, 0]] = feat_spec_real[i];
            feat_spec_5d[[0, 0, 0, i, 1]] = feat_spec_imag[i];
        }

        (spec, feat_erb_4d, feat_spec_5d)
    }

    // ── Reset ──────────────────────────────────────────────────────────────────

    pub fn reset(&mut self) {
        self.analysis_mem.fill(0.0);
        self.synthesis_mem.fill(0.0);

        let nb_erb = self.nb_erb;
        let nb_spec = self.nb_spec;
        for i in 0..nb_erb {
            self.erb_state[i] =
                -60.0 - (30.0 * i as f32) / (nb_erb.saturating_sub(1).max(1)) as f32;
        }
        for i in 0..nb_spec {
            self.unit_state[i] =
                0.001 - (0.0009 * i as f32) / (nb_spec.saturating_sub(1).max(1)) as f32;
        }
    }
}
