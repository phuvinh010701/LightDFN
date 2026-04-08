//! Streaming STFT / ISTFT using `realfft`.
//!
//! ## Analysis (STFT)
//! Called once per hop (480 new samples).  Maintains an internal 480-sample
//! overlap buffer so the caller only supplies the *new* hop each time.
//!
//! Window: periodic Hann of length 960, matching `torch.hann_window(960)`.
//!   `w[n] = 0.5 * (1 - cos(2π·n/960))`
//!
//! ## Synthesis (ISTFT)
//! Streaming overlap-add with a per-sample synthesis window that ensures
//! perfect reconstruction under the Hann / 50%-overlap COLA condition:
//!   `synth_win[n] = hann[n] / (hann[n]^2 + hann[(n ± hop) % N]^2) / N`
//!
//! The `/N` folds in the `realfft` normalisation so the hot path is just:
//!   1. IRFFT → y[960]
//!   2. y *= synth_win
//!   3. output = ola_buf + y[0..480]
//!   4. ola_buf = y[480..960]

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const FFT_SIZE: usize = 960;
pub const HOP_SIZE: usize = 480; // = FFT_SIZE / 2
pub const NB_FREQS: usize = FFT_SIZE / 2 + 1; // 481

pub type Complex32 = realfft::num_complex::Complex<f32>;

/// STFT/ISTFT state for one-hop-at-a-time streaming.
pub struct StreamingStft {
    // FFT plans (Arc, cheap to clone; plans are cached inside realfft)
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,

    // Pre-allocated scratch buffers (reused every frame, no heap allocation in hot path)
    fft_in: Vec<f32>,        // [FFT_SIZE] — analysis window × audio frame
    fft_out: Vec<Complex32>, // [NB_FREQS] — spectrum
    ifft_in: Vec<Complex32>, // [NB_FREQS] — copy of enhanced spectrum (c2r mutates input)
    ifft_out: Vec<f32>,      // [FFT_SIZE] — reconstructed time-domain frame

    // Analysis
    hann: [f32; FFT_SIZE],
    audio_overlap: [f32; HOP_SIZE], // last HOP_SIZE samples of the previous call

    // Synthesis
    synth_win: [f32; FFT_SIZE], // hann / (N * OLA_norm[n])
    ola_buf: [f32; HOP_SIZE],   // second-half of the previous windowed IRFFT frame
}

impl StreamingStft {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(FFT_SIZE);
        let c2r = planner.plan_fft_inverse(FFT_SIZE);

        let fft_in = r2c.make_input_vec();
        let fft_out = r2c.make_output_vec();
        let ifft_in = c2r.make_input_vec();
        let ifft_out = c2r.make_output_vec();

        // Periodic Hann: w[n] = 0.5 * (1 - cos(2π·n/N))
        // Matches torch.hann_window(N, periodic=True).
        let mut hann = [0.0f32; FFT_SIZE];
        for (n, h) in hann.iter_mut().enumerate() {
            *h = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / FFT_SIZE as f32).cos());
        }

        // Synthesis window: hann[n] / (N * (hann[n]^2 + hann[(n+hop) % N]^2))
        //
        // Derivation:
        //   IRFFT output is N×signal (unnormalised).  After dividing by N we get
        //   y[n] = x[n] * hann[n] (ideal case, no noise).
        //   OLA with 50% overlap needs: ∑_k hann[n-kH] * synth[n-kH] = 1
        //   → synth[n] = hann[n] / ∑_k hann[n-kH]^2
        //              = hann[n] / (hann[n]^2 + hann[n+H]^2)   (50% overlap, 2 terms)
        //   Folding the 1/N factor: synth_win[n] = hann[n] / (N * (hann[n]^2 + hann[n+H]^2))
        let mut synth_win = [0.0f32; FFT_SIZE];
        for n in 0..FFT_SIZE {
            let h0 = hann[n];
            let h1 = hann[(n + HOP_SIZE) % FFT_SIZE];
            let ola_norm = h0 * h0 + h1 * h1;
            synth_win[n] = if ola_norm > 1e-10 {
                h0 / (FFT_SIZE as f32 * ola_norm)
            } else {
                0.0
            };
        }

        StreamingStft {
            r2c,
            c2r,
            fft_in,
            fft_out,
            ifft_in,
            ifft_out,
            hann,
            audio_overlap: [0.0; HOP_SIZE],
            synth_win,
            ola_buf: [0.0; HOP_SIZE],
        }
    }

    /// STFT: consume `hop` new samples, return reference to the spectrum buffer.
    ///
    /// The spectrum has `NB_FREQS` (481) complex bins ordered DC → Nyquist.
    /// Internally this function:
    ///   1. Builds `[audio_overlap | new_hop]` (960 samples)
    ///   2. Multiplies by the Hann window in-place
    ///   3. Computes the real-to-complex FFT
    ///
    /// The caller must *not* hold references across the next call.
    pub fn forward(&mut self, new_hop: &[f32; HOP_SIZE]) -> &[Complex32; NB_FREQS] {
        // 1. Fill FFT input: overlap + new hop
        self.fft_in[..HOP_SIZE].copy_from_slice(&self.audio_overlap);
        self.fft_in[HOP_SIZE..].copy_from_slice(new_hop);

        // 2. Update overlap buffer for next frame
        self.audio_overlap.copy_from_slice(new_hop);

        // 3. Apply Hann window in-place
        for (s, &w) in self.fft_in.iter_mut().zip(self.hann.iter()) {
            *s *= w;
        }

        // 4. R2C FFT — overwrites fft_out
        self.r2c
            .process(&mut self.fft_in, &mut self.fft_out)
            .expect("R2C FFT failed");

        // SAFETY: NB_FREQS == fft_out.len() by construction from make_output_vec()
        unsafe { &*(self.fft_out.as_ptr() as *const [Complex32; NB_FREQS]) }
    }

    /// ISTFT: overlap-add one enhanced spectrum frame, return 480 output samples.
    ///
    /// `enhanced_spec` must contain exactly NB_FREQS complex values ordered DC → Nyquist,
    /// matching the layout from `forward()`.
    ///
    /// This function:
    ///   1. Copies the spectrum into the mutable scratch buffer (c2r mutates input)
    ///   2. Computes IRFFT (unnormalised, scale = N)
    ///   3. Multiplies by synth_win (which folds in the 1/N factor)
    ///   4. Outputs `ola_buf + y[0..HOP_SIZE]`, saves `y[HOP_SIZE..]` as new ola_buf
    pub fn inverse(&mut self, enhanced_spec: &[Complex32; NB_FREQS], out: &mut [f32; HOP_SIZE]) {
        // 1. Copy spectrum into mutable scratch (c2r overwrites input)
        self.ifft_in.copy_from_slice(enhanced_spec);

        // 2. C2R IRFFT — result is unnormalised (scale = FFT_SIZE)
        self.c2r
            .process(&mut self.ifft_in, &mut self.ifft_out)
            .expect("C2R IFFT failed");

        // 3. Apply synthesis window (folds in 1/N normalisation + OLA correction)
        for (s, &w) in self.ifft_out.iter_mut().zip(self.synth_win.iter()) {
            *s *= w;
        }

        // 4. Overlap-add: output = ola_buf + first half of windowed frame
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.ola_buf[i] + self.ifft_out[i];
        }

        // 5. Save second half as new OLA buffer for the next frame
        self.ola_buf
            .copy_from_slice(&self.ifft_out[HOP_SIZE..FFT_SIZE]);
    }

    /// Reset all streaming state (overlap, OLA buffer).
    pub fn reset(&mut self) {
        self.audio_overlap.fill(0.0);
        self.ola_buf.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify near-perfect reconstruction on a synthetic sinusoid.
    /// Analysis delay = HOP_SIZE samples (the initial overlap buffer is zero).
    #[test]
    fn roundtrip_sinusoid() {
        let mut stft = StreamingStft::new();

        let sr = 48_000.0f32;
        let freq = 1_000.0f32;
        let n_hops = 20usize;

        // Generate signal: length = n_hops * HOP_SIZE
        let signal: Vec<f32> = (0..n_hops * HOP_SIZE)
            .map(|n| (2.0 * std::f32::consts::PI * freq * n as f32 / sr).sin())
            .collect();

        // Encode + decode frame by frame
        let mut reconstructed = vec![0.0f32; n_hops * HOP_SIZE];
        for hop_idx in 0..n_hops {
            let start = hop_idx * HOP_SIZE;
            let hop: &[f32; HOP_SIZE] = signal[start..start + HOP_SIZE].try_into().unwrap();

            let spec = stft.forward(hop).to_owned(); // copy spectrum
            let spec_arr: &[Complex32; NB_FREQS] = &spec.try_into().unwrap();

            let mut out = [0.0f32; HOP_SIZE];
            stft.inverse(spec_arr, &mut out);
            reconstructed[start..start + HOP_SIZE].copy_from_slice(&out);
        }

        // Skip the first HOP_SIZE samples (startup transient from zero overlap buffer).
        // All subsequent frames should reconstruct within floating-point tolerance.
        let check_start = HOP_SIZE;
        let check_end = (n_hops - 1) * HOP_SIZE;

        let max_err = signal[check_start..check_end]
            .iter()
            .zip(&reconstructed[check_start..check_end])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Threshold: 2e-4 ≈ -74 dBFS, well below audio noise floor.
        assert!(
            max_err < 2e-4,
            "STFT roundtrip error too large: {:.2e}",
            max_err
        );
    }
}
