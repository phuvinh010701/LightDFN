//! ERB (Equivalent Rectangular Bandwidth) filterbank.
//!
//! Exact port of `src/utils/erb.py`.  The filterbank is a [NB_FREQS × NB_ERB]
//! sparse rectangular matrix: each frequency bin belongs to exactly one ERB band.
//! Storing it as a flat `(band_index, weight)` per bin gives O(NB_FREQS) apply
//! instead of O(NB_FREQS × NB_ERB) and is cache-friendly.
//!
//! Constants match `src/configs/default.yaml`:
//!   sr=48000, fft_size=960, nb_erb=32, min_nb_freqs=2.

pub const NB_FREQS: usize = 481; // fft_size/2 + 1
pub const NB_ERB: usize = 32;

const SR: u32 = 48_000;
const FFT_SIZE: usize = 960;
const MIN_NB_FREQS: usize = 2;

/// Python: `freq2erb(f) = 9.265 * log1p(f / (24.7 * 9.265))`
#[inline]
fn freq2erb(freq_hz: f64) -> f64 {
    9.265 * (freq_hz / (24.7 * 9.265)).ln_1p()
}

/// Python: `erb2freq(n) = 24.7 * 9.265 * (exp(n / 9.265) - 1)`
#[inline]
fn erb2freq(n_erb: f64) -> f64 {
    24.7 * 9.265 * (n_erb / 9.265_f64).exp_m1()
}

/// Compute band widths (number of FFT bins per ERB band).
/// Exact port of `erb_fb_widths(sr, fft_size, nb_bands, min_nb_freqs)`.
fn erb_fb_widths() -> [usize; NB_ERB] {
    let nyq_freq = (SR / 2) as f64; // 24 000 Hz
    let freq_width = SR as f64 / FFT_SIZE as f64; // 50 Hz / bin
    let erb_low = freq2erb(0.0); // ≈ 0
    let erb_high = freq2erb(nyq_freq);

    let step = (erb_high - erb_low) / NB_ERB as f64;

    let mut widths = [0usize; NB_ERB];
    let mut prev_freq: i32 = 0;
    let mut freq_over: i32 = 0;

    for i in 1..=(NB_ERB as i32) {
        let f = erb2freq(erb_low + i as f64 * step);
        let fb = (f / freq_width).round() as i32;
        let nb_freqs = fb - prev_freq - freq_over;

        let (nf_final, new_freq_over) = if nb_freqs < MIN_NB_FREQS as i32 {
            (MIN_NB_FREQS as i32, MIN_NB_FREQS as i32 - nb_freqs)
        } else {
            (nb_freqs, 0)
        };

        freq_over = new_freq_over;
        widths[(i - 1) as usize] = nf_final as usize;
        prev_freq = fb;
    }

    // Python adds 1 to the last band, then trims if total > NB_FREQS.
    widths[NB_ERB - 1] += 1;
    let total: usize = widths.iter().sum();
    if total > NB_FREQS {
        widths[NB_ERB - 1] -= total - NB_FREQS;
    }

    debug_assert_eq!(
        widths.iter().sum::<usize>(),
        NB_FREQS,
        "ERB widths must sum to NB_FREQS"
    );
    widths
}

/// Compact representation of the normalized ERB filterbank.
///
/// For every FFT bin `f` we store:
/// - `band[f]`   — the ERB band index it maps to  (u8, fits in 0..32)
/// - `weight[f]` — `1 / band_width`  (normalised contribution)
///
/// Apply: `erb_out[band[f]] += power[f] * weight[f]`
pub struct ErbFilterbank {
    pub band: [u8; NB_FREQS],
    pub weight: [f32; NB_FREQS],
}

impl ErbFilterbank {
    /// Build the filterbank. Called once at processor initialisation.
    pub fn new() -> Self {
        let widths = erb_fb_widths();
        let mut band = [0u8; NB_FREQS];
        let mut weight = [0.0f32; NB_FREQS];

        let mut pos = 0usize;
        for (b, &w) in widths.iter().enumerate() {
            let inv_w = 1.0 / w as f32;
            for f in pos..(pos + w) {
                band[f] = b as u8;
                weight[f] = inv_w;
            }
            pos += w;
        }

        ErbFilterbank { band, weight }
    }

    /// Apply ERB filterbank: `power[NB_FREQS]` → `erb_power[NB_ERB]`.
    ///
    /// Equivalent to `erb_power = power @ erb_fb` in Python.
    #[inline(always)]
    pub fn apply(&self, power: &[f32; NB_FREQS], out: &mut [f32; NB_ERB]) {
        out.fill(0.0);
        for f in 0..NB_FREQS {
            // Safety: band[f] < NB_ERB by construction.
            unsafe {
                *out.get_unchecked_mut(self.band[f] as usize) +=
                    *power.get_unchecked(f) * self.weight[f];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_sum_to_nb_freqs() {
        let w = erb_fb_widths();
        assert_eq!(w.iter().sum::<usize>(), NB_FREQS);
    }

    #[test]
    fn filterbank_covers_all_bins() {
        let fb = ErbFilterbank::new();
        // Every bin must have a non-zero weight
        for f in 0..NB_FREQS {
            assert!(fb.weight[f] > 0.0, "bin {} has zero weight", f);
        }
    }

    #[test]
    fn filterbank_unit_power() {
        // All-ones power spectrum → each band receives exactly 1.0
        // (since sum of weights in each band = 1/w * w = 1).
        let fb = ErbFilterbank::new();
        let power = [1.0f32; NB_FREQS];
        let mut out = [0.0f32; NB_ERB];
        fb.apply(&power, &mut out);
        for (b, &v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-5, "band {} expected 1.0 got {}", b, v);
        }
    }
}
