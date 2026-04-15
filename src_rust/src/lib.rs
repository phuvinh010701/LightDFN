pub mod features;
pub mod ligru;
pub mod model;
pub mod model_native;
pub mod streaming;

use js_sys::Float32Array;
use ndarray::Array2;
use streaming::StreamingProcessor;
use model_native::StreamingProcessorNative;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub struct LightDFNWasm {
    processor: StreamingProcessor,
    frame_count: usize,
    out_buf: Vec<f32>,
}

#[wasm_bindgen]
impl LightDFNWasm {
    /// Create new instance (pulsed ONNX variant — GRU states as explicit tensor I/O).
    ///
    /// This uses the three models produced by `export_pulsed_onnx.py`, which embed
    /// Li-GRU as standard ops with hidden states as explicit inputs/outputs.
    /// This matches DeepFilterNet3's architecture and allows tract to pre-allocate
    /// all intermediate buffers, achieving zero per-frame heap allocations for the GRU.
    ///
    /// Args:
    ///   enc_bytes      — enc.onnx bytes       (encoder with GRU state I/O)
    ///   erb_dec_bytes  — erb_dec.onnx bytes   (ERB decoder with GRU state I/O)
    ///   df_dec_bytes   — df_dec.onnx bytes    (DF decoder with GRU state I/O)
    ///   erb_fb_json    — ERB filterbank JSON string
    ///   enc_layers     — number of encoder Li-GRU layers (usually 1)
    ///   erb_layers     — number of ERB decoder Li-GRU layers (usually 2)
    ///   df_layers      — number of DF decoder Li-GRU layers (usually 2)
    #[wasm_bindgen(constructor)]
    pub fn new(
        enc_bytes: &[u8],
        erb_dec_bytes: &[u8],
        df_dec_bytes: &[u8],
        erb_fb_json: &str,
        enc_layers: usize,
        erb_layers: usize,
        df_layers: usize,
    ) -> Result<LightDFNWasm, JsValue> {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        console_log!(
            "[WASM] Initializing LightDFN pulsed (enc={}KB erb_dec={}KB df_dec={}KB)…",
            enc_bytes.len() / 1024,
            erb_dec_bytes.len() / 1024,
            df_dec_bytes.len() / 1024,
        );

        let erb_data: serde_json::Value = serde_json::from_str(erb_fb_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse ERB JSON: {}", e)))?;

        // Forward ERB filterbank [N_FREQS, NB_ERB] — used by FeatureExtractor
        let erb_fb_vec: Vec<Vec<f32>> = serde_json::from_value(erb_data["erb_fb"].clone())
            .map_err(|e| JsValue::from_str(&format!("Failed to parse erb_fb: {}", e)))?;
        let n_freqs = erb_fb_vec.len();
        let nb_erb = erb_fb_vec[0].len();
        let mut erb_fb = Array2::<f32>::zeros((n_freqs, nb_erb));
        for (i, row) in erb_fb_vec.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                erb_fb[[i, j]] = val;
            }
        }

        // Inverse ERB filterbank [NB_ERB, N_FREQS] — maps ERB gains → per-frequency gains
        let erb_inv_fb_vec: Vec<Vec<f32>> =
            serde_json::from_value(erb_data["erb_inv_fb"].clone())
                .map_err(|e| JsValue::from_str(&format!("Failed to parse erb_inv_fb: {}", e)))?;
        let nb_erb2 = erb_inv_fb_vec.len();
        let n_freqs2 = erb_inv_fb_vec[0].len();
        let mut erb_inv_fb = Array2::<f32>::zeros((nb_erb2, n_freqs2));
        for (i, row) in erb_inv_fb_vec.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                erb_inv_fb[[i, j]] = val;
            }
        }

        console_log!(
            "[WASM] ERB filterbank: [{}, {}]  inv: [{}, {}]  layers: enc={} erb={} df={}",
            n_freqs, nb_erb, nb_erb2, n_freqs2, enc_layers, erb_layers, df_layers
        );

        let processor = StreamingProcessor::new(
            enc_bytes,
            erb_dec_bytes,
            df_dec_bytes,
            erb_fb,
            erb_inv_fb,
            enc_layers,
            erb_layers,
            df_layers,
        )
        .map_err(|e| JsValue::from_str(&format!("Failed to initialize: {}", e)))?;

        console_log!("[WASM] ✓ LightDFN pulsed ready");

        Ok(LightDFNWasm {
            processor,
            frame_count: 0,
            out_buf: vec![0.0f32; 480],
        })
    }

    /// Process one audio frame (480 samples @ 48 kHz).
    /// Returns a Float32Array (view into WASM linear memory, zero-copy on most JS engines).
    #[wasm_bindgen]
    pub fn process_frame(&mut self, audio_frame: &[f32]) -> Result<Float32Array, JsValue> {
        if audio_frame.len() != 480 {
            return Err(JsValue::from_str(&format!(
                "Expected 480 samples, got {}",
                audio_frame.len()
            )));
        }

        let audio_array = ndarray::ArrayView1::from(audio_frame);
        let enhanced = self
            .processor
            .process_frame(audio_array)
            .map_err(|e| JsValue::from_str(&format!("Processing failed: {}", e)))?;

        let slice = enhanced.as_slice().expect("ndarray not contiguous");
        self.out_buf.copy_from_slice(slice);

        self.frame_count += 1;
        if self.frame_count % 500 == 0 {
            console_log!("[WASM] {} frames processed", self.frame_count);
        }

        Ok(Float32Array::from(self.out_buf.as_slice()))
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.processor.reset();
        self.frame_count = 0;
        console_log!("[WASM] States reset");
    }

    #[wasm_bindgen]
    pub fn get_frame_count(&self) -> usize {
        self.frame_count
    }
}

// ─── Native Li-GRU WASM binding ──────────────────────────────────────────────

/// WASM binding for the native Rust Li-GRU path.
///
/// Uses feedforward-only ONNX models (`enc_ff.onnx`, `erb_dec_ff.onnx`,
/// `df_dec_ff.onnx`) produced by `export_native_onnx.py`, plus a
/// `ligru_weights.json` file that carries all GRU weight matrices.
///
/// The GRU runs entirely in Rust — no tract op dispatch per cell step.
/// When the model is trained with BatchNorm (`ligru_normalization="batchnorm"`),
/// the running stats are folded into per-element scale/offset at export time,
/// so normalisation costs virtually nothing at inference.
#[wasm_bindgen]
pub struct LightDFNNativeWasm {
    processor: StreamingProcessorNative,
    frame_count: usize,
    out_buf: Vec<f32>,
}

#[wasm_bindgen]
impl LightDFNNativeWasm {
    /// Create a new native Li-GRU instance.
    ///
    /// Args:
    ///   enc_ff_bytes     — enc_ff.onnx bytes      (conv encoder, no GRU)
    ///   erb_dec_ff_bytes — erb_dec_ff.onnx bytes  (ERB decoder convs, no GRU)
    ///   df_dec_ff_bytes  — df_dec_ff.onnx bytes   (DF decoder convs, no GRU)
    ///   ligru_json       — ligru_weights.json text (all GRU weight matrices)
    ///   erb_fb_json      — ERB filterbank JSON string (same format as pulsed path)
    #[wasm_bindgen(constructor)]
    pub fn new(
        enc_ff_bytes: &[u8],
        erb_dec_ff_bytes: &[u8],
        df_dec_ff_bytes: &[u8],
        ligru_json: &str,
        erb_fb_json: &str,
    ) -> Result<LightDFNNativeWasm, JsValue> {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        console_log!(
            "[WASM-native] Initializing LightDFN native (enc={}KB erb={}KB df={}KB gru={}KB)…",
            enc_ff_bytes.len() / 1024,
            erb_dec_ff_bytes.len() / 1024,
            df_dec_ff_bytes.len() / 1024,
            ligru_json.len() / 1024,
        );

        let erb_data: serde_json::Value = serde_json::from_str(erb_fb_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse ERB JSON: {}", e)))?;

        let erb_fb_vec: Vec<Vec<f32>> = serde_json::from_value(erb_data["erb_fb"].clone())
            .map_err(|e| JsValue::from_str(&format!("Failed to parse erb_fb: {}", e)))?;
        let n_freqs = erb_fb_vec.len();
        let nb_erb = erb_fb_vec[0].len();
        let mut erb_fb = Array2::<f32>::zeros((n_freqs, nb_erb));
        for (i, row) in erb_fb_vec.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                erb_fb[[i, j]] = val;
            }
        }

        let erb_inv_fb_vec: Vec<Vec<f32>> =
            serde_json::from_value(erb_data["erb_inv_fb"].clone())
                .map_err(|e| JsValue::from_str(&format!("Failed to parse erb_inv_fb: {}", e)))?;
        let nb_erb2 = erb_inv_fb_vec.len();
        let n_freqs2 = erb_inv_fb_vec[0].len();
        let mut erb_inv_fb = Array2::<f32>::zeros((nb_erb2, n_freqs2));
        for (i, row) in erb_inv_fb_vec.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                erb_inv_fb[[i, j]] = val;
            }
        }

        console_log!(
            "[WASM-native] ERB: [{}, {}]  inv: [{}, {}]",
            n_freqs, nb_erb, nb_erb2, n_freqs2
        );

        let processor = StreamingProcessorNative::new(
            enc_ff_bytes,
            erb_dec_ff_bytes,
            df_dec_ff_bytes,
            ligru_json,
            erb_fb,
            erb_inv_fb,
        )
        .map_err(|e| JsValue::from_str(&format!("Failed to initialize native model: {}", e)))?;

        console_log!("[WASM-native] ✓ LightDFN native ready");

        Ok(LightDFNNativeWasm {
            processor,
            frame_count: 0,
            out_buf: vec![0.0f32; 480],
        })
    }

    /// Process one audio frame (480 samples @ 48 kHz).
    #[wasm_bindgen]
    pub fn process_frame(&mut self, audio_frame: &[f32]) -> Result<Float32Array, JsValue> {
        if audio_frame.len() != 480 {
            return Err(JsValue::from_str(&format!(
                "Expected 480 samples, got {}",
                audio_frame.len()
            )));
        }

        let audio_array = ndarray::ArrayView1::from(audio_frame);
        let enhanced = self
            .processor
            .process_frame(audio_array)
            .map_err(|e| JsValue::from_str(&format!("Native processing failed: {}", e)))?;

        let slice = enhanced.as_slice().expect("ndarray not contiguous");
        self.out_buf.copy_from_slice(slice);

        self.frame_count += 1;
        if self.frame_count % 500 == 0 {
            console_log!("[WASM-native] {} frames processed", self.frame_count);
        }

        Ok(Float32Array::from(self.out_buf.as_slice()))
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.processor.reset();
        self.frame_count = 0;
        console_log!("[WASM-native] States reset");
    }

    #[wasm_bindgen]
    pub fn get_frame_count(&self) -> usize {
        self.frame_count
    }
}
