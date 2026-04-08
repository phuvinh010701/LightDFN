use wasm_bindgen::prelude::*;

mod dsp;
mod erb;
mod stft;

// Full Rust inference via tract — DSP + ONNX inference in a single WASM call.
mod processor;

pub use dsp::LightDFNDSP;
pub use processor::LightDFNProcessor;

#[wasm_bindgen]
pub struct LightDFNTract {
    inner: LightDFNProcessor,
}

#[wasm_bindgen]
impl LightDFNTract {
    #[wasm_bindgen(constructor)]
    pub fn new(
        enc_onnx: &[u8],
        erb_dec_onnx: &[u8],
        df_dec_onnx: &[u8],
    ) -> Result<LightDFNTract, JsValue> {
        let inner = LightDFNProcessor::from_onnx_bytes(enc_onnx, erb_dec_onnx, df_dec_onnx)
            .map_err(|e| JsValue::from_str(&format!("Failed to init tract models: {e}")))?;
        Ok(LightDFNTract { inner })
    }

    pub fn process_frame(&mut self, input: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.inner
            .process_frame(input)
            .map_err(|e| JsValue::from_str(&format!("process_frame failed: {e}")))
    }

    pub fn reset(&mut self) -> Result<(), JsValue> {
        self.inner
            .reset()
            .map_err(|e| JsValue::from_str(&format!("reset failed: {e}")))
    }
}

/// Initialise the WASM module — called automatically by the generated JS glue.
/// Sets up a human-readable panic hook so Rust panics appear in the browser console.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
