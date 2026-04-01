use wasm_bindgen::prelude::*;

mod dsp;
mod erb;
mod stft;

// Full Rust inference via tract — DSP + ONNX inference in a single WASM call.
mod processor;

pub use dsp::LightDFNDSP;
pub use processor::LightDFNProcessor;

/// Initialise the WASM module — called automatically by the generated JS glue.
/// Sets up a human-readable panic hook so Rust panics appear in the browser console.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
