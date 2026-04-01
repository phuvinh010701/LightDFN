/// Convert streaming_best_inline.onnx → streaming_best.nnef.tar
///
/// This runs the expensive `into_typed()` graph analysis natively and serialises
/// the result as a tract-NNEF archive.  The WASM build then loads the NNEF
/// archive with `tract_nnef::nnef().model_for_read()` which is a simple
/// deserialization step — no recursive graph analysis, no call-stack overflow.
///
/// Usage:
///   cargo run --bin convert_nnef --release
///   # → writes  ../checkpoints/streaming_best.nnef.tar

use std::fs::File;
use tract_nnef::prelude::*;
use tract_onnx::prelude::InferenceModelExt;
use tract_onnx::WithOnnx;

fn main() -> anyhow::Result<()> {
    let onnx_path = "../checkpoints/streaming_best_inline.onnx";
    let nnef_path = "../checkpoints/streaming_best.nnef.tar";

    eprintln!("Loading ONNX model from: {onnx_path}");
    let typed_model = tract_onnx::onnx()
        .model_for_path(onnx_path)?
        .into_typed()?;

    // Debug: print all op names to identify non-serializable ones
    eprintln!("Nodes in typed model:");
    for node in typed_model.nodes() {
        eprintln!("  #{} {:?} op={}", node.id, node.name, node.op().name());
    }

    eprintln!("Saving NNEF archive to: {nnef_path}");
    let out_file = File::create(nnef_path)?;
    tract_nnef::nnef()
        .with_tract_core()
        .with_onnx()
        .write_to_tar(&typed_model, out_file)?;

    eprintln!("Done! {nnef_path} written.");
    Ok(())
}
