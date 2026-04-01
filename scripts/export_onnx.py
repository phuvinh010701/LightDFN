#!/usr/bin/env python3
"""Export a LightDeepFilterNet checkpoint to ONNX format, with optional post-export optimizations.

The exported model takes three inputs (spec, feat_erb, feat_spec) and returns
four outputs (enhanced_spec, erb_mask, lsnr, df_coefs), matching the PyTorch
forward signature exactly.

Example:
  uv run python -m scripts.export_onnx \\
    --ckpt checkpoints/best.pt \\
    --out checkpoints/lightdfn.onnx \\
    --config src/configs/default.yaml

  # With onnxruntime verification:
  uv run python -m scripts.export_onnx \\
    --ckpt checkpoints/best.pt \\
    --out checkpoints/lightdfn.onnx \\
    --verify

  # With onnx-simplifier + graph optimization + ORT format:
  uv run python -m scripts.export_onnx \\
    --ckpt checkpoints/best.pt \\
    --out checkpoints/lightdfn.ort \\
    --simplify --graph-opt --to-ort

  # FP16 for WebGPU:
  uv run python -m scripts.export_onnx \\
    --ckpt checkpoints/best.pt \\
    --out checkpoints/lightdfn_fp16.onnx \\
    --fp16
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.configs.config import load_config
from src.model.lightdeepfilternet import init_model
from src.model.streaming import StreamingLightDFN
from src.utils.io import get_device

logger = logging.getLogger(__name__)


def build_dummy_inputs(
    model_cfg,
    loader_cfg,
    T: int = 512,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dummy input tensors matching the model's expected shapes.

    Args:
        model_cfg: ModelConfig with fft_size, nb_erb, etc.
        loader_cfg: DataLoaderConfig with nb_spec.
        T: Number of time frames to export. The ONNX model will only accept
           exactly this many frames. Inference on longer audio should chunk
           into T-frame segments. Default: 512 (~5 s at 48 kHz / hop 480).
        device: Target device for tensors.

    Returns:
        (spec, feat_erb, feat_spec) dummy tensors.
    """
    B = 1
    F = model_cfg.fft_size // 2 + 1  # 481
    E = model_cfg.nb_erb  # 32
    Fc = loader_cfg.nb_spec  # 96

    spec = torch.randn(B, 1, T, F, 2, device=device)
    feat_erb = torch.randn(B, 1, T, E, device=device)
    feat_spec = torch.randn(B, 1, T, Fc, 2, device=device)
    return spec, feat_erb, feat_spec


def optimize_graph(input_path: Path, output_path: Path) -> None:
    """Apply ONNX graph optimizations (constant folding, operator fusion, etc.)."""
    import onnxruntime as ort

    logger.info("Applying graph optimizations...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(output_path)
    _ = ort.InferenceSession(str(input_path), sess_options)

    original_size = input_path.stat().st_size / 1024 / 1024
    optimized_size = output_path.stat().st_size / 1024 / 1024
    reduction = (1 - optimized_size / original_size) * 100
    logger.info(
        "Graph opt: %.2f MB → %.2f MB (%.1f%% reduction)",
        original_size,
        optimized_size,
        reduction,
    )


def convert_fp16(input_path: Path, output_path: Path) -> None:
    """Convert model to FP16 precision for GPU/WebGPU backends (~2x size reduction)."""
    logger.info("Converting to FP16...")
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        model = onnx.load(str(input_path))
        model_fp16 = convert_float_to_float16(
            model, keep_io_types=True, disable_shape_infer=False
        )
        onnx.save(model_fp16, str(output_path))

        original_size = input_path.stat().st_size / 1024 / 1024
        fp16_size = output_path.stat().st_size / 1024 / 1024
        reduction = (1 - fp16_size / original_size) * 100
        logger.info(
            "FP16: %.2f MB → %.2f MB (%.1f%% reduction)",
            original_size,
            fp16_size,
            reduction,
        )
    except ImportError:
        logger.warning(
            "FP16 skipped: install onnxruntime-transformers (uv pip install onnxruntime-transformers)"
        )
        import shutil

        shutil.copy(input_path, output_path)


def convert_to_ort(input_path: Path, output_path: Path) -> None:
    """Convert ONNX model to ORT format (faster loading, optimized serialization)."""
    import onnxruntime as ort

    logger.info("Converting to ORT format...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(output_path)
    _ = ort.InferenceSession(str(input_path), sess_options)

    original_size = input_path.stat().st_size / 1024 / 1024
    ort_size = output_path.stat().st_size / 1024 / 1024
    reduction = (1 - ort_size / original_size) * 100
    logger.info(
        "ORT: %.2f MB → %.2f MB (%.1f%% reduction)",
        original_size,
        ort_size,
        reduction,
    )


def verify_model(model_path: Path, reference_path: Path | None = None) -> None:
    """Verify that the model runs and (optionally) matches a reference."""
    import numpy as np
    import onnxruntime as ort

    logger.info("Verifying model: %s", model_path)
    session = ort.InferenceSession(str(model_path))

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    logger.info("Inputs:  %s", input_names)
    logger.info("Outputs: %s", output_names)

    feeds = {}
    for inp in session.get_inputs():
        shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
        feeds[inp.name] = np.random.randn(*shape).astype(np.float32)

    outputs = session.run(None, feeds)
    logger.info("OK — output shapes: %s", [out.shape for out in outputs])

    if reference_path and reference_path.exists():
        ref_session = ort.InferenceSession(str(reference_path))
        ref_outputs = ref_session.run(None, feeds)
        for i, (out, ref_out) in enumerate(zip(outputs, ref_outputs)):
            max_diff = float(abs(out - ref_out).max())
            mean_diff = float(abs(out - ref_out).mean())
            logger.info(
                "Output %d: max_diff=%.6f, mean_diff=%.6f", i, max_diff, mean_diff
            )
            if max_diff > 1e-3:
                logger.warning("Large difference detected in output %d", i)


def export(
    ckpt_path: Path,
    out_path: Path,
    config_path: Path,
    chunk_frames: int = 512,
    streaming: bool = False,
    opset: int = 18,
    verify: bool = False,
    simplify: bool = False,
    graph_opt: bool = False,
    fp16: bool = False,
    to_ort: bool = False,
) -> None:
    model_cfg, _aug_cfg, loader_cfg, _train_cfg, _loss_cfg = load_config(
        str(config_path)
    )

    # Li-GRU requires batch_size at construction; use 1 for export.
    model_cfg.batch_size = 1

    device = get_device()
    model = init_model(model_cfg).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if streaming:
        _export_streaming(model, model_cfg, loader_cfg, out_path, opset, device)
    else:
        _export_chunked(
            model, model_cfg, loader_cfg, out_path, chunk_frames, opset, device
        )

    _postprocess(
        out_path,
        simplify=simplify,
        graph_opt=graph_opt,
        fp16=fp16,
        to_ort=to_ort,
        verify=verify,
    )


def _export_chunked(
    model, model_cfg, loader_cfg, out_path, chunk_frames, opset, device
):
    """Export fixed-chunk model (processes chunk_frames STFT frames per call)."""
    # NOTE: The LiGRU cell uses a Python for-loop over the time axis, which
    # forces T to be a static constant in the exported graph.  We therefore
    # export with a fixed chunk size; the inference script pads/chunks audio
    # to match this size at runtime.
    dummy_inputs = build_dummy_inputs(
        model_cfg, loader_cfg, T=chunk_frames, device=device
    )
    spec, feat_erb, feat_spec = dummy_inputs

    with torch.no_grad():
        out = model(spec, feat_erb, feat_spec)
    logger.info("PyTorch forward pass OK — outputs: %s", [list(o.shape) for o in out])
    logger.info(
        "Chunk size: %d frames (%.2f s at %d Hz)",
        chunk_frames,
        chunk_frames * model_cfg.hop_size / model_cfg.sr,
        model_cfg.sr,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting to %s (opset=%d) …", out_path, opset)
    torch.onnx.export(
        model,
        args=dummy_inputs,
        f=str(out_path),
        input_names=["spec", "feat_erb", "feat_spec"],
        output_names=["enhanced_spec", "erb_mask", "lsnr", "df_coefs"],
        opset_version=opset,
        export_params=True,
        dynamo=True,
        optimize=True,
        verify=False,
        keep_initializers_as_inputs=False,
    )
    logger.info("Export done.")


def _export_streaming(model, model_cfg, loader_cfg, out_path, opset, device):
    """Export streaming model (T=1 per call, hidden states + conv buffers as explicit I/O)."""
    wrapper = StreamingLightDFN(model)
    wrapper.eval()

    dummy_frame = build_dummy_inputs(model_cfg, loader_cfg, T=1, device=device)
    h_enc, h_erb, h_df, buf_erb0, buf_df0, buf_dfp, buf_spec = wrapper.init_states(
        batch_size=1, device=device
    )

    with torch.no_grad():
        out = wrapper(
            *dummy_frame, h_enc, h_erb, h_df, buf_erb0, buf_df0, buf_dfp, buf_spec
        )
    logger.info("Streaming forward pass OK — outputs: %s", [list(o.shape) for o in out])
    logger.info(
        "Streaming: 1 frame per call (%.1f ms per call at %d Hz)",
        model_cfg.hop_size / model_cfg.sr * 1000,
        model_cfg.sr,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting to %s (opset=%d, streaming=True) …", out_path, opset)
    torch.onnx.export(
        wrapper,
        args=(*dummy_frame, h_enc, h_erb, h_df, buf_erb0, buf_df0, buf_dfp, buf_spec),
        f=str(out_path),
        input_names=[
            "spec",
            "feat_erb",
            "feat_spec",
            "h_enc",
            "h_erb",
            "h_df",
            "buf_erb0",
            "buf_df0",
            "buf_dfp",
            "buf_spec",
        ],
        output_names=[
            "enhanced_spec",
            "h_enc_new",
            "h_erb_new",
            "h_df_new",
            "buf_erb0_new",
            "buf_df0_new",
            "buf_dfp_new",
            "buf_spec_new",
        ],
        opset_version=opset,
        export_params=True,
        dynamo=True,
        optimize=True,
        verify=False,
        keep_initializers_as_inputs=False,
    )
    logger.info("Export done.")


def _postprocess(
    out_path: Path,
    simplify: bool = False,
    graph_opt: bool = False,
    fp16: bool = False,
    to_ort: bool = False,
    verify: bool = False,
) -> None:
    # Optional: onnx-simplifier
    if simplify:
        try:
            import onnx
            import onnxsim

            logger.info("Running onnx-simplifier …")
            model_onnx = onnx.load(str(out_path))
            model_simplified, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simplified, str(out_path))
                logger.info("Simplification successful.")
            else:
                logger.warning("Simplification check failed; keeping original.")
        except (ImportError, Exception) as e:
            logger.warning(
                "onnx-simplifier skipped (%s); install with: pip install onnxsim onnxruntime",
                e,
            )

    # Optional: graph optimization
    if graph_opt:
        import shutil

        temp_path = out_path.with_suffix(".graph_opt.onnx")
        try:
            optimize_graph(out_path, temp_path)
            shutil.copy(temp_path, out_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # Optional: FP16 conversion
    if fp16:
        import shutil

        temp_path = out_path.with_suffix(".fp16.onnx")
        try:
            convert_fp16(out_path, temp_path)
            shutil.copy(temp_path, out_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # Optional: convert to ORT format
    if to_ort:
        ort_path = out_path.with_suffix(".ort")
        convert_to_ort(out_path, ort_path)
        logger.info("ORT model saved: %s", ort_path)

    # Print model info
    try:
        import onnx

        m = onnx.load(str(out_path))
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info("ONNX model info:")
        logger.info("  Path:    %s", out_path)
        logger.info("  Size:    %.1f MB", size_mb)
        logger.info("  Opset:   %d", m.opset_import[0].version)
        logger.info("  Inputs:  %s", [n.name for n in m.graph.input])
        logger.info("  Outputs: %s", [n.name for n in m.graph.output])
    except ImportError:
        pass

    # Optional: verify
    if verify:
        try:
            verify_model(out_path)
        except Exception as e:
            logger.error("Verification failed: %s", e)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Export LightDeepFilterNet checkpoint to ONNX."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best.pt",
        help="Path to .pt checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .onnx path (default: same dir as checkpoint, same stem + .onnx)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/default.yaml",
        help="Model config YAML (default: src/configs/default.yaml)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Export streaming model (T=1 per call, hidden states as I/O). "
        "Processes one hop (480 samples) per call. "
        "Ignores --chunk-frames.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=512,
        help="Fixed time frames per chunk exported into the ONNX model "
        "(default: 512 ≈ 5 s at 48 kHz / hop 480). "
        "Inference script will pad/chunk audio to match this.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18; dynamo exporter requires >= 18)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported model with onnxruntime (requires onnxruntime)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnx-simplifier after export (requires onnxsim)",
    )
    parser.add_argument(
        "--graph-opt",
        action="store_true",
        help="Apply ONNX graph optimizations (constant folding, operator fusion, etc.)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert to FP16 precision (for GPU/WebGPU backends, ~2x size reduction)",
    )
    parser.add_argument(
        "--to-ort",
        action="store_true",
        help="Also save an ORT-format copy alongside the ONNX output (faster loading)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if args.out is None:
        out_path = ckpt_path.with_suffix(".onnx")
    else:
        out_path = Path(args.out)

    export(
        ckpt_path=ckpt_path,
        out_path=out_path,
        config_path=Path(args.config),
        chunk_frames=args.chunk_frames,
        streaming=args.streaming,
        opset=args.opset,
        verify=args.verify,
        simplify=args.simplify,
        graph_opt=args.graph_opt,
        fp16=args.fp16,
        to_ort=args.to_ort,
    )


if __name__ == "__main__":
    main()
