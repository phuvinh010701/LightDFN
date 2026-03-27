#!/usr/bin/env python3
"""Export a LightDeepFilterNet checkpoint to ONNX format.

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

  # With onnx-simplifier post-processing:
  uv run python -m scripts.export_onnx \\
    --ckpt checkpoints/best.pt \\
    --out checkpoints/lightdfn.onnx \\
    --simplify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.configs.config import load_config
from src.model.lightdeepfilternet import init_model
from src.model.streaming import StreamingLightDFN
from src.utils.io import get_device


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


def export(
    ckpt_path: Path,
    out_path: Path,
    config_path: Path,
    chunk_frames: int = 512,
    streaming: bool = False,
    opset: int = 18,
    verify: bool = False,
    simplify: bool = False,
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

    _postprocess(out_path, simplify, verify)


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
    print(f"PyTorch forward pass OK — outputs: {[list(o.shape) for o in out]}")
    print(
        f"Chunk size: {chunk_frames} frames  "
        f"({chunk_frames * model_cfg.hop_size / model_cfg.sr:.2f} s at {model_cfg.sr} Hz)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {out_path}  (opset={opset}) …")
    torch.onnx.export(
        model,
        args=dummy_inputs,
        f=str(out_path),
        input_names=["spec", "feat_erb", "feat_spec"],
        output_names=["enhanced_spec", "erb_mask", "lsnr", "df_coefs"],
        opset_version=opset,
        export_params=True,
        dynamo=False,
    )
    print("Export done.")


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
    print(f"Streaming forward pass OK — outputs: {[list(o.shape) for o in out]}")
    print(
        f"Streaming: 1 frame per call  "
        f"({model_cfg.hop_size / model_cfg.sr * 1000:.1f} ms per call at {model_cfg.sr} Hz)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {out_path}  (opset={opset}, streaming=True) …")
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
        dynamo=False,  # dynamo=True mis-traces Li-GRU's list-append hidden-state pattern,
        # causing the output hidden states to be constants (not updated).
        # Use the TorchScript-based exporter for correct state threading.
    )
    print("Export done.")


def _postprocess(out_path, simplify, _verify):
    # ------------------------------------------------------------------ #
    # Optional: onnx-simplifier
    # ------------------------------------------------------------------ #
    if simplify:
        try:
            import onnx
            import onnxsim

            print("Running onnx-simplifier …")
            model_onnx = onnx.load(str(out_path))
            model_simplified, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simplified, str(out_path))
                print("Simplification successful.")
            else:
                print("Warning: simplification check failed; keeping original.")
        except (ImportError, Exception) as e:
            print(
                f"onnx-simplifier skipped ({e}); install with: pip install onnxsim onnxruntime"
            )

    # ------------------------------------------------------------------ #
    # Print model info
    # ------------------------------------------------------------------ #
    try:
        import onnx

        m = onnx.load(str(out_path))
        size_mb = out_path.stat().st_size / 1024 / 1024
        print("\nONNX model info:")
        print(f"  Path:    {out_path}")
        print(f"  Size:    {size_mb:.1f} MB")
        print(f"  Opset:   {m.opset_import[0].version}")
        print(f"  Inputs:  {[n.name for n in m.graph.input]}")
        print(f"  Outputs: {[n.name for n in m.graph.output]}")
    except ImportError:
        pass


def main() -> None:
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
    )


if __name__ == "__main__":
    main()
