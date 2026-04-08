#!/usr/bin/env python3
"""Export a LightDeepFilterNet checkpoint to ONNX format.

Two export modes:

  (default)  Chunked — causal, fixed T frames per call.
    pad_feat look-ahead is removed so the model is fully causal and the ONNX
    graph can be used in real-time pipelines without future-frame buffering.

  --streaming  Python frame-by-frame inference.
    T=1 per call; GRU hidden states and conv buffers are explicit ONNX I/O.
    Used by scripts/infer_streaming_onnx.py.

Both modes use the dynamo-based exporter (opset 18) and support optional
post-processing with onnx-simplifier (--simplify).

Examples:
  # Causal chunked export (default):
  python scripts/export_onnx.py --ckpt checkpoints/best.pt \\
      --out checkpoints/lightdfn.onnx --simplify

  # Streaming export (opset 18, Python infer_streaming_onnx.py):
  python scripts/export_onnx.py --ckpt checkpoints/best.pt \\
      --out checkpoints/lightdfn_streaming.onnx --streaming --simplify

  # Streaming export for WASM / tract (opset 14, dynamo=False):
  python scripts/export_onnx.py --ckpt checkpoints/best.pt \\
      --out checkpoints/lightdfn_streaming_wasm.onnx --streaming --wasm --simplify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from src.configs.config import load_config
from src.model.lightdeepfilternet import init_model
from src.model.streaming import StreamingLightDFN
from src.utils.io import get_device


def _build_dummy_inputs(
    model_cfg,
    loader_cfg,
    T: int = 512,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = 1
    F = model_cfg.fft_size // 2 + 1
    E = model_cfg.nb_erb
    Fc = loader_cfg.nb_spec
    return (
        torch.randn(B, 1, T, F, 2, device=device),
        torch.randn(B, 1, T, E,    device=device),
        torch.randn(B, 1, T, Fc, 2, device=device),
    )


def export(
    ckpt_path: Path,
    out_path: Path,
    config_path: Path,
    chunk_frames: int = 512,
    streaming: bool = False,
    dfn_style: bool = False,
    wasm: bool = False,
    opset: int = 18,
    simplify: bool = False,
) -> None:
    model_cfg, _aug, loader_cfg, _train, _loss = load_config(str(config_path))

    device = get_device()
    model = init_model(model_cfg).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Remove look-ahead: pad_feat shifts features forward by conv_lookahead frames,
    # which requires future audio — incompatible with causal/real-time inference.
    # Replacing with Identity makes the model fully causal at a marginal quality cost.
    if model_cfg.conv_lookahead > 0:
        model.pad_feat = nn.Identity()

    # WASM/tract target: force dynamo=False and opset ≤ 14 for full op compatibility.
    # tract 0.21 supports GRU, Conv, Einsum, BatchNorm up to opset 14.
    use_dynamo = not wasm
    if wasm:
        opset = min(opset, 14)
        logger.info(f"WASM target: dynamo=False, opset capped at {opset}")

    if dfn_style:
        _export_split(model, model_cfg, loader_cfg, out_path, opset)
    elif streaming:
        _export_streaming(model, model_cfg, loader_cfg, out_path, opset)
    else:
        _export_chunked(model, model_cfg, loader_cfg, out_path, chunk_frames, opset)

    if simplify:
        _simplify(out_path)
    _print_onnx_info(out_path)


def _export_chunked(model, model_cfg, loader_cfg, out_path, chunk_frames, opset):
    device = next(model.parameters()).device
    dummy = _build_dummy_inputs(model_cfg, loader_cfg, T=chunk_frames, device=device)

    with torch.no_grad():
        out = model(*dummy)
    logger.info(f"PyTorch forward OK — outputs: {[list(o.shape) for o in out]}")
    logger.info(
        f"Chunk: {chunk_frames} frames "
        f"({chunk_frames * model_cfg.hop_size / model_cfg.sr:.2f} s at {model_cfg.sr} Hz)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting → {out_path}  (opset={opset}, dynamo=True) …")
    torch.onnx.export(
        model,
        args=dummy,
        f=str(out_path),
        input_names=["spec", "feat_erb", "feat_spec"],
        output_names=["enhanced_spec", "erb_mask", "lsnr", "df_coefs"],
        opset_version=opset,
        export_params=True,
        keep_initializers_as_inputs=False,
        external_data=False,
        verify=False,
        optimize=True,
        dynamo=True,
    )
    logger.info("Export done.")


def _export_streaming(model, model_cfg, loader_cfg, out_path, opset):
    device = next(model.parameters()).device
    wrapper = StreamingLightDFN(model)
    wrapper.eval()

    dummy_frame = _build_dummy_inputs(model_cfg, loader_cfg, T=1, device=device)
    h_enc, h_erb, h_df, buf_erb0, buf_df0, buf_dfp, buf_spec = wrapper.init_states(batch_size=1, device=device)

    with torch.no_grad():
        out = wrapper(*dummy_frame, h_enc, h_erb, h_df, buf_erb0, buf_df0, buf_dfp, buf_spec)
    logger.info(f"Streaming forward OK — outputs: {[list(o.shape) for o in out]}")
    logger.info(
        f"Streaming: 1 frame/call "
        f"({model_cfg.hop_size / model_cfg.sr * 1000:.1f} ms/call at {model_cfg.sr} Hz)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting → {out_path}  (opset={opset}, dynamo=True) …")
    torch.onnx.export(
        wrapper,
        args=(*dummy_frame, h_enc, h_erb, h_df, buf_erb0, buf_df0, buf_dfp, buf_spec),
        f=str(out_path),
        input_names=[
            "spec", "feat_erb", "feat_spec",
            "h_enc", "h_erb", "h_df",
            "buf_erb0", "buf_df0", "buf_dfp", "buf_spec",
        ],
        output_names=[
            "enhanced_spec",
            "h_enc_new", "h_erb_new", "h_df_new",
            "buf_erb0_new", "buf_df0_new", "buf_dfp_new", "buf_spec_new",
        ],
        opset_version=opset,
        export_params=True,
        keep_initializers_as_inputs=False,
        external_data=False,
        verify=False,
        optimize=True,
        dynamo=True,
    )
    logger.info("Export done.")


def _export_split(model, model_cfg, loader_cfg, out_path, opset):
    """DeepFilterNet style: Export 3 separate models with dynamic time axis."""
    device = next(model.parameters()).device
    model.eval()
    
    # 1. Base out directory
    out_dir = out_path.parent / "dfn_style"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Split export (DFN style) → {out_dir}")

    # Dummy inputs for full trace
    spec, feat_erb, feat_spec = _build_dummy_inputs(model_cfg, loader_cfg, T=16, device=device)
    
    # --- Export Encoder ---
    # Re-align feat_spec like the model expects internally [B, 2, T, F]
    feat_spec_enc = feat_spec.squeeze(1).permute(0, 3, 1, 2)
    
    # We wrap the forward to ensure 4D outputs [B, C, T, F]
    class EncoderWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, feat_erb, feat_spec):
            e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
            # Ensure all are 4D [B, C, S, F]
            # emb: [B, T, H] -> [B, H, T, 1]
            emb = emb.permute(0, 2, 1).unsqueeze(-1)
            # lsnr: [B, T, 1] -> [B, 1, T, 1]
            lsnr = lsnr.permute(0, 2, 1).unsqueeze(-1)
            return e0, e1, e2, e3, emb, c0, lsnr

    enc_wrapper = EncoderWrapper(model.enc)
    enc_path = out_dir / "enc.onnx"
    logger.info(f"  Exporting Encoder → {enc_path.name}")
    torch.onnx.export(
        enc_wrapper,
        args=(feat_erb, feat_spec_enc),
        f=str(enc_path),
        input_names=["feat_erb", "feat_spec"],
        output_names=["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
        dynamic_axes={
            "feat_erb": {2: "S"},
            "feat_spec": {2: "S"},
            "e0": {2: "S"}, "e1": {2: "S"}, "e2": {2: "S"}, "e3": {2: "S"},
            "emb": {2: "S"}, "c0": {2: "S"}, "lsnr": {2: "S"}
        },
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )

    # We need intermediate results for tracing next stages
    with torch.no_grad():
        e0, e1, e2, e3, emb, c0, _lsnr = enc_wrapper(feat_erb, feat_spec_enc)

    # --- Export ERB Decoder ---
    # Wrap to handle 4D emb
    class ErbDecWrapper(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec
        def forward(self, emb, e3, e2, e1, e0):
            # emb: [B, H, T, 1] -> [B, T, H]
            emb = emb.squeeze(-1).permute(0, 2, 1)
            return self.dec(emb, e3, e2, e1, e0)

    erb_wrapper = ErbDecWrapper(model.erb_dec)
    erb_path = out_dir / "erb_dec.onnx"
    logger.info(f"  Exporting ERB Decoder → {erb_path.name}")
    torch.onnx.export(
        erb_wrapper,
        args=(emb, e3, e2, e1, e0),
        f=str(erb_path),
        input_names=["emb", "e3", "e2", "e1", "e0"],
        output_names=["mask"],
        dynamic_axes={
            "emb": {2: "S"},
            "e3": {2: "S"}, "e2": {2: "S"}, "e1": {2: "S"}, "e0": {2: "S"},
            "mask": {2: "S"}
        },
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )

    # --- Export DF Decoder ---
    class DfDecWrapper(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec
        def forward(self, emb, c0):
            # emb: [B, H, T, 1] -> [B, T, H]
            emb = emb.squeeze(-1).permute(0, 2, 1)
            return self.dec(emb, c0)

    df_wrapper = DfDecWrapper(model.df_dec)
    df_path = out_dir / "df_dec.onnx"
    logger.info(f"  Exporting DF Decoder → {df_path.name}")
    torch.onnx.export(
        df_wrapper,
        args=(emb, c0),
        f=str(df_path),
        input_names=["emb", "c0"],
        output_names=["coefs"],
        dynamic_axes={
            "emb": {2: "S"},
            "c0": {2: "S"},
            "coefs": {2: "S"} # Will be B, C, S, 1
        },
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )

    # Simplify all
    for p in [enc_path, erb_path, df_path]:
        _simplify(p)

    logger.info("Split export done.")


def _simplify(out_path: Path) -> None:
    try:
        import onnx
        import onnxsim

        logger.info(f"Simplifying {out_path.name} …")
        m, check = onnxsim.simplify(onnx.load(str(out_path)))
        if check:
            onnx.save(m, str(out_path))
            logger.info("Simplification OK.")
        else:
            logger.warning("Simplification check failed — keeping original.")
    except ImportError:
        logger.warning("onnxsim not found (pip install onnxsim onnxruntime), skipping.")
    except Exception as e:
        logger.warning(f"onnxsim error: {e} — keeping original.")


def _print_onnx_info(out_path: Path) -> None:
    try:
        import onnx

        m = onnx.load(str(out_path))
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info(f"ONNX model: {out_path}")
        logger.info(f"  Size:    {size_mb:.1f} MB")
        logger.info(f"  Opset:   {m.opset_import[0].version}")
        logger.info(f"  Inputs:  {[n.name for n in m.graph.input]}")
        logger.info(f"  Outputs: {[n.name for n in m.graph.output]}")
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LightDeepFilterNet checkpoint to ONNX (causal, opset 18)."
    )
    parser.add_argument(
        "--ckpt", type=str, default="checkpoints/best.pt",
        help="Checkpoint path (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output .onnx path (default: same dir/stem as --ckpt)",
    )
    parser.add_argument(
        "--config", type=str, default="src/configs/default.yaml",
        help="Model config YAML (default: src/configs/default.yaml)",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Export streaming model (T=1/call, GRU + conv states as explicit I/O). "
             "For scripts/infer_streaming_onnx.py.",
    )
    parser.add_argument(
        "--chunk-frames", type=int, default=512,
        help="Frames per chunk for chunked export (default: 512 ≈ 5 s at 48 kHz).",
    )
    parser.add_argument(
        "--opset", type=int, default=18,
        help="ONNX opset version (default: 18).",
    )
    parser.add_argument(
        "--dfn-style", action="store_true",
        help="Export split models (Encoder, ERB Decoder, DF Decoder) like DFN3. "
             "Enables optimized streaming via tract-pulse.",
    )
    parser.add_argument(
        "--wasm", action="store_true",
        help="Target WASM/tract: force dynamo=False and opset <= 14.",
    )
    parser.add_argument(
        "--simplify", action="store_true",
        help="Run onnx-simplifier after export (requires: pip install onnxsim onnxruntime).",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".onnx")

    export(
        ckpt_path=ckpt_path,
        out_path=out_path,
        config_path=Path(args.config),
        chunk_frames=args.chunk_frames,
        streaming=args.streaming,
        dfn_style=args.dfn_style,
        wasm=args.wasm,
        opset=args.opset,
        simplify=args.simplify,
    )


if __name__ == "__main__":
    main()
