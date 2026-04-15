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

"""

import argparse
import logging
from pathlib import Path

import torch

from src.configs.config import load_config
from src.model.lightdeepfilternet import init_model
from src.model.streaming import (
    StreamingDfDecoder,
    StreamingEncoder,
    StreamingErbDecoder,
)
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
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if streaming:
        _export_streaming(model, model_cfg, loader_cfg, out_path, opset, device)
        # Apply post-processing to each of the 3 exported models
        # out_path may be a directory (e.g. "checkpoints/" or "checkpoints") or a file path;
        # treat it as a directory when it has no .onnx suffix.
        out_dir = out_path if out_path.suffix != ".onnx" else out_path.parent
        for stem in ("enc", "erb_dec", "df_dec"):
            postprocess(
                out_dir / f"{stem}.onnx",
                simplify=simplify,
                verify=verify,
            )
    else:
        _export_chunked(
            model, model_cfg, loader_cfg, out_path, chunk_frames, opset, device
        )
        postprocess(
            out_path,
            simplify=simplify,
            verify=verify,
        )


def _export_streaming(model, model_cfg, loader_cfg, out_path, opset, device):
    """Export 3 separate ONNX models following the DFN3 architecture.

    Mirrors DeepFilterNet3's export.py split into enc / erb_dec / df_dec.
    All DSP (STFT, ERB, normalisation, DF application) runs in the Rust/WASM
    caller; these ONNX files contain only neural-network inference.

    Output files (placed next to ``out_path``):
        enc.onnx      — StreamingEncoder
        erb_dec.onnx  — StreamingErbDecoder
        df_dec.onnx   — StreamingDfDecoder
    """
    model.eval()

    out_dir = out_path if out_path.suffix != ".onnx" else out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_mod = StreamingEncoder(model).eval().to(device)
    erb_dec_mod = StreamingErbDecoder(model).eval().to(device)
    df_dec_mod = StreamingDfDecoder(model).eval().to(device)

    nb_erb = model_cfg.nb_erb
    nb_df = model_cfg.nb_df
    conv_ch = model_cfg.conv_ch
    emb_hidden = model_cfg.emb_hidden_dim
    df_hidden = model_cfg.df_hidden_dim
    kt_inp = model_cfg.conv_kernel_inp[0]  # 3 → buf depth 2
    kt_dfp = model_cfg.df_pathway_kernel_size_t  # 5 → buf depth 4
    df_order = model_cfg.df_order

    # Read actual layer counts from the model, not config.
    # SqueezedLiGRU_S may use num_layers != emb_num_layers (enc uses 1, dec uses emb_num_layers-1).
    enc_layers = len(model.enc.emb_gru.ligru.rnn)
    erb_layers = len(model.erb_dec.emb_gru.ligru.rnn)
    df_layers = len(model.df_dec.df_gru.ligru.rnn)

    # ---- Encoder ----
    feat_erb_in = torch.randn(1, 1, 1, nb_erb, device=device)
    feat_spec_in = torch.randn(1, 2, 1, nb_df, device=device)
    buf_erb0 = torch.zeros(1, 1, kt_inp - 1, nb_erb, device=device)
    buf_df0 = torch.zeros(1, 2, kt_inp - 1, nb_df, device=device)
    h_enc = torch.zeros(enc_layers, 1, emb_hidden, device=device)

    with torch.no_grad():
        enc_out = enc_mod(feat_erb_in, feat_spec_in, buf_erb0, buf_df0, h_enc)
    e0, e1, e2, e3, emb, c0 = enc_out[:6]
    logger.info("Encoder forward OK — out shapes: %s", [list(t.shape) for t in enc_out])

    enc_path = out_dir / "enc.onnx"
    torch.onnx.export(
        enc_mod,
        (feat_erb_in, feat_spec_in, buf_erb0, buf_df0, h_enc),
        enc_path,
        input_names=["feat_erb", "feat_spec", "buf_erb0", "buf_df0", "h_enc"],
        output_names=[
            "e0",
            "e1",
            "e2",
            "e3",
            "emb",
            "c0",
            "lsnr",
            "buf_erb0_new",
            "buf_df0_new",
            "h_enc_new",
        ],
        opset_version=opset,
        dynamo=True,
        optimize=True,
        external_data=False,
    )
    logger.info("Exported: %s", enc_path)

    # ---- ERB decoder ----
    h_erb = torch.zeros(erb_layers, 1, emb_hidden, device=device)

    with torch.no_grad():
        erb_out = erb_dec_mod(emb, e3, e2, e1, e0, h_erb)
    logger.info(
        "ERB decoder forward OK — out shapes: %s", [list(t.shape) for t in erb_out]
    )

    erb_path = out_dir / "erb_dec.onnx"
    torch.onnx.export(
        erb_dec_mod,
        (emb, e3, e2, e1, e0, h_erb),
        erb_path,
        input_names=["emb", "e3", "e2", "e1", "e0", "h_erb"],
        output_names=["mask", "h_erb_new"],
        opset_version=opset,
        dynamo=True,
        optimize=True,
        external_data=False,
    )
    logger.info("Exported: %s", erb_path)

    # ---- DF decoder ----
    buf_dfp = torch.zeros(1, conv_ch, kt_dfp - 1, nb_df, device=device)
    h_df = torch.zeros(df_layers, 1, df_hidden, device=device)

    with torch.no_grad():
        df_out = df_dec_mod(emb, c0, buf_dfp, h_df)
    logger.info(
        "DF decoder forward OK — out shapes: %s", [list(t.shape) for t in df_out]
    )
    logger.info(
        "  coefs shape: %s  (df_order=%d, nb_df=%d)",
        list(df_out[0].shape),
        df_order,
        nb_df,
    )

    df_path = out_dir / "df_dec.onnx"
    torch.onnx.export(
        df_dec_mod,
        (emb, c0, buf_dfp, h_df),
        df_path,
        input_names=["emb", "c0", "buf_dfp", "h_df"],
        output_names=["coefs", "buf_dfp_new", "h_df_new"],
        opset_version=opset,
        dynamo=True,
        optimize=True,
        external_data=False,
    )
    logger.info("Exported: %s", df_path)
    logger.info("Streaming export done — 3 models in %s", out_dir)


def _export_chunked(
    model, model_cfg, loader_cfg, out_path, chunk_frames, opset, device
):
    """Export fixed-chunk model (T=chunk_frames STFT frames per call).

    LiGRU specializes T to a constant at export time — dynamic T is not
    supported. Used for testing only; production uses --streaming.
    """
    spec, feat_erb, feat_spec = build_dummy_inputs(
        model_cfg, loader_cfg, T=chunk_frames, device=device
    )

    with torch.no_grad():
        spec_e, mask, lsnr, df_coefs, *_ = model(spec, feat_erb, feat_spec)
    logger.info(
        "PyTorch forward OK — outputs: %s",
        [list(o.shape) for o in (spec_e, mask, lsnr, df_coefs)],
    )
    logger.info(
        "Chunk: %d frames (%.2f s at %d Hz)",
        chunk_frames,
        chunk_frames * model_cfg.hop_size / model_cfg.sr,
        model_cfg.sr,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting to %s (opset=%d, T=%d) …", out_path, opset, chunk_frames)

    torch.onnx.export(
        model,
        (spec, feat_erb, feat_spec),
        out_path,
        input_names=["spec", "feat_erb", "feat_spec"],
        output_names=["spec_e", "mask", "lsnr", "df_coefs", "h_enc", "h_erb", "h_df"],
        opset_version=opset,
        dynamo=True,
        optimize=True,
        external_data=False,
    )
    logger.info("Export done.")


def simplify_model(model_path: Path) -> None:
    """Simplify the model using onnx-simplifier."""
    import onnx
    import onnxsim

    logger.info("Simplifying model %s …", model_path)
    model = onnx.load(str(model_path))
    model_simplified, check = onnxsim.simplify(model)

    if check:
        onnx.save(model_simplified, str(model_path))
        logger.info("Simplification successful.")
    else:
        logger.warning("Simplification check failed; keeping original.")


def postprocess(
    out_path: Path,
    simplify: bool = False,
    verify: bool = False,
) -> None:
    # Optional: onnx-simplifier
    if simplify:
        simplify_model(out_path)

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
