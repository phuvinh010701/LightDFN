"""Training script for LightDeepFilterNet."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from loguru import logger
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.configs.config import ModelConfig, TrainConfig, load_config
from src.dataloader.loader import DataLoaderBuilder, DeepFilterNetDataLoader, DsBatch
from src.losses.loss import Loss, LossConfig
from src.model.lightdeepfilternet import init_model
from src.utils.erb import get_erb_filterbanks
from src.utils.io import get_device
from src.utils.utils import count_parameters


def compute_stft(audio: Tensor, fft_size: int, hop_size: int, window: Tensor) -> Tensor:
    """STFT consistent with FftDataset (center=False, hann window).

    Args:
        audio (Tensor): Input waveform of shape ``[B, C, T_samples]``.
        fft_size (int): FFT size.
        hop_size (int): Hop size between frames.
        window (Tensor): Analysis window of length ``fft_size``.

    Returns:
        Tensor: Real-valued STFT output of shape ``[B, C, T_frames, F, 2]``.
    """
    B, C, T = audio.shape
    out = torch.stft(
        audio.reshape(B * C, T),
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window,
        return_complex=True,
        center=False,
    )
    # out: [B*C, F, T_frames] → [B, C, T_frames, F, 2]
    F_bins, T_frames = out.shape[-2], out.shape[-1]
    out = out.view(B, C, F_bins, T_frames).permute(0, 1, 3, 2)
    return torch.view_as_real(out.contiguous())


def prepare_batch(
    batch: DsBatch,
    fft_size: int,
    hop_size: int,
    window: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert a DsBatch to model-ready tensors.

    Returns:
        spec_noisy: [B, 1, T_frames, F, 2]
        spec_clean: [B, 1, T_frames, F, 2]
        feat_erb:   [B, 1, T_frames, nb_erb]
        feat_spec:  [B, 1, T_frames, nb_spec, 2]
    """
    noisy = batch.noisy[:, :1].to(device, non_blocking=True)
    clean = batch.speech[:, :1].to(device, non_blocking=True)

    spec_noisy = compute_stft(noisy, fft_size, hop_size, window)
    spec_clean = compute_stft(clean, fft_size, hop_size, window)

    feat_erb = batch.feat_erb[:, :1].to(device, non_blocking=True)
    feat_spec = torch.view_as_real(batch.feat_spec[:, :1].to(device, non_blocking=True))

    return spec_noisy, spec_clean, feat_erb, feat_spec


def run_epoch(
    model: nn.Module,
    loader: DeepFilterNetDataLoader,
    loss_fn: Loss,
    optimizer: AdamW | None,
    device: torch.device,
    window: Tensor,
    fft_size: int,
    hop_size: int,
    is_train: bool,
    grad_clip: float = 1.0,
    max_nans: int = 50,
    log_every_n_steps: int = 100,
    epoch: int = 0,
) -> tuple[float, dict[str, float]]:
    """Run one train or validation epoch.

    Returns:
        avg_loss:  Mean loss over all batches.
        summaries: Per-component mean losses (only populated during validation).
    """
    model.train(is_train)
    # Collect detailed per-component breakdown only on validation (mirrors DeepFilterNet)
    loss_fn.store_losses = not is_train

    total_loss = 0.0
    n_batches = 0
    n_nans = 0
    all_summaries: dict[str, list[float]] = {}

    prefix = "train" if is_train else "val"
    pbar = tqdm(
        loader, desc=f"{'Train' if is_train else 'Val':5s} epoch {epoch}", leave=False
    )

    with torch.set_grad_enabled(is_train):
        for step, batch in enumerate(pbar):
            spec_noisy, spec_clean, feat_erb, feat_spec = prepare_batch(
                batch, fft_size, hop_size, window, device
            )

            # Val uses a clone so the spec tensor is not modified in-place by the model
            model_input = spec_noisy if is_train else spec_noisy.clone()
            enhanced, mask, lsnr, _ = model(model_input, feat_erb, feat_spec)

            snrs = torch.from_numpy(batch.snr).float().to(device, non_blocking=True)
            max_freq = torch.from_numpy(batch.max_freq).to(device, non_blocking=True)

            try:
                loss = loss_fn(
                    spec_clean,
                    spec_noisy,
                    enhanced,
                    mask,
                    lsnr,
                    snrs,
                    max_freq=max_freq,
                )
            except Exception as e:
                msg = str(e).lower()
                if "nan" in msg or "finite" in msg:
                    logger.warning(f"NaN in loss at step {step}: {e}. Skipping.")
                    n_nans += 1
                    if n_nans > max_nans:
                        raise RuntimeError(f"Exceeded max NaNs ({max_nans})") from e
                    continue
                raise

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                try:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip, error_if_nonfinite=True
                    )
                except RuntimeError as e:
                    if "nan" in str(e).lower() or "non-finite" in str(e).lower():
                        logger.warning(
                            f"NaN in gradients at step {step}: {e}. Skipping."
                        )
                        optimizer.zero_grad()
                        n_nans += 1
                        if n_nans > max_nans:
                            raise RuntimeError(f"Exceeded max NaNs ({max_nans})") from e
                        continue
                    raise
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Collect per-component summaries (only stored when loss_fn.store_losses=True)
            for k, vals in loss_fn.summaries.items():
                all_summaries.setdefault(k, []).extend(v.item() for v in vals)
            loss_fn.reset_summaries()

            pbar.set_postfix(loss=f"{loss.item():.4f}", nans=n_nans)

            # Step-level wandb logging (train only)
            if is_train and (step + 1) % log_every_n_steps == 0:
                global_step = epoch * len(loader) + step
                wandb.log({f"{prefix}/loss_step": loss.item(), "step": global_step})

    avg_loss = total_loss / max(n_batches, 1)
    avg_summaries = {k: sum(v) / len(v) for k, v in all_summaries.items() if v}
    return avg_loss, avg_summaries


def build_loss(model_cfg: ModelConfig, device: torch.device) -> Loss:
    """Build the composite loss function from a model config.

    Args:
        model_cfg (ModelConfig): Model configuration used to align loss parameters.
        device (torch.device): Device to place loss tensors on.

    Returns:
        Loss: Configured loss function instance.
    """
    erb_fb, _ = get_erb_filterbanks(
        model_cfg.sr, model_cfg.fft_size, model_cfg.nb_erb, model_cfg.min_nb_freqs
    )
    return Loss(
        LossConfig(),
        erb_fb=erb_fb.to(device),
        fft_size=model_cfg.fft_size,
        hop_size=model_cfg.hop_size,
        sr=model_cfg.sr,
        lsnr_min=model_cfg.lsnr_min,
        lsnr_max=model_cfg.lsnr_max,
    )


def build_scheduler(
    optimizer: AdamW, train_cfg: TrainConfig
) -> LinearLR | CosineAnnealingLR | SequentialLR:
    """Cosine schedule with linear warmup, mirroring DeepFilterNet's cosine_scheduler."""
    if train_cfg.warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=train_cfg.lr_min / train_cfg.lr,
            end_factor=1.0,
            total_iters=train_cfg.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.epochs - train_cfg.warmup_epochs,
            eta_min=train_cfg.lr_min,
        )
        return SequentialLR(
            optimizer, [warmup, cosine], milestones=[train_cfg.warmup_epochs]
        )
    return CosineAnnealingLR(
        optimizer, T_max=train_cfg.epochs, eta_min=train_cfg.lr_min
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: object,
    best_val_loss: float,
) -> None:
    """Persist model, optimiser, and scheduler state to disk.

    Args:
        path (Path): Destination file path.
        epoch (int): Current epoch index.
        model (nn.Module): Model to checkpoint.
        optimizer (AdamW): Optimiser whose state to save.
        scheduler (object): LR scheduler whose state to save.
        best_val_loss (float): Best validation loss seen so far.
    """
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    """Entry point: parse arguments, build all components, and run the training loop."""
    parser = argparse.ArgumentParser(description="Train LightDeepFilterNet")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    model_cfg, aug_cfg, loader_cfg, train_cfg = load_config(args.config)
    device = get_device()

    # Align model batch size with dataloader (used by Li-GRU hidden state init)
    model_cfg.batch_size = loader_cfg.batch_size

    wandb.init(
        project=train_cfg.wandb_project,
        name=args.run_name,
        config={
            "model": vars(model_cfg),
            "train": vars(train_cfg),
            "data": {
                k: v
                for k, v in vars(loader_cfg).items()
                if k not in ("speech_hdf5", "noise_hdf5", "rir_hdf5")
            },
        },
        resume="allow",
    )

    model = init_model(model_cfg).to(device)
    n_params = count_parameters(model)
    logger.info(f"Model parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    builder = DataLoaderBuilder(loader_cfg)
    train_loader = builder.build("train", aug_cfg)
    val_loader = builder.build("valid", aug_cfg)

    loss_fn = build_loss(model_cfg, device)

    # amsgrad=True matches DeepFilterNet's AdamW default
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.999),
        amsgrad=True,
    )
    scheduler = build_scheduler(optimizer, train_cfg)

    # STFT window matching FftDataset (hann, center=False)
    window = torch.hann_window(model_cfg.fft_size).to(device)

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, train_cfg.epochs):
        train_loader.start_epoch(epoch)

        train_loss, _ = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            window,
            model_cfg.fft_size,
            model_cfg.hop_size,
            is_train=True,
            grad_clip=train_cfg.grad_clip,
            max_nans=train_cfg.max_nans,
            log_every_n_steps=train_cfg.log_every_n_steps,
            epoch=epoch,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        log: dict = {"epoch": epoch, "train/loss": train_loss, "lr": current_lr}

        if (epoch + 1) % train_cfg.val_every_n_epochs == 0:
            val_loader.start_epoch(epoch)
            val_loss, val_sums = run_epoch(
                model,
                val_loader,
                loss_fn,
                None,
                device,
                window,
                model_cfg.fft_size,
                model_cfg.hop_size,
                is_train=False,
                max_nans=train_cfg.max_nans,
                epoch=epoch,
            )
            log["val/loss"] = val_loss
            log.update({f"val/{k}": v for k, v in val_sums.items()})

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_loss,
                )
                wandb.save(str(ckpt_dir / "best.pt"))
                logger.info(f"  → new best val loss: {best_val_loss:.4f}")
        else:
            val_loss = float("nan")
            is_best = False

        save_checkpoint(
            ckpt_dir / "last.pt", epoch, model, optimizer, scheduler, best_val_loss
        )
        wandb.log(log)

        logger.info(
            f"Epoch {epoch:03d}/{train_cfg.epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}"
            + ("  [best]" if is_best else "")
        )

    wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
