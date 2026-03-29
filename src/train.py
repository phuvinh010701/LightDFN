"""Training script for LightDeepFilterNet."""

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import wandb
from src.configs.config import LossConfig, ModelConfig, TrainConfig, load_config
from src.dataloader.loader import DataLoaderBuilder, DeepFilterNetDataLoader, DsBatch
from src.losses.loss import Loss
from src.model.lightdeepfilternet import init_model
from src.utils.audio import compute_stft, spec_to_audio, spectrogram_to_db
from src.utils.erb import get_erb_filterbanks
from src.utils.io import get_device
from src.utils.utils import count_parameters

matplotlib.use("Agg")  # Non-interactive backend for server-side rendering


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
    loss_fn.store_losses = True

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

            # Collect per-component summaries for this step
            _abbrev = {
                "MaskLoss": "ml",
                "SpectralLoss": "sl",
                "MultiResSpecLoss": "mrsl",
                "SdrLoss": "sdr",
                "LocalSnrLoss": "lsnr",
            }
            step_sums: dict[str, float] = {}
            for k, vals in loss_fn.summaries.items():
                if vals:
                    step_sums[k] = sum(v.item() for v in vals)
                    all_summaries.setdefault(k, []).extend(v.item() for v in vals)
            loss_fn.reset_summaries()

            postfix: dict[str, object] = {"loss": f"{loss.item():.4f}", "nans": n_nans}
            for k, v in step_sums.items():
                postfix[_abbrev.get(k, k)] = f"{v:.4f}"
            pbar.set_postfix(postfix)

            # Step-level wandb logging (train only)
            if is_train and (step + 1) % log_every_n_steps == 0:
                global_step = epoch * len(loader) + step
                step_log: dict[str, object] = {
                    f"{prefix}/loss_step": loss.item(),
                }
                for k, v in step_sums.items():
                    step_log[f"{prefix}/{k}_step"] = v
                wandb.log(step_log, step=global_step)

    avg_loss = total_loss / max(n_batches, 1)
    avg_summaries = {k: sum(v) / len(v) for k, v in all_summaries.items() if v}
    return avg_loss, avg_summaries


def _make_spectrogram_figure(
    noisy: Tensor,
    clean: Tensor,
    enhanced: Tensor,
    sr: int,
    hop_size: int,
    snr: float,
) -> plt.Figure:
    """Create a side-by-side spectrogram figure for one sample.

    Args:
        noisy:    ``[1, T_frames, F, 2]`` noisy spectrogram.
        clean:    ``[1, T_frames, F, 2]`` clean spectrogram.
        enhanced: ``[1, T_frames, F, 2]`` model output spectrogram.
        sr:       Sample rate in Hz.
        hop_size: Hop size in samples (used to set time axis).
        snr:      Input SNR of the sample in dB.

    Returns:
        Matplotlib Figure with three spectrogram subplots.
    """
    titles = ["Noisy", "Clean (target)", "Enhanced"]
    specs = [noisy, clean, enhanced]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(f"Input SNR: {snr:.1f} dB", fontsize=12)

    T_frames = noisy.shape[1]
    duration_s = T_frames * hop_size / sr
    freq_max = sr / 2 / 1000  # kHz

    for ax, title, spec in zip(axes, titles, specs):
        db = spectrogram_to_db(spec)  # [F, T]
        ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=[0, duration_s, 0, freq_max],
            cmap="magma",
            vmin=-80,
            vmax=0,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")

    return fig


def eval_audio_samples(
    model: nn.Module,
    loader: DeepFilterNetDataLoader,
    device: torch.device,
    window: Tensor,
    fft_size: int,
    hop_size: int,
    sr: int,
    num_samples: int,
    epoch: int,
) -> None:
    """Run inference on a few validation samples and log spectrograms + audio to WandB.

    Args:
        model:       Model in eval mode.
        loader:      Validation dataloader.
        device:      Compute device.
        window:      STFT window.
        fft_size:    FFT size.
        hop_size:    Hop size.
        sr:          Sample rate in Hz.
        num_samples: Number of examples to log.
        epoch:       Current epoch index (used for WandB step key).
    """
    model.eval()
    logged = 0
    log_dict: dict = {"epoch": epoch}

    with torch.no_grad():
        for batch in loader:
            if logged >= num_samples:
                break

            spec_noisy, spec_clean, feat_erb, feat_spec = prepare_batch(
                batch, fft_size, hop_size, window, device
            )
            enhanced, _mask, _lsnr, _ = model(spec_noisy.clone(), feat_erb, feat_spec)

            batch_snrs = batch.snr.tolist()
            n = min(num_samples - logged, spec_noisy.shape[0])

            for i in range(n):
                snr_val = float(batch_snrs[i])
                tag = f"audio_eval/sample_{logged}"

                # --- Spectrograms ---
                fig = _make_spectrogram_figure(
                    spec_noisy[i, :1],
                    spec_clean[i, :1],
                    enhanced[i, :1],
                    sr=sr,
                    hop_size=hop_size,
                    snr=snr_val,
                )
                log_dict[f"{tag}/spectrogram"] = wandb.Image(fig)
                plt.close(fig)

                # --- Audio clips ---
                noisy_wav = spec_to_audio(
                    spec_noisy[i : i + 1, :1], fft_size, hop_size, window
                ).squeeze()
                clean_wav = spec_to_audio(
                    spec_clean[i : i + 1, :1], fft_size, hop_size, window
                ).squeeze()
                enh_wav = spec_to_audio(
                    enhanced[i : i + 1, :1], fft_size, hop_size, window
                ).squeeze()

                for name, wav in [
                    ("noisy", noisy_wav),
                    ("clean", clean_wav),
                    ("enhanced", enh_wav),
                ]:
                    wav_np = wav.cpu().float().numpy()
                    # Normalize to [-1, 1] to avoid clipping artefacts in player
                    peak = np.abs(wav_np).max()
                    if peak > 0:
                        wav_np = wav_np / peak
                    log_dict[f"{tag}/{name}_audio"] = wandb.Audio(
                        wav_np, sample_rate=sr, caption=f"{name} (SNR={snr_val:.1f}dB)"
                    )

                logged += 1

    wandb.log(log_dict)
    logger.info(f"Logged {logged} audio eval sample(s) at epoch {epoch}.")


def build_loss(
    model_cfg: ModelConfig, loss_cfg: LossConfig, device: torch.device
) -> Loss:
    """Build the composite loss function from a model config.

    Args:
        model_cfg (ModelConfig): Model configuration used to align loss parameters.
        loss_cfg (LossConfig): Loss weights and hyperparameters.
        device (torch.device): Device to place loss tensors on.

    Returns:
        Loss: Configured loss function instance.
    """
    erb_fb, _ = get_erb_filterbanks(
        model_cfg.sr, model_cfg.fft_size, model_cfg.nb_erb, model_cfg.min_nb_freqs
    )
    return Loss(
        loss_cfg,
        erb_fb=erb_fb.to(device),
        fft_size=model_cfg.fft_size,
        hop_size=model_cfg.hop_size,
        sr=model_cfg.sr,
        lsnr_min=model_cfg.lsnr_min,
        lsnr_max=model_cfg.lsnr_max,
    ).to(device)


def cosine_weight_decay(
    epoch: int, total_epochs: int, wd_start: float, wd_end: float
) -> float:
    """Cosine annealing of weight decay from wd_start to wd_end over total_epochs."""
    return wd_end + 0.5 * (wd_start - wd_end) * (
        1.0 + math.cos(math.pi * epoch / total_epochs)
    )


def build_scheduler(
    optimizer: AdamW, train_cfg: TrainConfig
) -> LinearLR | CosineAnnealingLR | SequentialLR:
    """Cosine schedule with linear warmup, mirroring DeepFilterNet's cosine_scheduler."""
    if train_cfg.warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=train_cfg.lr_warmup / train_cfg.lr,
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
    epochs_without_improvement: int = 0,
) -> None:
    """Persist model, optimiser, and scheduler state to disk.

    Args:
        path (Path): Destination file path.
        epoch (int): Current epoch index.
        model (nn.Module): Model to checkpoint.
        optimizer (AdamW): Optimiser whose state to save.
        scheduler (object): LR scheduler whose state to save.
        best_val_loss (float): Best validation loss seen so far.
        epochs_without_improvement (int): Early-stopping counter.
    """
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
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

    model_cfg, aug_cfg, loader_cfg, train_cfg, loss_cfg = load_config(args.config)
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

    loss_fn = build_loss(model_cfg, loss_cfg, device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=tuple(train_cfg.adam_betas),  # type: ignore[arg-type]
        amsgrad=train_cfg.amsgrad,
    )
    scheduler = build_scheduler(optimizer, train_cfg)

    # STFT window matching FftDataset (hann, center=False)
    window = torch.hann_window(model_cfg.fft_size).to(device)

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        logger.info(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, train_cfg.epochs):
        train_loader.start_epoch(epoch)

        train_loss, train_sums = run_epoch(
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

        current_wd = cosine_weight_decay(
            epoch, train_cfg.epochs, train_cfg.weight_decay, train_cfg.weight_decay_end
        )
        for pg in optimizer.param_groups:
            pg["weight_decay"] = current_wd

        log: dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "lr": current_lr,
            "weight_decay": current_wd,
        }
        log.update({f"train/{k}": v for k, v in train_sums.items()})

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

            if (
                train_cfg.audio_eval_every_n_epochs > 0
                and (epoch + 1) % train_cfg.audio_eval_every_n_epochs == 0
            ):
                val_loader.start_epoch(epoch)
                eval_audio_samples(
                    model,
                    val_loader,
                    device,
                    window,
                    model_cfg.fft_size,
                    model_cfg.hop_size,
                    model_cfg.sr,
                    train_cfg.audio_eval_num_samples,
                    epoch,
                )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_loss,
                    epochs_without_improvement,
                )
                wandb.save(str(ckpt_dir / "best.pt"))
                logger.info(f"  → new best val loss: {best_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
        else:
            val_loss = float("nan")
            is_best = False

        save_checkpoint(
            ckpt_dir / "last.pt",
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_loss,
            epochs_without_improvement,
        )
        wandb.log(log)

        logger.info(
            f"Epoch {epoch:03d}/{train_cfg.epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}"
            + ("  [best]" if is_best else "")
        )

        if (
            train_cfg.early_stopping_patience > 0
            and epochs_without_improvement >= train_cfg.early_stopping_patience
        ):
            logger.info(
                f"Early stopping: no improvement for {epochs_without_improvement} epochs."
            )
            break

    wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
