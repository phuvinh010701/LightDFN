"""Loss functions for LightDeepFilterNet noise suppression.

Ported and adapted from DeepFilterNet:
  https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/loss.py
"""

from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.modules import as_complex, as_real


class _Angle(torch.autograd.Function):
    """Gradient-stable atan2 for complex tensors."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(torch.stack((-x.imag * inv, x.real * inv), dim=-1))


angle = _Angle.apply


def wg(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    """Wiener Gain mask: SS / (SS + NN)."""
    N = X - S
    SS = as_complex(S).abs().square()
    NN = as_complex(N).abs().square()
    return (SS / (SS + NN + eps)).clamp(0, 1)


def irm(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    """Ideal Ratio Mask: |S| / (|S| + |N|)."""
    N = X - S
    SS_mag = as_complex(S).abs()
    NN_mag = as_complex(N).abs()
    return (SS_mag / (SS_mag + NN_mag + eps)).clamp(0, 1)


def iam(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    """Ideal Amplitude Mask: |S| / |X|."""
    SS_mag = as_complex(S).abs()
    XX_mag = as_complex(X).abs()
    return (SS_mag / (XX_mag + eps)).clamp(0, 1)


class Stft(nn.Module):
    def __init__(self, n_fft: int, hop: int | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        self.register_buffer("w", torch.hann_window(n_fft))

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[-1]
        sh = x.shape[:-1]
        out = torch.stft(
            x.reshape(-1, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,  # type: ignore[arg-type]
            normalized=True,
            return_complex=True,
        )
        return out.view(*sh, *out.shape[-2:])


class Istft(nn.Module):
    def __init__(self, n_fft: int, hop: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.register_buffer("w", torch.hann_window(n_fft))

    def forward(self, x: Tensor) -> Tensor:
        x = as_complex(x)
        t, f = x.shape[-2], x.shape[-1]
        sh = x.shape[:-2]
        out = torch.istft(
            F.pad(x.reshape(-1, t, f).transpose(1, 2), (0, 1)),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,  # type: ignore[arg-type]
            normalized=True,
        )
        if x.ndim > 2:
            out = out.view(*sh, out.shape[-1])
        return out


def _local_energy(x: Tensor, ws: int, device: torch.device) -> Tensor:
    """Windowed local energy from a real-format spectrogram [B, 1, T, F, 2]."""
    if ws % 2 == 0:
        ws += 1
    ws_half = ws // 2
    energy = x.pow(2).sum(-1)  # [..., F]
    if energy.dim() == 4:  # [B, 1, T, F]
        energy = energy.sum(-1)  # [B, 1, T]
    energy = F.pad(energy, (ws_half, ws_half, 0, 0))
    w = torch.hann_window(ws, device=device, dtype=energy.dtype)
    energy = energy.unfold(-1, ws, 1) * w  # [B, 1, T, ws]
    return energy.sum(-1).div(ws)  # [B, 1, T]


class LocalSnrTarget(nn.Module):
    """Computes ground-truth local SNR from clean and noise spectrograms."""

    def __init__(
        self,
        sr: int,
        fft_size: int,
        hop_size: int,
        ws_ms: int = 20,
        lsnr_range: tuple[float, float] = (-16.0, 36.0),
    ):
        super().__init__()
        self.ws = self._calc_ws(ws_ms, sr, fft_size, hop_size)
        self.ws_ns = self.ws * 2
        self.lsnr_range = lsnr_range

    @staticmethod
    def _calc_ws(ws_ms: int, sr: int, fft_size: int, hop_size: int) -> int:
        fft_ms = fft_size / sr * 1000
        hop_ms = hop_size / sr * 1000
        ws = 1 + (ws_ms - fft_ms) / hop_ms
        return max(int(round(ws)), 1)

    def forward(self, clean: Tensor, noise: Tensor) -> Tensor:
        # clean, noise: [B, 1, T, F, 2]
        x_clean = as_real(clean)
        x_noise = as_real(noise)
        E_speech = _local_energy(x_clean, self.ws, clean.device)  # [B, 1, T]
        E_noise = _local_energy(x_noise, self.ws_ns, clean.device)  # [B, 1, T]
        eps = torch.finfo(clean.dtype).eps
        snr = E_speech / E_noise.clamp_min(eps)
        snr = snr.clamp_min(eps).log10().mul(10)
        lo, hi = self.lsnr_range
        return snr.clamp(lo, hi).squeeze(1)  # [B, T]


class MultiResSpecLoss(nn.Module):
    """Multi-resolution STFT magnitude (+ optional complex) loss."""

    def __init__(
        self,
        n_ffts: list[int],
        gamma: float = 1.0,
        factor: float = 1.0,
        f_complex: float | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.f = factor
        self.stfts = nn.ModuleDict({str(n): Stft(n) for n in n_ffts})
        if f_complex is None or f_complex == 0:
            self.f_complex: list[float] | None = None
        elif isinstance(f_complex, (list, tuple)):
            self.f_complex = list(f_complex)
        else:
            self.f_complex = [float(f_complex)] * len(self.stfts)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros((), device=input.device, dtype=input.dtype)
        for i, stft in enumerate(self.stfts.values()):
            Y = stft(input)
            S = stft(target)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            loss = loss + F.mse_loss(Y_abs, S_abs) * self.f
            if self.f_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle(Y))
                    S = S_abs * torch.exp(1j * angle(S))
                loss = loss + (
                    F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S))
                    * self.f_complex[i]
                )
        return loss


class SpectralLoss(nn.Module):
    """Spectral magnitude (+ optional complex) loss with under-weighting."""

    def __init__(
        self,
        gamma: float = 1.0,
        factor_magnitude: float = 1.0,
        factor_complex: float = 0.0,
        factor_under: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.f_m = factor_magnitude
        self.f_c = factor_complex
        self.f_u = factor_under

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        inp = as_complex(input)
        tgt = as_complex(target)
        i_abs = inp.abs()
        t_abs = tgt.abs()
        if self.gamma != 1:
            i_abs = i_abs.clamp_min(1e-12).pow(self.gamma)
            t_abs = t_abs.clamp_min(1e-12).pow(self.gamma)
        tmp = (i_abs - t_abs).pow(2)
        if self.f_u != 1:
            tmp = tmp * torch.where(i_abs < t_abs, self.f_u, 1.0)
        loss = tmp.mean() * self.f_m
        if self.f_c > 0:
            if self.gamma != 1:
                inp = i_abs * torch.exp(1j * angle(inp))
                tgt = t_abs * torch.exp(1j * angle(tgt))
            loss = (
                loss
                + F.mse_loss(torch.view_as_real(inp), torch.view_as_real(tgt))
                * self.f_c
            )
        return loss


class MaskLoss(nn.Module):
    """ERB-domain mask loss with optional under-weighting and multi-power terms."""

    def __init__(
        self,
        erb_fb: Tensor,
        mask: str = "iam",
        gamma: float = 0.6,
        gamma_pred: float | None = None,
        powers: list[int] | None = None,
        factors: list[float] | None = None,
        f_under: float = 1.0,
        factor: float = 1.0,
    ):
        super().__init__()
        mask_fns = {"wg": wg, "irm": irm, "iam": iam}
        if mask not in mask_fns:
            raise ValueError(f"Unknown mask: '{mask}'. Choose from {list(mask_fns)}")
        self.mask_fn = mask_fns[mask]
        self.gamma = gamma
        self.gamma_pred = gamma if gamma_pred is None else gamma_pred
        self.powers = powers if powers is not None else [2]
        self.factors = factors if factors is not None else [1.0]
        self.f_under = f_under
        self.factor = factor
        self.eps = 1e-12
        self.register_buffer("erb_fb", erb_fb)  # [F, E]

    def _to_erb(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.erb_fb)

    def _target_mask(self, clean: Tensor, noisy: Tensor) -> Tensor:
        mask = self.mask_fn(clean, noisy)  # [B, 1, T, F], values in [0, 1]
        mask = self._to_erb(mask)  # [B, 1, T, E]
        return mask.clamp_min(self.eps).pow(self.gamma)

    def forward(self, input: Tensor, clean: Tensor, noisy: Tensor) -> Tensor:
        """
        Args:
            input: Predicted ERB mask  [B, 1, T, E]
            clean: Clean spectrum      [B, 1, T, F, 2]
            noisy: Noisy spectrum      [B, 1, T, F, 2]
        """
        g_t = self._target_mask(clean, noisy)  # [B, 1, T, E]
        g_p = input.clamp_min(self.eps).pow(self.gamma_pred)  # [B, 1, T, E]
        tmp = g_t.sub(g_p).pow(2)
        if self.f_under != 1:
            tmp = tmp * torch.where(g_p < g_t, self.f_under, 1.0)
        loss = torch.zeros((), device=input.device)
        for power, fac in zip(self.powers, self.factors):
            loss = (
                loss
                + tmp.clamp_min(1e-13).pow(power // 2).mean().mul(fac) * self.factor
            )
        return loss


class SiSdr(nn.Module):
    """Scale-Invariant SNR metric (higher is better)."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input, target: [B, T]
        eps = torch.finfo(input.dtype).eps
        t = input.shape[-1]
        target = target.reshape(-1, t)
        input = input.reshape(-1, t)
        Rss = torch.einsum("bi,bi->b", target, target).unsqueeze(-1)
        a = torch.einsum("bi,bi->b", target, input).add(eps).unsqueeze(-1) / Rss.add(
            eps
        )
        e_true = a * target
        e_res = input - e_true
        Sss = e_true.square().sum(-1)
        Snn = e_res.square().sum(-1)
        return 10 * torch.log10(Sss.add(eps) / Snn.add(eps))


class SdrLoss(nn.Module):
    """Negative Si-SDR loss."""

    def __init__(self, factor: float = 0.2):
        super().__init__()
        self.factor = factor
        self.sdr = SiSdr()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return -self.sdr(input, target).mean() * self.factor


class SegSdrLoss(nn.Module):
    """Multi-window segmental Si-SDR loss."""

    def __init__(
        self, window_sizes: list[int], factor: float = 0.2, overlap: float = 0.0
    ):
        super().__init__()
        self.window_sizes = window_sizes
        self.factor = factor
        self.hop_ratio = 1.0 - overlap
        self.sdr = SiSdr()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input, target: [B, T]
        loss = torch.zeros((), device=input.device)
        for ws in self.window_sizes:
            ws = min(ws, input.size(-1))
            hop = max(int(self.hop_ratio * ws), 1)
            loss = (
                loss
                + self.sdr(
                    input=input.unfold(-1, ws, hop).reshape(-1, ws),
                    target=target.unfold(-1, ws, hop).reshape(-1, ws),
                ).mean()
            )
        return -loss * self.factor


class LocalSnrLoss(nn.Module):
    """MSE loss on the predicted local SNR head."""

    def __init__(self, factor: float = 5e-4):
        super().__init__()
        self.factor = factor

    def forward(self, input: Tensor, target_lsnr: Tensor) -> Tensor:
        # input: [B, T, 1],  target_lsnr: [B, T]
        return F.mse_loss(input.squeeze(-1), target_lsnr) * self.factor


@dataclass
class LossConfig:
    """Configuration for the combined training loss."""

    # MaskLoss
    ml_factor: float = 1.0
    ml_mask: str = "iam"  # "iam" | "irm" | "wg"
    ml_gamma: float = 0.6
    ml_gamma_pred: float = 0.6
    ml_f_under: float = 2.0
    ml_powers: list[int] = field(default_factory=lambda: [2, 4])
    ml_factors: list[float] = field(default_factory=lambda: [1.0, 10.0])

    # SpectralLoss
    sl_factor_magnitude: float = 0.0
    sl_factor_complex: float = 0.0
    sl_factor_under: float = 1.0
    sl_gamma: float = 1.0

    # MultiResSpecLoss
    mrsl_factor: float = 0.0
    mrsl_factor_complex: float = 0.0
    mrsl_gamma: float = 1.0
    mrsl_fft_sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048])

    # SdrLoss
    sdr_factor: float = 0.0
    sdr_segmental_ws: list[int] = field(default_factory=list)

    # LocalSnrLoss
    lsnr_factor: float = 5e-4


class Loss(nn.Module):
    """Combined loss for LightDeepFilterNet training.

    Aggregates MaskLoss, SpectralLoss, MultiResSpecLoss, SdrLoss, and
    LocalSnrLoss according to ``LossConfig``.
    """

    def __init__(
        self,
        cfg: LossConfig,
        erb_fb: Tensor,
        fft_size: int,
        hop_size: int,
        sr: int,
        lsnr_min: int,
        lsnr_max: int,
    ):
        super().__init__()
        self.store_losses = False
        self.summaries: dict[str, list[Tensor]] = defaultdict(list)

        self.lsnr_target = LocalSnrTarget(
            sr=sr,
            fft_size=fft_size,
            hop_size=hop_size,
            lsnr_range=(float(lsnr_min) - 1.0, float(lsnr_max) + 1.0),
        )

        needs_td = cfg.mrsl_factor > 0 or cfg.sdr_factor > 0
        self.istft: Istft | None = Istft(fft_size, hop_size) if needs_td else None

        self.ml_f = cfg.ml_factor
        self.ml: MaskLoss | None = (
            MaskLoss(
                erb_fb=erb_fb,
                mask=cfg.ml_mask,
                gamma=cfg.ml_gamma,
                gamma_pred=cfg.ml_gamma_pred,
                f_under=cfg.ml_f_under,
                powers=cfg.ml_powers,
                factors=cfg.ml_factors,
                factor=cfg.ml_factor,
            )
            if cfg.ml_factor != 0
            else None
        )

        self.sl_f = cfg.sl_factor_magnitude + cfg.sl_factor_complex
        self.sl: SpectralLoss | None = (
            SpectralLoss(
                gamma=cfg.sl_gamma,
                factor_magnitude=cfg.sl_factor_magnitude,
                factor_complex=cfg.sl_factor_complex,
                factor_under=cfg.sl_factor_under,
            )
            if self.sl_f > 0
            else None
        )

        self.mrsl_f = cfg.mrsl_factor
        self.mrsl: MultiResSpecLoss | None = (
            MultiResSpecLoss(
                cfg.mrsl_fft_sizes,
                cfg.mrsl_gamma,
                cfg.mrsl_factor,
                cfg.mrsl_factor_complex or None,
            )
            if cfg.mrsl_factor > 0
            else None
        )

        self.sdr_f = cfg.sdr_factor
        self.sdrl: nn.Module | None = None
        if cfg.sdr_factor > 0:
            if cfg.sdr_segmental_ws:
                self.sdrl = SegSdrLoss(cfg.sdr_segmental_ws, factor=cfg.sdr_factor)
            else:
                self.sdrl = SdrLoss(cfg.sdr_factor)

        self.lsnr_f = cfg.lsnr_factor
        self.lsnrl: LocalSnrLoss | None = (
            LocalSnrLoss(cfg.lsnr_factor) if cfg.lsnr_factor > 0 else None
        )

    def forward(
        self,
        clean: Tensor,
        noisy: Tensor,
        enhanced: Tensor,
        mask: Tensor,
        lsnr: Tensor,
        snrs: Tensor,
    ) -> Tensor:
        """Compute the combined training loss.

        Args:
            clean:    Clean complex spectrum   [B, 1, T, F, 2]
            noisy:    Noisy complex spectrum   [B, 1, T, F, 2]
            enhanced: Enhanced spectrum        [B, 1, T, F, 2]
            mask:     Predicted ERB mask       [B, 1, T, E]
            lsnr:     Predicted local SNR      [B, T, 1]
            snrs:     Per-sample input SNRs    [B]
        """
        noise = noisy - clean
        lsnr_gt = self.lsnr_target(clean, noise)  # [B, T]

        ml = sl = mrsl = sdrl = lsnrl = torch.zeros((), device=clean.device)

        if self.ml is not None:
            ml = self.ml(mask, clean, noisy)

        if self.sl is not None:
            sl = self.sl(enhanced, clean)

        enh_td: Tensor | None = None
        clean_td: Tensor | None = None
        if self.istft is not None and (self.mrsl is not None or self.sdrl is not None):
            enh_td = self.istft(enhanced)
            clean_td = self.istft(clean)

        if self.mrsl is not None and enh_td is not None and clean_td is not None:
            mrsl = self.mrsl(enh_td, clean_td)

        if self.sdrl is not None and enh_td is not None and clean_td is not None:
            sdrl = self.sdrl(enh_td.squeeze(1), clean_td.squeeze(1))

        if self.lsnrl is not None:
            lsnrl = self.lsnrl(lsnr, lsnr_gt)

        if self.store_losses:
            self._store_summaries(ml, sl, mrsl, sdrl, lsnrl)

        return ml + sl + mrsl + sdrl + lsnrl

    def reset_summaries(self) -> None:
        self.summaries = defaultdict(list)

    @torch.no_grad()
    def _store_summaries(
        self,
        ml: Tensor,
        sl: Tensor,
        mrsl: Tensor,
        sdrl: Tensor,
        lsnrl: Tensor,
    ) -> None:
        for name, val in [
            ("MaskLoss", ml),
            ("SpectralLoss", sl),
            ("MultiResSpecLoss", mrsl),
            ("SdrLoss", sdrl),
            ("LocalSnrLoss", lsnrl),
        ]:
            if val.item() != 0:
                self.summaries[name].append(val.detach())
