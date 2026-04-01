"""LightDeepFilterNet implementation with Li-GRU instead of standard GRU.

This module implements the LightDeepFilterNet architecture, a lightweight version
based on DeepFilterNet3 that replaces all GRU layers with Li-GRU layers for improved efficiency.
"""

from functools import partial

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn
from typing_extensions import Final

from src.configs.config import ModelConfig
from src.model.modules import (
    Add,
    Concat,
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    GroupedLinearEinsum,
    Mask,
    SqueezedLiGRU_S,
)
from src.utils.erb import get_erb_filterbanks
from src.utils.io import get_device

PI = 3.1415926535897932384626433
eps = 1e-12


class LightEncoder(nn.Module):
    """Encoder module with Li-GRU for LightDeepFilterNet.

    Args:
        config (ModelConfig): Model architecture configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialise LightEncoder layers."""
        super().__init__()
        assert config.nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        self.erb_conv0 = Conv2dNormAct(
            in_ch=1,
            out_ch=config.conv_ch,
            kernel_size=config.conv_kernel_inp,
            bias=False,
            separable=True,
        )
        conv_layer = partial(
            Conv2dNormAct,
            in_ch=config.conv_ch,
            out_ch=config.conv_ch,
            kernel_size=config.conv_kernel,
            bias=False,
            separable=True,
        )
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)
        self.df_conv0 = Conv2dNormAct(
            in_ch=2,
            out_ch=config.conv_ch,
            kernel_size=config.conv_kernel_inp,
            bias=False,
            separable=True,
        )
        self.df_conv1 = conv_layer(fstride=2)
        self.erb_bins = config.nb_erb
        self.nb_spec = (
            config.nb_df
        )  # F dim of feat_spec input (before df_conv1 halving)
        self.conv_ch = config.conv_ch
        self.emb_in_dim = config.conv_ch * config.nb_erb // 4
        self.emb_dim = config.emb_hidden_dim
        self.emb_out_dim = config.conv_ch * config.nb_erb // 4
        df_fc_emb = GroupedLinearEinsum(
            config.conv_ch * config.nb_df // 2,
            self.emb_in_dim,
            groups=config.enc_lin_groups,
        )
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))
        if config.enc_concat:
            self.emb_in_dim *= 2
            self.combine = Concat()
        else:
            self.combine = Add()
        self.emb_n_layers = config.emb_num_layers
        if config.emb_gru_skip_enc == "none":
            skip_op = None
        elif config.emb_gru_skip_enc == "identity":
            assert self.emb_in_dim == self.emb_out_dim, "Dimensions do not match"
            skip_op = partial(nn.Identity)
        elif config.emb_gru_skip_enc == "groupedlinear":
            skip_op = partial(
                GroupedLinearEinsum,
                input_size=self.emb_out_dim,
                hidden_size=self.emb_out_dim,
                groups=config.lin_groups,
            )
        else:
            raise NotImplementedError()

        self.emb_gru = SqueezedLiGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=1,
            batch_first=True,
            gru_skip_op=skip_op,
            linear_groups=config.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
            batch_size=config.batch_size,
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = config.lsnr_max - config.lsnr_min
        self.lsnr_offset = config.lsnr_min

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode ERB and complex spectrogram features into embeddings and LSNR estimate.

        Args:
            feat_erb (Tensor): ERB features in dB scale, shape ``[B, 1, T, Fe]``.
            feat_spec (Tensor): Complex spectrogram features, shape ``[B, 2, T, Fc]``.

        Returns:
            tuple[Tensor, ...]: ``(e0, e1, e2, e3, emb, c0, lsnr)`` encoder activations
            and local SNR estimate of shape ``[B, T, 1]``.
        """
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/4]
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc/2]

        cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, -1]
        cemb = self.df_fc_emb(cemb)  # [B, T, C * F/4]
        emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C * F]
        emb = self.combine(emb, cemb)
        emb, _ = self.emb_gru(emb)  # [B, T, -1]
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


class LightErbDecoder(nn.Module):
    """ERB Decoder module with Li-GRU for LightDeepFilterNet.

    Args:
        config (ModelConfig): Model architecture configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialise LightErbDecoder layers."""
        super().__init__()
        assert config.nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        self.emb_in_dim = config.conv_ch * config.nb_erb // 4
        self.emb_dim = config.emb_hidden_dim
        self.emb_out_dim = config.conv_ch * config.nb_erb // 4

        if config.emb_gru_skip == "none":
            skip_op = None
        elif config.emb_gru_skip == "identity":
            assert self.emb_in_dim == self.emb_out_dim, "Dimensions do not match"
            skip_op = partial(nn.Identity)
        elif config.emb_gru_skip == "groupedlinear":
            skip_op = partial(
                GroupedLinearEinsum,
                input_size=self.emb_in_dim,
                hidden_size=self.emb_out_dim,
                groups=config.lin_groups,
            )
        else:
            raise NotImplementedError()

        self.emb_gru = SqueezedLiGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=config.emb_num_layers - 1,
            batch_first=True,
            gru_skip_op=skip_op,
            linear_groups=config.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
            batch_size=config.batch_size,
        )
        tconv_layer = partial(
            ConvTranspose2dNormAct,
            kernel_size=config.convt_kernel,
            bias=False,
            separable=True,
        )
        conv_layer = partial(
            Conv2dNormAct,
            bias=False,
            separable=True,
        )
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = conv_layer(config.conv_ch, config.conv_ch, kernel_size=1)
        self.convt3 = conv_layer(
            config.conv_ch, config.conv_ch, kernel_size=config.conv_kernel
        )
        self.conv2p = conv_layer(config.conv_ch, config.conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(config.conv_ch, config.conv_ch, fstride=2)
        self.conv1p = conv_layer(config.conv_ch, config.conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(config.conv_ch, config.conv_ch, fstride=2)
        self.conv0p = conv_layer(config.conv_ch, config.conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(
            config.conv_ch,
            1,
            kernel_size=config.conv_kernel,
            activation_layer=nn.Sigmoid,
        )

    def forward(
        self, emb: Tensor, e3: Tensor, e2: Tensor, e1: Tensor, e0: Tensor
    ) -> Tensor:
        """Decode encoder activations into an ERB gain mask.

        Args:
            emb (Tensor): Shared embedding from the encoder, shape ``[B, T, emb_dim]``.
            e3 (Tensor): Encoder skip connection at stride 4, shape ``[B, C, T, F/4]``.
            e2 (Tensor): Encoder skip connection at stride 2, shape ``[B, C, T, F/2]``.
            e1 (Tensor): Encoder skip connection at stride 1, shape ``[B, C, T, F]``.
            e0 (Tensor): Encoder input features, shape ``[B, C, T, F]``.

        Returns:
            Tensor: ERB gain mask of shape ``[B, 1, T, nb_erb]``.
        """
        b, _, t, f8 = e3.shape
        emb, _ = self.emb_gru(emb)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C*8, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C*4, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C*2, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m


class LightDfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule.

    Requires input of shape B, C, T, F, 2.

    Args:
        df_order (int): Deep filtering order (number of filter taps).
        df_bins (int): Number of frequency bins processed by deep filtering.
    """

    def __init__(self, df_order: int, df_bins: int) -> None:
        """Initialise LightDfOutputReshapeMF."""
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        """Reshape raw DF output into complex coefficient tensors.

        Args:
            coefs (Tensor): Raw coefficients of shape ``[B, T, F, O*2]``.

        Returns:
            Tensor: Reshaped coefficients of shape ``[B, O, T, F, 2]``.
        """
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs


class LightDfDecoder(nn.Module):
    """Deep Filtering Decoder with Li-GRU.

    Args:
        config (ModelConfig): Model architecture configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialise LightDfDecoder layers."""
        super().__init__()
        layer_width = config.conv_ch

        self.emb_in_dim = config.conv_ch * config.nb_erb // 4
        self.emb_dim = config.df_hidden_dim

        self.df_n_hidden = config.df_hidden_dim
        self.df_n_layers = config.df_num_layers
        self.df_order = config.df_order
        self.df_bins = config.nb_df
        self.df_out_ch = config.df_order * 2

        conv_layer = partial(Conv2dNormAct, separable=True, bias=False)
        kt = config.df_pathway_kernel_size_t
        self.df_convp = conv_layer(
            layer_width, self.df_out_ch, fstride=1, kernel_size=(kt, 1)
        )

        self.df_gru = SqueezedLiGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            num_layers=self.df_n_layers,
            batch_first=True,
            gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
            batch_size=config.batch_size,
        )
        config.df_gru_skip = config.df_gru_skip.lower()
        assert config.df_gru_skip in ("none", "identity", "groupedlinear")
        self.df_skip: nn.Module | None
        if config.df_gru_skip == "none":
            self.df_skip = None
        elif config.df_gru_skip == "identity":
            assert config.emb_hidden_dim == config.df_hidden_dim, (
                "Dimensions do not match"
            )
            self.df_skip = nn.Identity()
        elif config.df_gru_skip == "groupedlinear":
            self.df_skip = GroupedLinearEinsum(
                self.emb_in_dim, self.emb_dim, groups=config.lin_groups
            )
        else:
            raise NotImplementedError()
        self.df_out: nn.Module
        out_dim = self.df_bins * self.df_out_ch
        df_out = GroupedLinearEinsum(
            self.df_n_hidden, out_dim, groups=config.lin_groups
        )
        self.df_out = nn.Sequential(df_out, nn.Tanh())

    def forward(self, emb: Tensor, c0: Tensor) -> Tensor:
        """Compute per-frequency DF filter coefficients.

        Args:
            emb (Tensor): Shared encoder embedding, shape ``[B, T, emb_dim]``.
            c0 (Tensor): DF pathway conv features, shape ``[B, C, T, Fc]``.

        Returns:
            Tensor: DF coefficients of shape ``[B, T, nb_df, df_order*2]``.
        """
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)  # [B, T, H], H: df_n_hidden
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O*2], channels_last
        c = self.df_out(c)  # [B, T, F*O*2], O: df_order
        c = c.view(b, t, self.df_bins, self.df_out_ch) + c0  # [B, T, F, O*2]
        return c


class LightDeepFilteringModule(nn.Module):
    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.lookahead = lookahead
        self._pad_before = frame_size - 1 - lookahead
        self._pad_after = lookahead

    def forward(self, spec: Tensor, coefs: Tensor) -> Tensor:
        """
        spec:  [B, 1, T, F, 2]
        coefs: [B, O, T, F_df, 2]
        """
        O_len = self.frame_size
        F_df = self.num_freqs

        # Build temporal window: [B, 1, T, F, O, 2]
        if O_len > 1:
            spec_padded = F.pad(
                spec, [0, 0, 0, 0, self._pad_before, self._pad_after]
            )  # [B, 1, T+pad, F, 2]

            # Replace unfold (unsupported by TorchScript ONNX exporter when the
            # input size is not statically accessible) with explicit slice stacking.
            # range(O_len) is a static Python range so the loop is unrolled at trace
            # time, producing O plain Slice ops instead of a dynamic Unfold.
            T = spec.shape[2]
            spec_win = torch.stack(
                [spec_padded[:, :, i : i + T, :, :] for i in range(O_len)],
                dim=4,
            )  # [B, 1, T, F, O, 2]
        else:
            spec_win = spec.unsqueeze(-2)  # [B, 1, T, F, 1, 2]

        # Only DF bins
        spec_win = spec_win[:, :, :, :F_df, :, :]  # [B, 1, T, F_df, O, 2]

        # Reorder coefs to match spec layout: [B, 1, T, F_df, O, 2]
        coefs = coefs.permute(0, 2, 3, 1, 4).unsqueeze(1)

        # Real/imag parts
        sr = spec_win[..., 0]
        si = spec_win[..., 1]
        cr = coefs[..., 0]
        ci = coefs[..., 1]

        # Complex multiply + sum over frame axis O
        result_r = (sr * cr - si * ci).sum(dim=-1)  # [B, 1, T, F_df]
        result_i = (sr * ci + si * cr).sum(dim=-1)  # [B, 1, T, F_df]

        filtered = torch.stack((result_r, result_i), dim=-1)  # [B, 1, T, F_df, 2]

        # Concatenate filtered low-freq bins with unmodified high-freq bins.
        # torch.cat avoids cloning the entire spectrogram just to overwrite part of it.
        return torch.cat([filtered, spec[:, :, :, F_df:, :]], dim=3)


class LightDeepFilterNet(nn.Module):
    """LightDeepFilterNet main model with Li-GRU.

    This is a lightweight version based on DeepFilterNet3 architecture,
    replacing all GRU layers with Li-GRU layers for improved efficiency.
    """

    run_df: Final[bool]
    run_erb: Final[bool]

    def __init__(
        self,
        config: ModelConfig,
        erb_fb_tensor: Tensor | None = None,
        erb_inv_fb_tensor: Tensor | None = None,
        run_df: bool = True,
        train_mask: bool = True,
    ) -> None:
        """Initialise LightDeepFilterNet.

        Args:
            config (ModelConfig): Model architecture configuration.
            erb_fb_tensor (Tensor | None): Pre-computed ERB filterbank matrix.
            erb_inv_fb_tensor (Tensor | None): Pre-computed inverse ERB filterbank matrix.
            run_df (bool): Whether to run the deep filtering branch.
            train_mask (bool): Whether to train the ERB mask decoder.
        """
        super().__init__()
        layer_width = config.conv_ch
        assert config.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.df_lookahead = config.df_lookahead
        self.nb_df = config.nb_df
        self.freq_bins: int = config.fft_size // 2 + 1
        self.emb_dim: int = layer_width * config.nb_erb
        self.erb_bins: int = config.nb_erb

        # Padding layers
        if config.conv_lookahead > 0:
            assert config.conv_lookahead >= config.df_lookahead
            self.pad_feat = nn.ConstantPad2d(
                (0, 0, -config.conv_lookahead, config.conv_lookahead), 0.0
            )
        else:
            self.pad_feat = nn.Identity()
        if config.df_lookahead > 0:
            self.pad_spec = nn.ConstantPad3d(
                (0, 0, 0, 0, -config.df_lookahead, config.df_lookahead), 0.0
            )
        else:
            self.pad_spec = nn.Identity()

        # Initialize ERB filterbanks if not provided
        self.register_buffer("erb_fb", erb_fb_tensor)
        self.enc = LightEncoder(config)
        self.erb_dec = LightErbDecoder(config)
        self.mask = Mask(erb_inv_fb_tensor)
        self.erb_inv_fb = erb_inv_fb_tensor
        self.post_filter = config.mask_pf
        self.post_filter_beta = config.pf_beta

        self.df_order = config.df_order
        self.df_op = LightDeepFilteringModule(
            num_freqs=config.nb_df,
            frame_size=config.df_order,
            lookahead=self.df_lookahead,
        )
        self.df_dec = LightDfDecoder(config)
        self.df_out_transform = LightDfOutputReshapeMF(self.df_order, config.nb_df)

        self.run_erb = config.nb_df + 1 < self.freq_bins
        self.run_df = run_df
        self.train_mask = train_mask
        self.lsnr_dropout = config.lsnr_dropout

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,
        atten_lim: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward method of DeepFilterNet3 with Li-GRU.

        Args:
            spec (Tensor): Spectrum of shape [B, 1, T, F, 2]
            feat_erb (Tensor): ERB features of shape [B, 1, T, E]
            feat_spec (Tensor): Complex spectrogram features of shape [B, 1, T, F', 2]
            atten_lim (Tensor | None): Per-sample attenuation limit in dB, shape [B].
                Clamps the ERB mask from below so suppression never exceeds this limit.
                E.g. 12 dB means at most 12 dB of noise suppression.

        Returns:
            spec (Tensor): Enhanced spectrum of shape [B, 1, T, F, 2]
            m (Tensor): ERB mask estimate of shape [B, 1, T, E]
            lsnr (Tensor): Local SNR estimate of shape [B, T, 1]
            df_coefs (Tensor): DF coefficients
        """
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)

        feat_erb = self.pad_feat(feat_erb)
        feat_spec = self.pad_feat(feat_spec)
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)

        # LSNR dropout: skip processing for low SNR frames
        use_lsnr_dropout = self.lsnr_dropout and self.training
        if use_lsnr_dropout:
            idcs = lsnr.squeeze() > -10.0
            b, t = spec.shape[0], spec.shape[2]
            m = torch.zeros((b, 1, t, self.erb_bins), device=spec.device)
            df_coefs = torch.zeros(
                (b, t, self.nb_df, self.df_order * 2), device=spec.device
            )
            emb = emb[:, idcs]
            e0 = e0[:, :, idcs]
            e1 = e1[:, :, idcs]
            e2 = e2[:, :, idcs]
            e3 = e3[:, :, idcs]
            c0 = c0[:, :, idcs]

        if self.run_erb:
            if use_lsnr_dropout:
                m[:, :, idcs] = self.erb_dec(emb, e3, e2, e1, e0)
                spec_m = self.mask(spec, m, atten_lim)
            else:
                m = self.erb_dec(emb, e3, e2, e1, e0)
                spec_m = self.mask(spec, m, atten_lim)
        else:
            m = torch.zeros((), device=spec.device)
            spec_m = torch.zeros_like(spec)

        if self.run_df:
            if use_lsnr_dropout:
                df_coefs[:, idcs] = self.df_dec(emb, c0)
            else:
                df_coefs = self.df_dec(emb, c0)
            df_coefs = self.df_out_transform(df_coefs)
            spec_e = self.df_op(spec, df_coefs)
            # Replace in-place index assignment (creates onnx::Placeholder /
            # index_put_ which is not supported by the TorchScript ONNX exporter)
            # with a functional torch.cat over the frequency axis.
            spec_e = torch.cat(
                [spec_e[..., : self.nb_df, :], spec_m[..., self.nb_df :, :]], dim=-2
            )
        else:
            df_coefs = torch.zeros((), device=spec.device)
            spec_e = spec_m

        # Post-filter
        if self.post_filter and not self.training:
            spec_complex = torch.view_as_complex(spec_e)
            spec_orig_complex = torch.view_as_complex(spec)
            mask = (spec_complex.abs() / spec_orig_complex.abs().add(eps)).clamp(eps, 1)
            mask_sin = mask * torch.sin(PI * mask / 2).clamp_min(eps)
            pf = (1 + self.post_filter_beta) / (
                1 + self.post_filter_beta * mask.div(mask_sin).pow(2)
            )
            spec_e = spec_e * pf.unsqueeze(-1)

        return spec_e, m, lsnr, df_coefs


def init_model(
    model_config: ModelConfig, run_df: bool = True, train_mask: bool = True
) -> LightDeepFilterNet:
    """Build and move a :class:`LightDeepFilterNet` to the available device.

    Args:
        model_config (ModelConfig): Model architecture configuration.
        run_df (bool): Whether to enable the deep filtering branch.
        train_mask (bool): Whether to train the ERB mask decoder.

    Returns:
        LightDeepFilterNet: Initialised model on the target device.
    """
    erb_fb_tensor, erb_inv_fb_tensor = get_erb_filterbanks(
        sr=model_config.sr,
        fft_size=model_config.fft_size,
        nb_erb=model_config.nb_erb,
        min_nb_freqs=model_config.min_nb_freqs,
    )

    model = LightDeepFilterNet(
        model_config, erb_fb_tensor, erb_inv_fb_tensor, run_df, train_mask
    )
    return model.to(device=get_device())


if __name__ == "__main__":
    # Simple test for LightDeepFilterNet forward pass.
    from src.configs.config import load_config
    from src.utils.utils import count_parameters

    model_config, _, _, _, _ = load_config()

    model = init_model(model_config)
    model.eval()
    device = get_device()

    # Random inputs
    B, T, n_freq = 1, 10, 481
    spec = torch.randn(B, 1, T, n_freq, 2, device=device)
    feat_erb = torch.randn(B, 1, T, 32, device=device)
    feat_spec = torch.randn(B, 1, T, 96, 2, device=device)

    # Forward
    with torch.no_grad():
        enhanced_spec, erb_feat, lsnr, df_coefs = model(spec, feat_erb, feat_spec)

    logger.info("Forward pass OK")
    logger.info(f"  enhanced_spec: {list(enhanced_spec.shape)}")
    logger.info(f"  erb_feat: {list(erb_feat.shape)}")
    logger.info(f"  lsnr: {list(lsnr.shape)}")
    logger.info(f"  df_coefs: {list(df_coefs.shape)}")
    logger.info(f"Number of parameters: {count_parameters(model)}")
