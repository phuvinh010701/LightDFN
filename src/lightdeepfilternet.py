"""LightDeepFilterNet implementation with Li-GRU instead of standard GRU.

This module implements the LightDeepFilterNet architecture, a lightweight version
based on DeepFilterNet3 that replaces all GRU layers with Li-GRU layers for improved efficiency.
"""

from functools import partial
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from typing_extensions import Final

from src.config import ModelConfig, config
from src.erb import get_erb_filterbanks
from src.modules import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    GroupedLinearEinsum,
    Mask,
    SqueezedLiGRU_S,
    get_device,
)

PI = 3.1415926535897932384626433
eps = 1e-12


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)


class LightEncoder(nn.Module):
    """Encoder module with Li-GRU for LightDeepFilterNet."""

    def __init__(self, config: ModelConfig = config):
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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encodes erb; erb should be in dB scale + normalized; Fe are number of erb bands.
        # erb: [B, 1, T, Fe]
        # spec: [B, 2, T, Fc]
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/4]
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc/2]

        # Debug shapes (disabled)
        # print(f"c1 shape before permute: {c1.shape}")
        cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, -1]
        # print(f"cemb shape after flatten: {cemb.shape}, expected last dim: {self.emb_in_dim}")

        cemb = self.df_fc_emb(cemb)  # [T, B, C * F/4]
        emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C * F]
        emb = self.combine(emb, cemb)
        emb, _ = self.emb_gru(emb)  # [B, T, -1]
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


class LightErbDecoder(nn.Module):
    """ERB Decoder module with Li-GRU for LightDeepFilterNet."""

    def __init__(self, config: ModelConfig = config):
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

        # Using Li-GRU instead of standard GRU
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
        # Estimates erb mask
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
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs


class LightDfDecoder(nn.Module):
    """Deep Filtering Decoder with Li-GRU."""

    def __init__(self, config: ModelConfig):
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

        # Using Li-GRU instead of standard GRU
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
        self.df_skip: Optional[nn.Module]
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
        self.df_fc_a = nn.Sequential(nn.Linear(self.df_n_hidden, 1), nn.Sigmoid())

    def forward(self, emb: Tensor, c0: Tensor) -> Tensor:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)  # [B, T, H], H: df_n_hidden
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O*2], channels_last
        c = self.df_out(c)  # [B, T, F*O*2], O: df_order
        c = c.view(b, t, self.df_bins, self.df_out_ch) + c0  # [B, T, F, O*2]
        return c


class LightDeepFilteringModule(nn.Module):
    """Optimized deep filtering using unfold + einsum"""

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        super().__init__()
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.lookahead = lookahead
        # Padding for multi-frame filtering
        self.pad = nn.ConstantPad2d((0, 0, frame_size - 1 - lookahead, lookahead), 0.0)

    def forward(self, spec: Tensor, coefs: Tensor) -> Tensor:
        """Apply deep filtering coefficients to spectrogram.

        Args:
            spec: Complex spectrogram [B, 1, T, F, 2] where F is total frequency bins
            coefs: Filter coefficients [B, O, T, F_df, 2] where F_df is DF bins (subset)

        Returns:
            Filtered spectrogram [B, 1, T, F_df, 2] - only processes DF bins
        """
        # Convert to complex for efficient operations
        spec_complex = torch.view_as_complex(spec)  # [B, 1, T, F]
        coefs_complex = torch.view_as_complex(coefs)  # [B, O, T, F_df]

        # Apply padding and unfold to get overlapping frames
        # spec_complex: [B, 1, T, F] -> pad -> [B, 1, T+pad, F] -> unfold -> [B, 1, T, F, O]
        if self.frame_size > 1:
            spec_padded = self.pad(spec_complex)
            spec_unfolded = spec_padded.unfold(2, self.frame_size, 1)  # [B, 1, T, F, O]
        else:
            spec_unfolded = spec_complex.unsqueeze(-1)  # [B, 1, T, F, 1]

        # Extract only DF bins
        spec_f = spec_unfolded.narrow(-2, 0, self.num_freqs)  # [B, 1, T, F_df, O]

        # Reshape coefs: [B, O, T, F_df] -> [B, 1, O, T, F_df]
        coefs_reshaped = coefs_complex.view(
            coefs_complex.shape[0],
            -1,
            self.frame_size,
            coefs_complex.shape[2],
            coefs_complex.shape[3],
        )  # [B, 1, O, T, F_df]

        # Apply deep filtering using einsum
        # spec_f: [B, 1, T, F_df, O]
        # coefs_reshaped: [B, 1, O, T, F_df]
        # Result: [B, 1, T, F_df]
        spec_filtered = torch.einsum("...tfn,...ntf->...tf", spec_f, coefs_reshaped)

        spec[..., : self.num_freqs, :] = torch.view_as_real(spec_filtered)
        return spec


class LightDeepFilterNet(nn.Module):
    """LightDeepFilterNet main model with Li-GRU.

    This is a lightweight version based on DeepFilterNet3 architecture,
    replacing all GRU layers with Li-GRU layers for improved efficiency.
    """

    run_df: Final[bool]
    run_erb: Final[bool]

    def __init__(
        self,
        config: ModelConfig = config,
        erb_fb_tensor: Optional[Tensor] = None,
        erb_inv_fb_tensor: Optional[Tensor] = None,
        run_df: bool = True,
        train_mask: bool = True,
    ):
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
        self.enc = LightEncoder()
        self.erb_dec = LightErbDecoder()
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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward method of DeepFilterNet3 with Li-GRU.

        Args:
            spec (Tensor): Spectrum of shape [B, 1, T, F, 2]
            feat_erb (Tensor): ERB features of shape [B, 1, T, E]
            feat_spec (Tensor): Complex spectrogram features of shape [B, 1, T, F', 2]

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
            spec_m = spec.clone()
            emb = emb[:, idcs]
            e0 = e0[:, :, idcs]
            e1 = e1[:, :, idcs]
            e2 = e2[:, :, idcs]
            e3 = e3[:, :, idcs]
            c0 = c0[:, :, idcs]

        if self.run_erb:
            if use_lsnr_dropout:
                m[:, :, idcs] = self.erb_dec(emb, e3, e2, e1, e0)
                spec_m = self.mask(spec, m)
            else:
                m = self.erb_dec(emb, e3, e2, e1, e0)
                spec_m = self.mask(spec, m)
        else:
            m = torch.zeros((), device=spec.device)
            spec_m = torch.zeros_like(spec)

        if self.run_df:
            if use_lsnr_dropout:
                df_coefs[:, idcs] = self.df_dec(emb, c0)
            else:
                df_coefs = self.df_dec(emb, c0)
            df_coefs = self.df_out_transform(df_coefs)
            spec_e = self.df_op(spec.clone(), df_coefs)
            spec_e[..., self.nb_df :, :] = spec_m[..., self.nb_df :, :]
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


def init_model(run_df: bool = True, train_mask: bool = True):
    # Generate proper ERB filterbanks
    erb_fb_tensor, erb_inv_fb_tensor = get_erb_filterbanks(
        sr=config.sr, fft_size=config.fft_size, nb_erb=config.nb_erb
    )

    model = LightDeepFilterNet(
        config, erb_fb_tensor, erb_inv_fb_tensor, run_df, train_mask
    )
    return model.to(device=get_device())


if __name__ == "__main__":
    """Simple test for LightDeepFilterNet forward pass."""
    # Init model
    from src.utils import count_parameters

    model = init_model()
    model.eval()
    device = get_device()

    # Random inputs
    B, T, F = 1, 10, 481
    spec = torch.randn(B, 1, T, F, 2, device=device)
    feat_erb = torch.randn(B, 1, T, 32, device=device)
    feat_spec = torch.randn(B, 1, T, 96, 2, device=device)

    # Forward
    with torch.no_grad():
        enhanced_spec, erb_feat, lsnr, df_coefs = model(spec, feat_erb, feat_spec)

    print("✅ Forward pass OK")
    print(f"   enhanced_spec: {list(enhanced_spec.shape)}")
    print(f"   erb_feat: {list(erb_feat.shape)}")
    print(f"   lsnr: {list(lsnr.shape)}")
    print(f"   df_coefs: {list(df_coefs.shape)}")
    print(f"Number of parameters: {count_parameters(model)}")
