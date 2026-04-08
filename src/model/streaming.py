"""Streaming wrapper for LightDeepFilterNet single-frame ONNX inference.

Exposes all GRU hidden states AND causal convolution buffers as explicit
ONNX inputs/outputs so they can be threaded between consecutive hops in a
real-time pipeline.

The wrapper processes one STFT frame (T=1) per call.  Because the original
model uses conv_lookahead > 0 via pad_feat, that look-ahead is dropped here;
quality is slightly lower than offline mode but the model runs with one-hop
(480-sample) latency.

Three encoder convolutions have temporal kernel > 1 and use causal
zero-padding internally.  For correct streaming we replace that zero-padding
with explicit input buffers that carry the previous frames:

  buf_erb0  : [1, 1,  2, nb_erb]    — input buffer for enc.erb_conv0 (kt=3)
  buf_df0   : [1, 2,  2, nb_spec]   — input buffer for enc.df_conv0  (kt=3)
  buf_dfp   : [1, C,  4, nb_spec]   — input buffer for df_dec.df_convp (kt=5)

Hidden state shapes (batch=1, by default) — nn.GRU convention [num_layers, B, H]:
    h_enc : [enc_layers,  1, emb_hidden_dim]  (encoder emb_gru)
    h_erb : [erb_layers,  1, emb_hidden_dim]  (ERB-decoder emb_gru)
    h_df  : [df_layers,   1, df_hidden_dim]   (DF-decoder df_gru)

With default config (emb_num_layers=3, df_num_layers=2, hidden=128):
    h_enc : [1, 1, 128]
    h_erb : [2, 1, 128]
    h_df  : [2, 1, 128]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.lightdeepfilternet import LightDeepFilterNet


def _non_pad_sequential(seq: nn.Sequential) -> nn.Sequential:
    """Return a new Sequential with ConstantPad2d layers stripped out.

    Called once at init to pre-cache the layers used in ``_conv_with_buf``,
    avoiding an ``isinstance`` check on every streaming frame.
    """
    return nn.Sequential(
        *[layer for layer in seq if not isinstance(layer, nn.ConstantPad2d)]
    )


def _conv_with_buf(
    conv_layers: nn.Sequential,
    buf: Tensor,
    x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply a causal conv block using an explicit context buffer.

    Args:
        conv_layers: Pre-filtered Sequential (no ConstantPad2d) from
            :func:`_non_pad_sequential`.  Built once at init.
        buf:  ``[B, C_in, kt-1, F]`` — past frames kept from the previous call.
        x:    ``[B, C_in, 1, F]``    — current single frame.

    Returns:
        out     : ``[B, C_out, 1, F_out]`` — conv output for the current frame.
        new_buf : ``[B, C_in, kt-1, F]``   — updated buffer (slide window by 1).
    """
    ctx = torch.cat([buf, x], dim=2)
    h = conv_layers(ctx)
    # Slice ctx instead of a second torch.cat — zero-copy view, saves one allocation.
    new_buf = ctx[:, :, 1:, :]
    return h, new_buf


class StreamingLightDFN(nn.Module):
    """Single-frame streaming wrapper around LightDeepFilterNet.

    Args:
        model: A pre-loaded, eval-mode LightDeepFilterNet instance.
    """

    def __init__(self, model: LightDeepFilterNet) -> None:
        super().__init__()
        self.model = model
        # Pre-cache conv layers with ConstantPad2d stripped — avoids per-frame isinstance checks.
        self._erb0_layers = _non_pad_sequential(model.enc.erb_conv0)
        self._df0_layers = _non_pad_sequential(model.enc.df_conv0)
        self._dfp_layers = _non_pad_sequential(model.df_dec.df_convp)

    def forward(
        self,
        spec: Tensor,  # [1, 1, 1, F, 2]
        feat_erb: Tensor,  # [1, 1, 1, E]
        feat_spec: Tensor,  # [1, 1, 1, F', 2]
        h_enc: Tensor,  # [enc_layers, 1, emb_hidden_dim]
        h_erb: Tensor,  # [erb_layers, 1, emb_hidden_dim]
        h_df: Tensor,  # [df_layers,  1, df_hidden_dim]
        buf_erb0: Tensor,  # [1, 1, 2, nb_erb]
        buf_df0: Tensor,  # [1, 2, 2, nb_spec]
        buf_dfp: Tensor,  # [1, conv_ch, 4, nb_spec]
        buf_spec: Tensor,  # [1, 1, _pad_before, F, 2]  causal spec context for df_op
        atten_lim_db: float | None = None,  # noise attenuation limit in dB
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Process one STFT frame.

        Args:
            atten_lim_db: Optional attenuation limit in dB. E.g. 12 means at most 12 dB
                of noise suppression — the remaining noise is mixed back in.

        Returns:
            enhanced_spec  : [1, 1, 1, F, 2]
            h_enc_new      : updated encoder hidden state
            h_erb_new      : updated ERB-decoder hidden state
            h_df_new       : updated DF-decoder hidden state
            buf_erb0_new   : updated erb_conv0 input buffer
            buf_df0_new    : updated df_conv0 input buffer
            buf_dfp_new    : updated df_convp input buffer
            buf_spec_new   : updated spec causal context buffer
        """
        m = self.model
        enc = m.enc
        erb_dec = m.erb_dec
        df_dec = m.df_dec

        # ---- pre-process (skip pad_feat — not compatible with T=1) ----
        feat_spec_in = feat_spec.squeeze(1).permute(0, 3, 1, 2)  # [B, 2, 1, F']

        # ---- encoder — causal convs with explicit buffers ----
        e0, buf_erb0_new = _conv_with_buf(self._erb0_layers, buf_erb0, feat_erb)
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)

        c0, buf_df0_new = _conv_with_buf(self._df0_layers, buf_df0, feat_spec_in)
        c1 = enc.df_conv1(c0)

        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = enc.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2)
        emb = enc.combine(emb, cemb)
        emb, h_enc_new = enc.emb_gru(emb, h=h_enc)

        # ---- ERB decoder ----
        b, _, t, f8 = e3.shape
        emb_erb, h_erb_new = erb_dec.emb_gru(emb, h=h_erb)
        emb_erb = emb_erb.view(b, t, f8, -1).permute(0, 3, 1, 2)
        e3d = erb_dec.convt3(erb_dec.conv3p(e3) + emb_erb)
        e2d = erb_dec.convt2(erb_dec.conv2p(e2) + e3d)
        e1d = erb_dec.convt1(erb_dec.conv1p(e1) + e2d)
        mask = erb_dec.conv0_out(erb_dec.conv0p(e0) + e1d)
        spec_m = m.mask(spec, mask)

        # ---- DF decoder — df_convp has temporal kernel=5, needs buffer ----
        c, h_df_new = df_dec.df_gru(emb, h=h_df)
        if df_dec.df_skip is not None:
            c = c + df_dec.df_skip(emb)
        c0d, buf_dfp_new = _conv_with_buf(self._dfp_layers, buf_dfp, c0)
        c0d = c0d.permute(0, 2, 3, 1)
        c = df_dec.df_out(c)
        c = c.view(b, t, df_dec.df_bins, df_dec.df_out_ch) + c0d
        df_coefs = m.df_out_transform(c)  # [B, O, 1, F_df, 2]

        # ---- apply deep filtering with proper causal context ----
        # buf_spec holds the last _pad_before spec frames so that the df_op
        # 5-frame window uses real past data instead of zeros.
        # Window built: [spec_{t-2}, spec_{t-1}, spec_t, 0, 0]
        df_op = m.df_op
        spec_ctx = torch.cat([buf_spec, spec], dim=2)  # [B, 1, _pad_before+1, F, 2]
        # Slide the spec buffer: slice spec_ctx instead of a second torch.cat.
        buf_spec_new = spec_ctx[:, :, 1:, :, :]

        # Append zero future frames only when lookahead > 0; for causal streaming
        # (lookahead=0) spec_ctx already holds exactly frame_size steps.
        if df_op._pad_after > 0:
            spec_padded = F.pad(
                spec_ctx, [0, 0, 0, 0, 0, df_op._pad_after]
            )  # [B, 1, frame_size, F, 2]
        else:
            spec_padded = spec_ctx  # [B, 1, frame_size, F, 2] — no copy needed

        # spec_padded [B, 1, frame_size, F, 2] contains exactly one frame_size window.
        # unfold(2, frame_size, 1) is unsupported by the TorchScript ONNX exporter.
        # Equivalent (single-window) replacement: permute dims to [B, 1, F, frame_size, 2]
        # then unsqueeze the T=1 axis → [B, 1, 1, F, frame_size=O, 2].
        spec_unfolded = spec_padded.permute(0, 1, 3, 2, 4).unsqueeze(
            2
        )  # [B, 1, 1, F, O, 2]
        spec_f = spec_unfolded.narrow(3, 0, df_op.num_freqs)  # [B, 1, 1, F_df, O, 2]

        # Align df_coefs [B, O, T, F_df, 2] → [B, 1, T, F_df, O, 2] to match spec_f.
        # permute is a view (zero-copy); element-wise ops replace the 4 einsum calls.
        df_coefs_aligned = df_coefs.permute(0, 2, 3, 1, 4).unsqueeze(1)
        sr, si = spec_f[..., 0], spec_f[..., 1]  # [B, 1, T, F_df, O]
        cr, ci = df_coefs_aligned[..., 0], df_coefs_aligned[..., 1]
        result_r = (sr * cr - si * ci).sum(-1)  # [B, 1, T, F_df]
        result_i = (sr * ci + si * cr).sum(-1)

        # Avoid in-place index assignments (not supported by TorchScript ONNX exporter).
        # Concatenate the DF-filtered low bands with the ERB-masked high bands.
        df_out = torch.stack([result_r, result_i], dim=-1)  # [B, 1, 1, nb_df, 2]
        erb_out = spec_m[..., m.nb_df :, :]  # [B, 1, 1, F-nb_df, 2]
        spec_e = torch.cat([df_out, erb_out], dim=-2)  # [B, 1, 1, F, 2]

        # Attenuation limit: mix noisy + enhanced so suppression never exceeds atten_lim_db.
        if atten_lim_db is not None and abs(atten_lim_db) > 0:
            lim = 10 ** (-abs(atten_lim_db) / 20)
            spec_e = spec * lim + spec_e * (1 - lim)

        return (
            spec_e,
            h_enc_new,
            h_erb_new,
            h_df_new,
            buf_erb0_new,
            buf_df0_new,
            buf_dfp_new,
            buf_spec_new,
        )

    # ------------------------------------------------------------------
    # Helpers to create zero-filled initial states
    # ------------------------------------------------------------------

    def init_states(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return zero-initialised (h_enc, h_erb, h_df, buf_erb0, buf_df0, buf_dfp, buf_spec)."""
        enc_layers = self.model.enc.emb_gru.gru.num_layers
        erb_layers = self.model.erb_dec.emb_gru.gru.num_layers
        df_layers = self.model.df_dec.df_gru.gru.num_layers
        emb_dim = self.model.enc.emb_gru.hidden_size
        df_dim = self.model.df_dec.df_gru.hidden_size

        def _z(*shape):
            return torch.zeros(*shape, device=device)

        h_enc = _z(enc_layers, batch_size, emb_dim)
        h_erb = _z(erb_layers, batch_size, emb_dim)
        h_df = _z(df_layers, batch_size, df_dim)

        nb_erb = self.model.enc.erb_bins
        nb_spec = self.model.enc.nb_spec  # F dim of feat_spec input (= config.nb_df)
        conv_ch = self.model.enc.conv_ch

        # erb_conv0: in_ch=1, kt=3 → buf [B, 1, 2, nb_erb]
        buf_erb0 = _z(batch_size, 1, 2, nb_erb)

        # df_conv0: in_ch=2, kt=3 → buf [B, 2, 2, nb_spec]
        buf_df0 = _z(batch_size, 2, 2, nb_spec)

        # df_convp: in_ch=conv_ch, kt=5 → buf [B, conv_ch, 4, nb_spec]
        buf_dfp = _z(batch_size, conv_ch, 4, nb_spec)

        # df_op causal context: last _pad_before frames of spec → buf [B, 1, _pad_before, F_bins, 2]
        df_pad_before = self.model.df_op._pad_before
        F_bins = self.model.freq_bins
        buf_spec = _z(batch_size, 1, df_pad_before, F_bins, 2)

        return h_enc, h_erb, h_df, buf_erb0, buf_df0, buf_dfp, buf_spec
