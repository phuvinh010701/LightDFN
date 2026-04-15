import math

import torch
import torch.nn as nn
from torch import Tensor

from src.model.lightdeepfilternet import LightDeepFilterNet

PI: float = 3.1415926535897932384626433
EPS: float = 1e-12

LSNR_MIN: float = -10.0  # below → zero mask  (pure noise)
LSNR_ERB: float = 30.0  # above → ones mask  (clean, skip erb_dec call)
LSNR_DF: float = 20.0  # above → skip df_dec call


def _non_pad_sequential(seq: nn.Sequential) -> nn.Sequential:
    """Strip ConstantPad2d — causal temporal buffering is handled externally."""
    return nn.Sequential(*[m for m in seq if not isinstance(m, nn.ConstantPad2d)])


def _conv_with_buf(
    conv_layers: nn.Sequential,
    buf: Tensor,
    x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Causal convolution with an explicit ring buffer.

    Args:
        conv_layers: ConstantPad2d-stripped Sequential.
        buf:  ``[B, C_in, kt-1, F]`` — past context frames.
        x:    ``[B, C_in, 1,    F]`` — current frame.

    Returns:
        out:     ``[B, C_out, 1, F_out]``
        new_buf: ``[B, C_in, kt-1, F]``   zero-copy view (no allocation)
    """
    ctx = torch.cat([buf, x], dim=2)  # [B, C_in, kt, F]
    return conv_layers(ctx), ctx[:, :, 1:, :]


def build_rfft_matrix(fft_size: int, hop_size: int) -> Tensor:
    """RFFT basis with Vorbis window and wnorm pre-baked.

    Analysis:  ``spec [F*2] = frame_buf [fft_size] @ rfft_matrix``
    No additional window or wnorm multiply needed.
    """
    freq_bins = fft_size // 2 + 1
    rfft_mat = torch.view_as_real(torch.fft.rfft(torch.eye(fft_size)))  # [fft, F, 2]
    rfft_mat_2d = rfft_mat.reshape(fft_size, freq_bins * 2)  # [fft, F*2]
    t = (torch.arange(fft_size) + 0.5) / (fft_size // 2)
    window = torch.sin(0.5 * math.pi * torch.sin(0.5 * math.pi * t) ** 2)
    wnorm = 1.0 / (fft_size**2 / (2 * hop_size))
    return rfft_mat_2d * (window * wnorm).unsqueeze(1)  # [fft, F*2]


def build_irfft_matrix(fft_size: int, hop_size: int) -> Tensor:
    """IRFFT basis with ``fft_size * window`` scaling pre-baked.

    Synthesis:  ``x [fft_size] = spec_flat [F*2] @ irfft_matrix``
    No additional scaling multiply needed.
    """
    freq_bins = fft_size // 2 + 1
    rfft_mat = torch.view_as_real(torch.fft.rfft(torch.eye(fft_size)))
    rfft_mat_2d = rfft_mat.reshape(fft_size, freq_bins * 2)
    irfft_mat_2d = torch.linalg.pinv(rfft_mat_2d)  # [F*2, fft]
    t = (torch.arange(fft_size) + 0.5) / (fft_size // 2)
    window = torch.sin(0.5 * math.pi * torch.sin(0.5 * math.pi * t) ** 2)
    return irfft_mat_2d * (fft_size * window).unsqueeze(0)  # [F*2, fft]


def frame_analysis(
    frame: Tensor,  # [hop_size]
    analysis_mem: Tensor,  # [hop_size]
    rfft_matrix: Tensor,  # [fft_size, freq_bins*2]  window+wnorm baked in
) -> tuple[Tensor, Tensor]:
    """STFT analysis: PCM hop → complex spectrum.

    Rust equivalent: ``DFState::frame_analysis()`` in libDF/src/lib.rs.

    Returns:
        spec           [freq_bins, 2]
        new_analysis_mem  [hop_size]
    """
    buf = torch.cat([analysis_mem, frame])  # [fft_size]
    spec_flat = buf @ rfft_matrix  # [freq_bins*2]
    return spec_flat.view(-1, 2), frame


def frame_synthesis(
    spec: Tensor,  # [freq_bins, 2]
    synthesis_mem: Tensor,  # [hop_size]
    irfft_matrix: Tensor,  # [freq_bins*2, fft_size]  fft_size*window baked in
    hop_size: int,
) -> tuple[Tensor, Tensor]:
    """STFT synthesis: complex spectrum → PCM hop (overlap-add).

    Rust equivalent: ``DFState::frame_synthesis()`` in libDF/src/lib.rs.

    Returns:
        frame          [hop_size]
        new_synthesis_mem [hop_size]
    """
    x = spec.reshape(-1) @ irfft_matrix  # [fft_size]
    return x[:hop_size] + synthesis_mem, x[hop_size:]


def erb_power_db(spec: Tensor, erb_fb: Tensor) -> Tensor:
    """Complex spectrum → ERB log-power features.

    Rust equivalent: ``DFState::feat_erb()`` + ``compute_band_corr()``
    in libDF/src/lib.rs.

    Args:
        spec:   [freq_bins, 2]
        erb_fb: [freq_bins, nb_erb]

    Returns: [nb_erb]
    """
    return (spec.pow(2).sum(-1) @ erb_fb + 1e-10).log10() * 10.0


def band_mean_norm(xs: Tensor, state: Tensor, alpha: float) -> tuple[Tensor, Tensor]:
    """Exponential running-mean normalisation for ERB features.

    Rust equivalent: ``band_mean_norm_erb()`` in libDF/src/lib.rs.

    Returns: (normalised [nb_erb], new_state [nb_erb])
    """
    new_state = xs * (1.0 - alpha) + state * alpha
    return (xs - new_state) / 40.0, new_state


def band_unit_norm(xs: Tensor, state: Tensor, alpha: float) -> tuple[Tensor, Tensor]:
    """Exponential running unit-norm for complex spec features.

    Rust equivalent: ``band_unit_norm()`` in libDF/src/lib.rs.

    Returns: (normalised [nb_df, 2], new_state [nb_df])
    """
    xs_abs = xs.norm(dim=-1)  # [nb_df]
    new_state = (xs_abs * (1.0 - alpha) + state * alpha).clamp(min=1e-10)
    return xs / new_state.unsqueeze(-1).sqrt(), new_state


def apply_erb_mask(
    spec: Tensor,  # [freq_bins, 2]
    mask: Tensor,  # [1, 1, 1, nb_erb]  or broadcastable
    erb_inv_fb: Tensor,  # [nb_erb, freq_bins]
) -> Tensor:
    """Expand ERB gain mask to full spectrum and multiply.

    Rust equivalent: ``apply_interp_band_gain()`` in libDF/src/lib.rs.

    Returns: [freq_bins, 2]
    """
    gain = mask.reshape(1, -1) @ erb_inv_fb  # [1, freq_bins]
    return spec * gain.T  # [freq_bins, 2]


def apply_deep_filter(
    spec_ctx: Tensor,  # [pad_before+1, freq_bins, 2]  past frames + current
    coefs: Tensor,  # [df_order, nb_df, 2]          complex filter coefs
    nb_df: int,
    pad_after: int = 0,  # zero-pad future frames (streaming: no lookahead data)
) -> Tensor:
    """Complex multiply-accumulate deep filter over DF frequency bins.

    Rust equivalent: ``df()`` in libDF/src/lib.rs lines 724-767.
    High-freq bins are untouched (caller concatenates with ERB-masked spectrum).

    Returns: filtered low-freq bins [nb_df, 2]
    """
    if pad_after > 0:
        zeros = torch.zeros(
            pad_after,
            spec_ctx.shape[1],
            2,
            dtype=spec_ctx.dtype,
            device=spec_ctx.device,
        )
        spec_win = torch.cat([spec_ctx, zeros], dim=0)  # [df_order, freq_bins, 2]
    else:
        spec_win = spec_ctx  # [df_order, freq_bins, 2]

    sr = spec_win[:, :nb_df, 0]  # [df_order, nb_df]
    si = spec_win[:, :nb_df, 1]
    cr = coefs[..., 0]  # [df_order, nb_df]
    ci = coefs[..., 1]

    df_r = (sr * cr - si * ci).sum(0)  # [nb_df]
    df_i = (sr * ci + si * cr).sum(0)
    return torch.stack([df_r, df_i], dim=-1)  # [nb_df, 2]


class StreamingEncoder(nn.Module):
    """Encoder — single-frame ONNX model with explicit causal conv state.

    Mirrors the encoder path of DeepFilterNet3's ``enc.onnx``.
    All temporal state (conv ring buffers + GRU hidden state) is threaded
    explicitly so the ONNX graph is stateless from the runtime's view.

    Inputs
    ------
    feat_erb  : [1, 1, 1, nb_erb]          normalised ERB log-power
    feat_spec : [1, 2, 1, nb_df]           normalised complex spec (re/im channels)
    buf_erb0  : [1, 1, kt-1, nb_erb]       causal ring buffer for erb_conv0
    buf_df0   : [1, 2, kt-1, nb_df]        causal ring buffer for df_conv0
    h_enc     : [enc_layers, 1, emb_dim]   GRU hidden state

    Outputs
    -------
    e0..e3       encoder skip connections (for erb_dec skip path)
    emb          shared embedding  [1, 1, emb_out_dim]
    c0           DF pathway feature [1, conv_ch, 1, nb_df]
    lsnr         LSNR estimate [1]
    buf_erb0_new updated ring buffer
    buf_df0_new  updated ring buffer
    h_enc_new    updated GRU hidden state
    """

    def __init__(self, model: LightDeepFilterNet) -> None:
        super().__init__()
        enc = model.enc
        # Causal input conv — ConstantPad2d stripped; buffering is explicit
        self._erb0_layers = _non_pad_sequential(enc.erb_conv0)
        self._df0_layers = _non_pad_sequential(enc.df_conv0)
        # Inner conv layers use kernel=(1,3) — no temporal dependency at T=1
        self.erb_conv1 = enc.erb_conv1
        self.erb_conv2 = enc.erb_conv2
        self.erb_conv3 = enc.erb_conv3
        self.df_conv1 = enc.df_conv1
        self.df_fc_emb = enc.df_fc_emb
        self.combine = enc.combine
        self.emb_gru = enc.emb_gru
        self.lsnr_fc = enc.lsnr_fc
        self.lsnr_scale = enc.lsnr_scale
        self.lsnr_offset = enc.lsnr_offset

    def forward(
        self,
        feat_erb: Tensor,  # [1, 1, 1, nb_erb]
        feat_spec: Tensor,  # [1, 2, 1, nb_df]
        buf_erb0: Tensor,  # [1, 1, kt-1, nb_erb]
        buf_df0: Tensor,  # [1, 2, kt-1, nb_df]
        h_enc: Tensor,  # [enc_layers, 1, emb_dim]
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        e0, buf_erb0_new = _conv_with_buf(self._erb0_layers, buf_erb0, feat_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)

        c0, buf_df0_new = _conv_with_buf(self._df0_layers, buf_df0, feat_spec)
        c1 = self.df_conv1(c0)

        cemb = self.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))  # [1,1,emb_in]
        emb_raw = e3.permute(0, 2, 3, 1).flatten(2)  # [1,1,emb_in]
        emb_raw = self.combine(emb_raw, cemb)
        emb, h_enc_new = self.emb_gru.step(emb_raw, h_enc)  # [1,1,emb_out]

        lsnr = (self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset).reshape(1)

        return e0, e1, e2, e3, emb, c0, lsnr, buf_erb0_new, buf_df0_new, h_enc_new


class StreamingErbDecoder(nn.Module):
    """ERB decoder — produces gain mask for ERB bands.

    Mirrors DeepFilterNet3's ``erb_dec.onnx``.

    Inputs
    ------
    emb   : [1, 1, emb_out_dim]
    e3    : [1, conv_ch, 1, nb_erb/4]
    e2    : [1, conv_ch, 1, nb_erb/2]
    e1    : [1, conv_ch, 1, nb_erb]
    e0    : [1, conv_ch, 1, nb_erb]
    h_erb : [erb_layers, 1, emb_dim]

    Outputs
    -------
    mask      [1, 1, 1, nb_erb]  ∈ (0, 1)
    h_erb_new
    """

    def __init__(self, model: LightDeepFilterNet) -> None:
        super().__init__()
        d = model.erb_dec
        self.emb_gru = d.emb_gru
        self.conv3p = d.conv3p
        self.convt3 = d.convt3
        self.conv2p = d.conv2p
        self.convt2 = d.convt2
        self.conv1p = d.conv1p
        self.convt1 = d.convt1
        self.conv0p = d.conv0p
        self.conv0_out = d.conv0_out

    def forward(
        self,
        emb: Tensor,  # [1, 1, emb_out_dim]
        e3: Tensor,  # [1, conv_ch, 1, nb_erb/4]
        e2: Tensor,  # [1, conv_ch, 1, nb_erb/2]
        e1: Tensor,  # [1, conv_ch, 1, nb_erb]
        e0: Tensor,  # [1, conv_ch, 1, nb_erb]
        h_erb: Tensor,  # [erb_layers, 1, emb_dim]
    ) -> tuple[Tensor, Tensor]:
        b, _, t, f8 = e3.shape
        emb_d, h_erb_new = self.emb_gru.step(emb, h_erb)
        emb_4d = emb_d.view(b, t, f8, -1).permute(0, 3, 1, 2)
        e3d = self.convt3(self.conv3p(e3) + emb_4d)
        e2d = self.convt2(self.conv2p(e2) + e3d)
        e1d = self.convt1(self.conv1p(e1) + e2d)
        mask = self.conv0_out(self.conv0p(e0) + e1d)  # [1,1,1,nb_erb]
        return mask, h_erb_new


class StreamingDfDecoder(nn.Module):
    """DF decoder — per-frequency complex filter coefficients.

    Mirrors DeepFilterNet3's ``df_dec.onnx``.

    Inputs
    ------
    emb     : [1, 1, emb_out_dim]
    c0      : [1, conv_ch, 1, nb_df]          DF pathway feature from encoder
    buf_dfp : [1, conv_ch, kt-1, nb_df]       causal ring buffer for df_convp
    h_df    : [df_layers, 1, df_dim]

    Outputs
    -------
    coefs       [1, df_order, 1, nb_df, 2]
    buf_dfp_new updated ring buffer
    h_df_new    updated GRU hidden state
    """

    def __init__(self, model: LightDeepFilterNet) -> None:
        super().__init__()
        d = model.df_dec
        self._dfp_layers = _non_pad_sequential(d.df_convp)
        self.df_gru = d.df_gru
        self.df_skip = d.df_skip
        self.df_out = d.df_out
        self.df_out_transform = model.df_out_transform
        self.df_bins: int = d.df_bins
        self.df_out_ch: int = d.df_out_ch

    def forward(
        self,
        emb: Tensor,  # [1, 1, emb_out_dim]
        c0: Tensor,  # [1, conv_ch, 1, nb_df]
        buf_dfp: Tensor,  # [1, conv_ch, kt-1, nb_df]
        h_df: Tensor,  # [df_layers, 1, df_dim]
    ) -> tuple[Tensor, Tensor, Tensor]:
        c_gru, h_df_new = self.df_gru.step(emb, h_df)
        if self.df_skip is not None:
            c_gru = c_gru + self.df_skip(emb)

        c0d, buf_dfp_new = _conv_with_buf(self._dfp_layers, buf_dfp, c0)
        c0d_4d = c0d.permute(0, 2, 3, 1)  # [1,1,nb_df,df_out_ch]

        c_out = self.df_out(c_gru)  # [1,1,nb_df*df_out_ch]
        c_out = c_out.view(1, 1, self.df_bins, self.df_out_ch) + c0d_4d
        coefs = self.df_out_transform(c_out)  # [1,df_order,1,nb_df,2]
        return coefs, buf_dfp_new, h_df_new


class LightDFNStreaming:
    """Frame-by-frame streaming orchestrator — Python reference for Rust/WASM.

    Manages all DSP state (Rust: ``DFState`` struct fields) and calls the
    three ONNX modules once per frame with LSNR/silence gating identical to
    DeepFilterNet3's ``DfTract::process()``.

    NOT exported to ONNX. Use for correctness testing in Python only.
    """

    def __init__(
        self,
        enc: StreamingEncoder,
        erb_dec: StreamingErbDecoder,
        df_dec: StreamingDfDecoder,
        model: LightDeepFilterNet,
        fft_size: int = 960,
        hop_size: int = 480,
        norm_tau: float = 1.0,
        sr: int = 48_000,
        silence_thresh: float = 1e-7,
        min_db_thresh: float = LSNR_MIN,
        max_db_erb_thresh: float = LSNR_ERB,
        max_db_df_thresh: float = LSNR_DF,
    ) -> None:
        self.enc = enc
        self.erb_dec = erb_dec
        self.df_dec = df_dec

        self.hop_size = hop_size
        self.freq_bins = fft_size // 2 + 1
        self.nb_erb = model.erb_bins
        self.nb_df = model.nb_df
        self.silence_thresh = silence_thresh
        self.min_db_thresh = min_db_thresh
        self.max_db_erb_thresh = max_db_erb_thresh
        self.max_db_df_thresh = max_db_df_thresh
        self.post_filter = model.post_filter
        self.post_filter_beta = model.post_filter_beta

        self._df_order = model.df_order
        self._pad_before = model.df_op._pad_before  # df_order-1-lookahead
        self._pad_after = model.df_op._pad_after  # lookahead (0 for true causal)

        fps = sr / hop_size
        self.alpha: float = math.exp(-1.0 / (fps * norm_tau))

        # Pre-computed STFT matrices (Rust: computed once in DFState::new)
        self._rfft = build_rfft_matrix(fft_size, hop_size)
        self._irfft = build_irfft_matrix(fft_size, hop_size)

        # ERB filterbanks
        self._erb_fb = model.erb_fb  # [freq_bins, nb_erb]
        self._erb_inv_fb = model.mask.erb_inv_fb  # [nb_erb, freq_bins]

        # ---- DSP state (Rust: DFState struct fields) ----
        self.analysis_mem = torch.zeros(hop_size)
        self.synthesis_mem = torch.zeros(hop_size)
        self.erb_norm_state = torch.linspace(-60.0, -90.0, self.nb_erb)
        self.unit_norm_state = torch.linspace(0.001, 0.0001, self.nb_df)
        # Rolling spec buffer: stores past pad_before frames + current slot
        # Rust: VecDeque<Vec<Complex32>> sized pad_before+1
        self.past_spec = torch.zeros(self._pad_before, self.freq_bins, 2)
        self._last_lsnr = torch.zeros(1)

        # ---- ONNX model state (threaded through each call) ----
        enc_layers = len(model.enc.emb_gru.ligru.rnn)
        erb_layers = len(model.erb_dec.emb_gru.ligru.rnn)
        df_layers = len(model.df_dec.df_gru.ligru.rnn)
        emb_dim = model.enc.emb_gru.hidden_size
        df_dim = model.df_dec.df_gru.hidden_size

        # Buffer sizes derived from first conv layer in each stripped sequential
        kt_erb0 = enc._erb0_layers[0].kernel_size[0]  # conv_kernel_inp[0] = 3
        kt_df0 = enc._df0_layers[0].kernel_size[0]  # conv_kernel_inp[0] = 3
        kt_dfp = df_dec._dfp_layers[0].kernel_size[0]  # df_pathway_kernel_size_t = 5

        self.h_enc = torch.zeros(enc_layers, 1, emb_dim)
        self.h_erb = torch.zeros(erb_layers, 1, emb_dim)
        self.h_df = torch.zeros(df_layers, 1, df_dim)
        self.buf_erb0 = torch.zeros(1, 1, kt_erb0 - 1, self.nb_erb)
        self.buf_df0 = torch.zeros(1, 2, kt_df0 - 1, self.nb_df)
        self.buf_dfp = torch.zeros(1, model.enc.conv_ch, kt_dfp - 1, self.nb_df)

    @torch.no_grad()
    def process_frame(
        self,
        input_frame: Tensor,  # [hop_size]  raw PCM float32
        atten_lim_db: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Process one audio hop — mirrors DfTract::process() in tract.rs.

        Returns:
            enhanced_frame [hop_size]
            lsnr           [1]
        """
        # ---- Silence gate (Rust: early return, no model call) ----
        if (input_frame.pow(2)).mean().item() < self.silence_thresh:
            return torch.zeros(self.hop_size), self._last_lsnr

        # ---- STFT analysis ----
        spec, self.analysis_mem = frame_analysis(
            input_frame, self.analysis_mem, self._rfft
        )  # spec: [freq_bins, 2]

        # ---- Feature extraction ----
        erb_db = erb_power_db(spec, self._erb_fb)  # [nb_erb]
        feat_erb_1d, self.erb_norm_state = band_mean_norm(
            erb_db, self.erb_norm_state, self.alpha
        )
        feat_spec_1d, self.unit_norm_state = band_unit_norm(
            spec[: self.nb_df], self.unit_norm_state, self.alpha
        )

        feat_erb_in = feat_erb_1d.reshape(1, 1, 1, self.nb_erb)
        feat_spec_in = feat_spec_1d.T.reshape(1, 2, 1, self.nb_df)

        # ---- ONNX call 1: encoder ----
        (
            e0,
            e1,
            e2,
            e3,
            emb,
            c0,
            lsnr,
            self.buf_erb0,
            self.buf_df0,
            self.h_enc,
        ) = self.enc(feat_erb_in, feat_spec_in, self.buf_erb0, self.buf_df0, self.h_enc)
        self._last_lsnr = lsnr
        lsnr_val = lsnr.item()

        # ---- ONNX call 2: ERB decoder — LSNR gated ----
        if lsnr_val < self.min_db_thresh:
            mask = torch.zeros(1, 1, 1, self.nb_erb)  # pure noise → zero mask
        elif lsnr_val > self.max_db_erb_thresh:
            mask = torch.ones(1, 1, 1, self.nb_erb)  # clean → pass-through
        else:
            mask, self.h_erb = self.erb_dec(emb, e3, e2, e1, e0, self.h_erb)

        # ---- Apply ERB mask ----
        spec_m = apply_erb_mask(spec, mask, self._erb_inv_fb)  # [freq_bins, 2]

        # ---- Update rolling spec buffer (Rust: VecDeque::push_back/pop_front) ----
        spec_ctx = torch.cat([self.past_spec, spec.unsqueeze(0)], dim=0)
        self.past_spec = spec_ctx[1:].clone()  # drop oldest, keep last pad_before

        # ---- ONNX call 3: DF decoder — LSNR gated ----
        if self.min_db_thresh <= lsnr_val <= self.max_db_df_thresh:
            coefs, self.buf_dfp, self.h_df = self.df_dec(
                emb, c0, self.buf_dfp, self.h_df
            )
            # coefs [1, df_order, 1, nb_df, 2] → [df_order, nb_df, 2]
            coefs_3d = coefs.reshape(self._df_order, self.nb_df, 2)
            df_low = apply_deep_filter(spec_ctx, coefs_3d, self.nb_df, self._pad_after)
            spec_e = torch.cat([df_low, spec_m[self.nb_df :]], dim=0)  # [freq_bins, 2]
        else:
            spec_e = spec_m

        # ---- Optional Valin post-filter ----
        if self.post_filter:
            spec_c = torch.view_as_complex(spec_e.contiguous())
            orig_c = torch.view_as_complex(spec.contiguous())
            pf_mask = (spec_c.abs() / orig_c.abs().clamp(min=EPS)).clamp(EPS, 1.0)
            pf_sin = (pf_mask * torch.sin(PI * pf_mask / 2)).clamp_min(EPS)
            pf = (1.0 + self.post_filter_beta) / (
                1.0 + self.post_filter_beta * pf_mask.div(pf_sin).pow(2)
            )
            spec_e = spec_e * pf.unsqueeze(-1)

        # ---- Attenuation limit mix-back ----
        if atten_lim_db > 1e-6:
            lim = 10.0 ** (-atten_lim_db / 20.0)
            spec_e = spec * lim + spec_e * (1.0 - lim)

        # ---- STFT synthesis ----
        enhanced_frame, self.synthesis_mem = frame_synthesis(
            spec_e, self.synthesis_mem, self._irfft, self.hop_size
        )
        return enhanced_frame, lsnr

    def reset(self) -> None:
        """Zero all DSP and ONNX state — equivalent to constructing a fresh DFState."""
        self.analysis_mem.zero_()
        self.synthesis_mem.zero_()
        self.erb_norm_state = torch.linspace(-60.0, -90.0, self.nb_erb)
        self.unit_norm_state = torch.linspace(0.001, 0.0001, self.nb_df)
        self.past_spec.zero_()
        self.h_enc.zero_()
        self.h_erb.zero_()
        self.h_df.zero_()
        self.buf_erb0.zero_()
        self.buf_df0.zero_()
        self.buf_dfp.zero_()
        self._last_lsnr.zero_()
