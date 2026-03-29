import math
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter
from typing_extensions import Final


def as_complex(x: Tensor):
    """Convert real tensor with last dim 2 to complex tensor."""
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(
            f"Last dimension need to be of length 2 (re + im), but got {x.shape}"
        )
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


def as_real(x: Tensor):
    """Convert complex tensor to real tensor."""
    if torch.is_complex(x):
        return torch.view_as_real(x)
    return x


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)


class Conv2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int | Iterable[int],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ):
        """Causal Conv2d by delaying the signal for any lookahead.

        Expected input format: [B, C, T, F]
        """
        lookahead = 0  # This needs to be handled on the input feature side
        # Padding on time axis
        kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        if fpad:
            fpad_ = kernel_size[1] // 2 + dilation - 1
        else:
            fpad_ = 0
        pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel_size) == 1:
            separable = False
        layers.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=(0, fpad_),
                stride=(1, fstride),  # Stride over time is always 1
                dilation=(1, dilation),  # Same for dilation
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int | Tuple[int, int],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ):
        """Causal ConvTranspose2d.

        Expected input format: [B, C, T, F]
        """
        lookahead = 0
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        if fpad:
            fpad_ = kernel_size[1] // 2
        else:
            fpad_ = 0
        pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=(kernel_size[0] - 1, fpad_ + dilation - 1),
                output_padding=(0, fpad_),
                stride=(1, fstride),
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class GroupedLinearEinsum(nn.Module):
    """Grouped linear layer using einsum for efficient computation."""

    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        assert input_size % groups == 0, (
            f"Input size {input_size} not divisible by {groups}"
        )
        assert hidden_size % groups == 0, (
            f"Hidden size {hidden_size} not divisible by {groups}"
        )
        self.ws = input_size // groups
        self.register_parameter(
            "weight",
            Parameter(
                torch.zeros(groups, input_size // groups, hidden_size // groups),
                requires_grad=True,
            ),
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., I]
        b, t, _ = x.shape
        # new_shape = list(x.shape)[:-1] + [self.groups, self.ws]
        new_shape = (b, t, self.groups, self.ws)
        x = x.view(new_shape)
        # The better way, but not supported by torchscript
        # x = x.unflatten(-1, (self.groups, self.ws))  # [..., G, I/G]
        x = torch.einsum("btgi,gih->btgh", x, self.weight)  # [..., G, H/G]
        x = x.flatten(2, 3)  # [B, T, H]

        return x

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(input_size: {self.input_size}, hidden_size: {self.hidden_size}, groups: {self.groups})"


class Mask(nn.Module):
    """Applies ERB mask to complex spectrogram."""

    def __init__(self, erb_inv_fb: Tensor, eps: float = 1e-12):
        super().__init__()
        self.erb_inv_fb: Tensor
        self.register_buffer("erb_inv_fb", erb_inv_fb)
        self.eps = eps

    def forward(self, spec: Tensor, mask: Tensor, atten_lim: Optional[Tensor] = None) -> Tensor:
        # spec (real) [B, 1, T, F, 2], F: freq_bins
        # mask (real): [B, 1, T, Fe], Fe: erb_bins
        # atten_lim: [B] — per-sample attenuation limit in dB (optional)
        if atten_lim is not None:
            # Convert dB to linear amplitude and clamp mask from below
            lim = 10 ** (-atten_lim / 20)
            mask = mask.clamp(min=lim.view(-1, 1, 1, 1))
        mask = mask.matmul(self.erb_inv_fb)  # [B, 1, T, F]
        if not spec.is_complex():
            mask = mask.unsqueeze(4)
        return spec * mask


class SqueezedLiGRU_S(nn.Module):
    """Squeezed GRU with Li-GRU instead of standard GRU.

    This module wraps Li-GRU to provide an interface compatible with the original
    DeepFilterNet3's SqueezedLiGRU_S that used standard GRU.
    """

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., nn.Module]] = None,
        linear_act_layer: Callable[..., nn.Module] = nn.Identity,
        batch_size: int = 1,  # Added for Li-GRU
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # Linear input projection
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups),
            linear_act_layer(),
        )

        # Use Li-GRU instead of standard GRU
        self.ligru = LiGRU(
            hidden_size=hidden_size,
            input_shape=(
                batch_size,
                1,
                hidden_size,
            ),  # Dummy shape, will be adjusted dynamically
            num_layers=num_layers,
            nonlinearity="relu",
            normalization="batchnorm",
            bias=True,
            dropout=0.0,
            bidirectional=False,
        )

        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None

        # Linear output projection
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups),
                linear_act_layer(),
            )
        else:
            self.linear_out = nn.Identity()

    def forward(
        self, input: Tensor, h: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)

        # Li-GRU returns (output, hidden_states)
        x, h_out = self.ligru(x, hx=h)

        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, h_out


@torch.jit.script
def _ligru_cell_relu(
    w: Tensor,
    u_weight: Tensor,
    drop_mask: Tensor,
    ht: Tensor,
) -> Tensor:
    """JIT-compiled LiGRU cell loop for ReLU activation.

    Compiles the recurrent time-step loop to C++, eliminating Python dispatch
    overhead (~300 iterations for a 3 s clip at 48 kHz / hop 480).

    Uses list + stack instead of pre-allocated tensor with index_put_ so the
    ONNX TorchScript exporter can convert the loop body to ONNX SequenceInsert /
    ConcatFromSequence ops (opset >= 11) without hitting unsupported index_put_.

    Args:
        w:         Pre-computed input projection ``W * x``, shape ``[B, T, 2H]``.
        u_weight:  Recurrent weight matrix of shape ``[2H, H]``.
        drop_mask: Dropout mask broadcast-compatible with ``[B, H]``.
        ht:        Initial hidden state, shape ``[B, H]`` (or ``[1, H]`` to broadcast).

    Returns:
        Full hidden-state sequence of shape ``[B, T, H]``.
    """
    u_t = u_weight.t()  # pre-transpose once outside the loop
    out: list[Tensor] = []
    for k in range(w.shape[1]):
        gates = w[:, k] + ht @ u_t
        at, zt = gates.chunk(2, 1)
        zt = torch.sigmoid(zt)
        hcand = torch.relu(at) * drop_mask
        ht = zt * ht + (1 - zt) * hcand
        out.append(ht)
    return torch.stack(out, dim=1)


class LiGRU_Layer(nn.Module):
    """This function implements Light-Gated Recurrent Units (ligru) layer.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    batch_size : int
        Batch size of the input tensors.
    hidden_size : int
        Number of output neurons.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    nonlinearity : str
        Type of nonlinearity (tanh, relu).
    normalization : str
        Type of normalization (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in no normalization.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    bidirectional : bool
        if True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        nonlinearity="relu",
        normalization="batchnorm",
        bidirectional=False,
    ):
        super(LiGRU_Layer, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.nonlinearity = nonlinearity

        self.w = nn.Linear(self.input_size, 2 * self.hidden_size, bias=False)

        self.u = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initializing batch norm
        self.normalize = False

        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(2 * self.hidden_size, momentum=0.05)
            self.normalize = True

        elif normalization == "layernorm":
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size)
            self.normalize = True
        else:
            # Normalization is disabled here. self.norm is only formally
            # initialized to avoid jit issues.
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size)
            self.normalize = False

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop()

        # Setting the activation function
        if nonlinearity == "tanh":
            self.act = torch.nn.Tanh()
        elif nonlinearity == "sin":
            self.act = torch.sin
        elif nonlinearity == "leaky_relu":
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = torch.nn.ReLU()

    def forward(self, x, hx: Optional[torch.Tensor] = None):
        """Returns the output of the liGRU layer.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Apply batch normalization
        if self.normalize:
            B, T, H2 = w.shape
            w = self.norm(w.view(B * T, H2)).view(B, T, H2)

        # Processing time steps
        if hx is not None:
            h = self._ligru_cell(w, hx)
        else:
            h = self._ligru_cell(w, self.h_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    def _ligru_cell(self, w: Tensor, ht: Tensor) -> Tensor:
        """Returns the hidden states for each time step.

        Dispatches to the JIT-compiled loop for ReLU (the common case) to
        eliminate Python per-step overhead.  Falls back to a Python loop for
        other activations.

        Arguments
        ---------
        w : Tensor
            Pre-computed input projection ``W * x``, shape ``[B, T, 2H]``.
        ht : Tensor
            Initial hidden state, shape ``[B, H]``.
        """
        drop_mask = self._sample_drop_mask(w)

        if self.nonlinearity == "relu":
            return _ligru_cell_relu(w, self.u.weight, drop_mask, ht)

        # Fallback for non-relu activations
        hiddens = []
        for k in range(w.shape[1]):
            gates = w[:, k] + self.u(ht)
            at, zt = gates.chunk(2, 1)
            zt = torch.sigmoid(zt)
            hcand = self.act(at) * drop_mask
            ht = zt * ht + (1 - zt) * hcand
            hiddens.append(ht)
        return torch.stack(hiddens, dim=1)

    def _init_drop(self):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)
        self.N_drop_masks = 16000
        self.drop_mask_cnt = 0

        self.register_buffer(
            "drop_masks",
            self.drop(torch.ones(self.N_drop_masks, self.hidden_size)).data,
        )
        self.register_buffer("drop_mask_te", torch.tensor([1.0]).float())

    def _sample_drop_mask(self, w):
        """Selects one of the pre-defined dropout masks"""
        if self.training:
            # Sample new masks when needed
            if self.drop_mask_cnt + self.batch_size > self.N_drop_masks:
                self.drop_mask_cnt = 0
                self.drop_masks = self.drop(
                    torch.ones(self.N_drop_masks, self.hidden_size, device=w.device)
                ).data

            # Sampling the mask
            drop_mask = self.drop_masks[
                self.drop_mask_cnt : self.drop_mask_cnt + self.batch_size
            ]
            self.drop_mask_cnt = self.drop_mask_cnt + self.batch_size

        else:
            if self.drop_mask_te.device != w.device:
                self.drop_mask_te = self.drop_mask_te.to(w.device)
            drop_mask = self.drop_mask_te

        return drop_mask

    def _change_batch_size(self, x):
        """This function changes the batch size when it is different from
        the one detected in the initialization method. This might happen in
        the case of multi-gpu or when we have different batch sizes in train
        and test. We also update the h_int and drop masks.
        """
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

            if self.training:
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks,
                        self.hidden_size,
                        device=x.device,
                    )
                ).data


class LiGRU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        input_shape: tuple[int, ...],
        nonlinearity: str = "relu",
        normalization: str = "batchnorm",
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        re_init: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.normalization = normalization
        self.bias = bias
        self.dropout = dropout
        self.re_init = re_init
        self.bidirectional = bidirectional
        self.reshape = False

        if len(input_shape) > 3:
            self.reshape = True
        self.fea_dim = math.prod(input_shape[2:])
        self.batch_size = input_shape[0]
        self.rnn = self._init_layers()

        if self.re_init:
            rnn_init(self.rnn)

    def _init_layers(self):
        """Initializes the layers of the liGRU."""
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for _ in range(self.num_layers):
            rnn_lay = LiGRU_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                nonlinearity=self.nonlinearity,
                normalization=self.normalization,
                bidirectional=self.bidirectional,
            )
            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size
        return rnn

    def forward(self, x, hx: Optional[torch.Tensor] = None):
        """Returns the output of the liGRU.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor.
        hx : torch.Tensor
            Starting hidden state.
        """
        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # run ligru
        output, hh = self._forward_ligru(x, hx=hx)

        return output, hh

    def _forward_ligru(self, x, hx: Optional[torch.Tensor]):
        """Returns the output of the vanilla liGRU.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        hx : torch.Tensor
        """
        if hx is not None:
            if self.bidirectional:
                hx = hx.reshape(self.num_layers, self.batch_size * 2, self.hidden_size)

        # Pre-allocate hidden state storage: avoids a Python list + torch.stack.
        # x[:, -1, :] = last time step, shape [B, H].
        # We collect into a pre-sized tensor directly.
        first_hx = hx[0] if hx is not None else None
        x = self.rnn[0](x, hx=first_hx)
        h_last = x.new_empty(self.num_layers, x.shape[0], self.hidden_size)
        h_last[0] = x[:, -1, :]

        for i in range(1, self.num_layers):
            x = self.rnn[i](x, hx=hx[i] if hx is not None else None)
            h_last[i] = x[:, -1, :]

        if self.bidirectional:
            h_last = h_last.reshape(
                h_last.shape[0] * 2, h_last.shape[1], self.hidden_size
            )
        else:
            h_last = h_last.transpose(0, 1)

        return x, h_last


def rnn_init(module):
    """This function is used to initialize the RNN weight.
    Recurrent connection: orthogonal initialization.

    Arguments
    ---------
    module: torch.nn.Module
        Recurrent neural network module.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(hidden_size=5, input_shape=inp_tensor.shape)
    >>> out_tensor = net(inp_tensor)
    >>> rnn_init(net)
    """
    for name, param in module.named_parameters():
        if "weight_hh" in name or ".u.weight" in name:
            nn.init.orthogonal_(param)
