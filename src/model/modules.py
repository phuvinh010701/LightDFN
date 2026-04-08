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

    def forward(
        self, spec: Tensor, mask: Tensor, atten_lim: Optional[Tensor] = None
    ) -> Tensor:
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


class SqueezedGRU_S(nn.Module):
    """Squeezed GRU with grouped linear projections and optional skip connection.

    Mirrors DeepFilterNet3's ``SqueezedGRU_S``.  The skip connection is applied
    *after* the output projection (linear_out → skip), matching the reference:

        x = linear_in(input)
        x, h = gru(x, h)
        x = linear_out(x)
        if skip: x = x + skip(input)

    Args:
        input_size: Dimensionality of the input features.
        hidden_size: GRU hidden (and internal) size.
        output_size: If given, project GRU output to this size; otherwise the
            GRU output is passed through unchanged (``nn.Identity``).
        num_layers: Number of stacked GRU layers.
        linear_groups: Number of groups for the grouped-linear projections.
        batch_first: Whether the time dimension is dim-1 (True) or dim-0 (False).
        gru_skip_op: Factory for the skip-connection module (called with no args).
            Pass ``None`` to disable the skip.
        linear_act_layer: Factory for the activation after each linear projection.
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
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups),
            linear_act_layer(),
        )
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None

        if output_size is not None:
            self.linear_out: nn.Module = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups),
                linear_act_layer(),
            )
        else:
            self.linear_out = nn.Identity()

    def forward(
        self, input: Tensor, h: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)
        x, h_out = self.gru(x, h)
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, h_out
