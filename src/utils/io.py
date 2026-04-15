from tempfile import NamedTemporaryFile
from typing import Any, Optional

import numpy as np
import torch
import torchaudio
from torch import Tensor


def get_device():
    """Get the device to use for computations."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def get_resample_params(method: str = "sinc_fast") -> dict[str, Any]:
    match method:
        case "sinc_fast":
            return {"resampling_method": "sinc_interp_hann", "lowpass_filter_width": 16}
        case "sinc_best":
            return {
                "resampling_method": "sunc_interp_hann",
                "lowpass_filter_width": 64,
            }
        case "kaiser_fast":
            return {
                "resampling_method": "sinc_interp_kaiser",
                "lowpass_filter_width": 16,
                "rolloff": 0.85,
                "beta": 8.555504641634386,
            }
        case "kaiser_best":
            return {
                "resampling_method": "sinc_interp_kaiser",
                "lowpass_filter_width": 16,
                "rolloff": 0.9475937167399596,
                "beta": 14.769656459379492,
            }
        case _:
            raise ValueError(f"Invalid resampling method: {method}")


def resample(audio: Tensor, sr: int, new_sr: int, method: str = "sinc_fast") -> Tensor:
    params = get_resample_params(method)
    return torchaudio.functional.resample(audio, sr, new_sr, **params)


def encode(
    x: Tensor, sr: int, dtype: np.dtype, codec: str, compression: Optional[int] = None
) -> np.ndarray:
    match codec:
        case "vorbis":
            with NamedTemporaryFile(suffix=".ogg") as tf:
                torchaudio.save(
                    tf.name, x, sr, format="vorbis", compression=compression
                )
                return np.array(list(tf.read()), dtype=np.uint8)
        case "flac":
            with NamedTemporaryFile(suffix=".flac") as tf:
                torchaudio.save(
                    tf.name,
                    x,
                    sr,
                    format="flac",
                    compression=compression,
                    bits_per_sample=16,
                )
                return np.array(list(tf.read()), dtype=np.uint8)
        case "pcm":
            if dtype == np.int16:
                x = x * 32767.0
            return x.numpy().astype(dtype)
        case _:
            raise NotImplementedError(f"Codec '{codec}' not supported.")


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
