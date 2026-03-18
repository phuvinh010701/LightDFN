from tempfile import NamedTemporaryFile
from typing import Any, Optional
from torch import Tensor
import torchaudio
import numpy as np


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
            return x.numpy().astype(dtype)
        case _:
            raise NotImplementedError(f"Codec '{codec}' not supported.")
