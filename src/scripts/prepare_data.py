#!/usr/bin/env python3

import argparse
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional

import h5py as h5
import numpy as np
import torch
import torchaudio
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from constants import AUDIO_DATASET_TYPES
from src.io import encode, resample


class PreProcessingDataset(Dataset):
    def __init__(
        self,
        sr: int,
        file_names: list[str],
        dtype: str = "float32",
        codec: str = "pcm",
        mono: bool = False,
        compression: Optional[int] = None,
    ):
        self.file_names = file_names
        self.sr = sr

        if dtype == "float32":
            self.dtype = np.float32
        elif dtype == "int16":
            self.dtype = np.int16
        else:
            raise ValueError("Unknown dtype. Expected 'float32' or 'int16'.")

        self.codec = codec.lower()
        if self.codec == "vorbis":
            self.dtype = np.float32

        self.mono = mono
        self.compression = compression

    def read(self, file_path: str) -> Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta = torchaudio.info(file_path)
            if meta.sample_rate != self.sr:
                x, sr = torchaudio.load(file_path, normalize=True)
                x = resample(x, sr, self.sr, method="kaiser_best")
            else:
                x, sr = torchaudio.load(file_path, normalize=False)
            if self.mono and x.shape[0] > 1:
                x = x.mean(0, keepdim=True)
            if x.dim() == 1:
                x = x.reshape(1, -1)
        return x

    def __getitem__(self, index: int) -> dict[str, object]:
        fn = self.file_names[index]
        logger.debug(f"Reading audio file {fn}")
        x = self.read(fn)
        assert x.dim() == 2 and x.shape[0] <= 16, (
            f"Got sample {fn} with unexpected shape {x.shape}"
        )
        n_samples = x.shape[1]
        encoded = encode(x, self.sr, self.codec, self.compression)
        return {"file_name": fn, "data": encoded, "n_samples": int(n_samples)}

    def __len__(self) -> int:
        return len(self.file_names)


def write_to_h5(
    output_file_name: str,
    dataset_type: Literal[AUDIO_DATASET_TYPES],
    audio_file_names: list[str],
    sr: int,
    max_freq: int = -1,
    dtype: str = "float32",
    codec: str = "pcm",
    mono: bool = False,
    compression: Optional[str] = None,
    num_workers: int = 4,
) -> None:
    if max_freq <= 0:
        max_freq = sr // 2
    compression_factor = 8 if codec is not None else None

    with (
        h5.File(output_file_name, "a", libver="latest", swmr=True) as f,
        torch.no_grad(),
    ):
        f.attrs["db_id"] = int(time.time())
        f.attrs["db_name"] = os.path.basename(output_file_name)
        f.attrs["max_freq"] = max_freq
        f.attrs["dtype"] = dtype
        f.attrs["sr"] = sr
        f.attrs["codec"] = codec

        group = f.create_group(dataset_type)

        dataset = PreProcessingDataset(
            sr=sr,
            file_names=audio_file_names,
            dtype=dtype,
            codec=codec,
            mono=mono,
            compression=compression_factor,
        )
        loader = DataLoader(
            dataset, num_workers=num_workers, batch_size=1, shuffle=False
        )
        n_total = len(dataset)

        base_dir = str(Path(audio_file_names[0]).parent) if audio_file_names else "."

        for i, sample in enumerate(loader):
            fn = os.path.relpath(sample["file_name"][0], base_dir)
            audio = sample["data"][0].numpy()

            if codec in ("flac", "vorbis"):
                audio = audio.squeeze()

            n_samples = int(sample["n_samples"][0])
            if n_samples < sr / 100:
                logger.warning(f"Short audio {fn}: {audio.shape}.")

            progress = i / max(n_total, 1) * 100
            logger.info(f"{progress:2.0f}% | Writing file {fn}")

            if n_samples == 0:
                continue

            ds_key = fn.replace("/", "_")
            if ds_key in group:
                logger.info(f"Found dataset {ds_key}. Replacing.")
                del group[ds_key]

            ds = group.create_dataset(ds_key, data=audio, compression=compression)
            ds.attrs["n_samples"] = n_samples

        logger.info(f"Added {n_total} samples to {output_file_name}")


def _check_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} not found")
    return path


def _load_audio_list(audio_files_list: str, num_workers: int) -> list[str]:
    list_path = Path(audio_files_list).resolve()
    working_dir = list_path.parent

    with open(list_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    candidates = [str((working_dir / line).resolve()) for line in lines]
    with Pool(max(num_workers, 1)) as p:
        return list(p.imap(_check_file, candidates, 100))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "type", type=str, help="Either 'speech', 'noise', 'noisy' or 'rir'."
    )
    parser.add_argument(
        "audio_files",
        type=str,
        help="Text file containing one audio file path per line.",
    )
    parser.add_argument(
        "hdf5_db", type=str, help="HDF5 file where data will be stored."
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_freq",
        type=int,
        default=-1,
        help=(
            "Only frequencies below this value are considered during training loss. "
            "Useful for upsampled signals."
        ),
    )
    parser.add_argument("--sr", type=int, default=48_000, help="Target sample rate.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="int16",
        help="Storage dtype: float32 or int16.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="pcm",
        help="Storage codec: pcm, vorbis or flac.",
    )
    parser.add_argument("--mono", action="store_true")
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="HDF5 dataset compression (e.g. gzip).",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO")

    if args.type not in AUDIO_DATASET_TYPES:
        raise ValueError(
            f"Dataset type must be one of {AUDIO_DATASET_TYPES}, but got {args.type}"
        )

    if not args.hdf5_db.endswith(".hdf5"):
        args.hdf5_db += ".hdf5"

    files = _load_audio_list(args.audio_files, args.num_workers)
    logger.info(f"Validated {len(files)} files")

    write_to_h5(
        output_file_name=args.hdf5_db,
        ds_type=args.type,
        audio_file_names=files,
        sr=args.sr,
        max_freq=args.max_freq,
        dtype=args.dtype,
        codec=args.codec.lower(),
        mono=args.mono,
        compression=args.compression,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
