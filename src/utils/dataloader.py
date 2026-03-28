import bisect

from src.dataloader.hdf5 import Hdf5Dataset


def lookup(
    datasets: list[Hdf5Dataset], cum: list[int], idx: int
) -> tuple[Hdf5Dataset, int]:
    """Translate a flat index into ``(dataset, key_idx)`` using cumulative sizes."""
    ds_idx = bisect.bisect_right(cum, idx)
    key_idx = idx - (cum[ds_idx - 1] if ds_idx > 0 else 0)
    return datasets[ds_idx], key_idx
