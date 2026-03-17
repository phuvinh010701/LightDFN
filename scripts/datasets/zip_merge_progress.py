#Reference: https://github.com/sealad886/DeepFilterNet4
#!/usr/bin/env python3
"""Merge split ZIP archives while displaying a tqdm progress bar.

This wraps ``zip -s 0 ... --out ...`` and translates its per-file ``copying:``
output into a single progress bar, which is much easier to watch during the
large FSD50K merge step.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a split ZIP archive with a progress bar.")
    parser.add_argument("--download-dir", required=True, help="Directory containing the split ZIP files.")
    parser.add_argument("--zip-base", required=True, help="ZIP basename, e.g. FSD50K.dev_audio.zip")
    return parser.parse_args()


def count_zip_members(download_dir: Path, zip_base: str) -> int:
    result = subprocess.run(
        ["unzip", "-Z1", zip_base],
        cwd=download_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to enumerate ZIP entries for {zip_base}: {result.stderr.strip()}")
    return sum(1 for line in result.stdout.splitlines() if line.strip())


def verify_zip_archive(zip_path: Path) -> bool:
    result = subprocess.run(
        ["unzip", "-tqq", zip_path.name],
        cwd=zip_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def merge_split_zip_with_progress(download_dir: Path, zip_base: str) -> Path:
    total_members = count_zip_members(download_dir, zip_base)
    merged_name = f"{zip_base}.merged.zip"
    merged_path = download_dir / merged_name
    temp_merged_name = f"{zip_base}.merged.tmp.zip"
    temp_merged_path = download_dir / temp_merged_name

    if merged_path.exists() and verify_zip_archive(merged_path):
        tqdm.write(f"[skip] merged archive already exists: {merged_path}")
        return merged_path

    if temp_merged_path.exists():
        temp_merged_path.unlink()

    process = subprocess.Popen(
        ["zip", "-s", "0", zip_base, "--out", temp_merged_name],
        cwd=download_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    with tqdm(total=total_members, desc=f"Merging {zip_base}", unit="file", dynamic_ncols=True) as pbar:
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("copying: "):
                pbar.update(1)
                pbar.set_postfix_str(line.removeprefix("copying: "))
            else:
                tqdm.write(line)

        return_code = process.wait()
        if return_code != 0:
            if temp_merged_path.exists():
                temp_merged_path.unlink()
            raise RuntimeError(f"zip merge failed for {zip_base} with exit code {return_code}")

        if pbar.n < total_members:
            pbar.update(total_members - pbar.n)

    temp_merged_path.replace(merged_path)
    return merged_path


def main() -> int:
    args = parse_args()
    merge_split_zip_with_progress(Path(args.download_dir), args.zip_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
