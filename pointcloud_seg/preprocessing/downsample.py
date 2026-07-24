import os
import json
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from pointcloud_seg.config import Config


def downsample_tower(raw_path: str, config: Config) -> str:
    """Returns path to downsampled .las, or raises on failure."""
    stem = Path(raw_path).stem
    out_filename = f"{stem}_{config.downsample_cm}cm.las"
    out_path = os.path.join(config.downsampled_dir, out_filename)

    os.makedirs(config.downsampled_dir, exist_ok=True)

    radius = config.downsample_cm / 100.0

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": raw_path},
            {
                "type": "filters.sample",
                "radius": radius,
            },
            {
                "type": "writers.las",
                "filename": out_path,
                "dataformat_id": config.pdal_dataformat,
                "minor_version": 4,
                "extra_dims": "all",
            },
        ]
    }

    tmp_json = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pipeline, f)
            tmp_json = f.name

        result = subprocess.run(
            ["pdal", "pipeline", tmp_json],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PDAL failed for {raw_path}:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
    finally:
        if tmp_json and os.path.exists(tmp_json):
            os.remove(tmp_json)

    return out_path


def _worker(args: Tuple[str, Config]) -> Tuple[str, str, bool]:
    raw_path, config = args
    try:
        out = downsample_tower(raw_path, config)
        return raw_path, out, True
    except Exception as e:
        return raw_path, str(e), False


def batch_downsample(raw_paths: List[str], config: Config) -> Tuple[List[str], List[str]]:
    """Returns (succeeded_paths, failed_paths)."""
    args = [(p, config) for p in raw_paths]

    succeeded = []
    failed = []

    with Pool(processes=config.pdal_workers) as pool:
        for raw_path, out_or_err, ok in tqdm(
            pool.imap_unordered(_worker, args),
            total=len(args),
            desc="Downsampling (PDAL)",
            unit="tower",
        ):
            if ok:
                succeeded.append(out_or_err)
            else:
                print(f"[FAIL] {raw_path}: {out_or_err}")
                failed.append(raw_path)

    if failed:
        print(f"\n{len(failed)} file(s) failed downsampling:")
        for p in failed:
            print(f"  {p}")

    return succeeded, failed
