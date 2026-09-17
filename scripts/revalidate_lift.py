#!/usr/bin/env python3
"""Validate the local LIFT data, checkpoint, and normalization layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet

from openpi.shared.episode_schema import DEFAULT_INTERVENTION_VALUE
from openpi.shared.episode_schema import is_intervention_chunk

DATASETS = {
    "towel": ("towelv3_100", "Fold the towel twice into a triangle."),
    "book": ("book_insertion_v3_100", "Insert the book into the shelf."),
    "hanoi": (
        "hanoi_v1_100",
        "Solve the Tower of Hanoi by moving the rings onto the target peg.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")) / "flexiv",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(os.environ.get("OPENPI_CHECKPOINT_ROOT", "./checkpoints")),
    )
    parser.add_argument("--tasks", nargs="+", choices=DATASETS)
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=[],
        help="Validate an explicit offline dataset directory; repeat for multiple datasets.",
    )
    parser.add_argument("--expected-fps", type=int, default=10)
    parser.add_argument("--intervention-value", type=float, default=DEFAULT_INTERVENTION_VALUE)
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint params directory to validate.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("OPENPI_ASSET_ROOT", "~/.cache/openpi/openpi-assets")),
    )
    parser.add_argument("--online-dataset", help="Optional online dataset directory relative to --lerobot-root.")
    parser.add_argument("--action-horizon", type=int, default=10)
    return parser.parse_args()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _read_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {info_path}")
    return json.loads(info_path.read_text())


def _parquet_files(dataset_root: Path) -> list[Path]:
    files = sorted((dataset_root / "data").glob("**/episode_*.parquet"))
    if files:
        return files
    return sorted(dataset_root.glob("episode_*/data/**/episode_*.parquet"))


def _check_dataset(
    dataset_root: Path,
    *,
    online: bool,
    action_horizon: int,
    expected_fps: int = 10,
    intervention_value: float = DEFAULT_INTERVENTION_VALUE,
) -> dict[str, Any]:
    info = _read_info(dataset_root)
    features = info.get("features", {})
    required = {"left_wrist_img", "state", "actions"}
    if online:
        required.update({"left_wrench", "control_flag"})
    missing = sorted(required - features.keys())
    if missing:
        raise ValueError(f"{dataset_root} is missing features: {missing}")
    if info.get("fps") != expected_fps:
        raise ValueError(f"Expected {expected_fps} Hz data in {dataset_root}, got {info.get('fps')!r}")

    files = _parquet_files(dataset_root)
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_root}")

    valid_chunks = 0
    possible_chunks = 0
    control_values: set[float] = set()
    for episode_path in files:
        columns = ["actions"]
        if online:
            columns.append("control_flag")
        table = parquet.read_table(episode_path, columns=columns)
        frame_count = table.num_rows
        possible_chunks += max(0, frame_count - action_horizon + 1)
        if online:
            flags = np.asarray(table["control_flag"].to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1)
            control_values.update(float(value) for value in np.unique(flags))
            if flags.size >= action_horizon:
                valid_chunks += sum(
                    is_intervention_chunk(flags[start : start + action_horizon], intervention_value)
                    for start in range(flags.size - action_horizon + 1)
                )

    result = {
        "path": str(dataset_root),
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "parquet_files": len(files),
        "features": sorted(features),
    }
    if online:
        result.update(
            {
                "control_flag_values": sorted(control_values),
                "intervention_chunks": valid_chunks,
                "intervention_value": intervention_value,
                "possible_action_chunks": possible_chunks,
            }
        )
        if valid_chunks == 0:
            raise ValueError(f"No control_flag=={intervention_value:g} action chunks found in {dataset_root}")
    return result


def _check_checkpoint(candidate: Path) -> dict[str, Any]:
    if not candidate.is_dir():
        raise FileNotFoundError(f"Missing checkpoint params directory: {candidate}")
    asset_files = sorted(str(path) for path in (candidate.parent / "assets").glob("**/norm_stats.json"))
    return {
        "path": str(candidate),
        "exists": candidate.is_dir(),
        "same_checkpoint_norm_stats": asset_files,
    }


def _check_base_assets(asset_root: Path) -> dict[str, Any]:
    partial = asset_root / "checkpoints" / "pi05_base" / "params.partial"
    complete = asset_root / "checkpoints" / "pi05_base" / "params"
    return {
        "complete_params": str(complete),
        "complete_exists": complete.is_dir(),
        "partial_params": str(partial),
        "partial_exists": partial.is_dir(),
    }


def _find_norm_stats(checkpoint_root: Path, dataset_name: str) -> list[str]:
    pattern = f"assets/flexiv/{dataset_name}/norm_stats.json"
    return sorted(str(path) for path in checkpoint_root.glob(f"*/**/{pattern}"))


def main() -> None:
    args = parse_args()
    lerobot_root = _resolved(args.lerobot_root)
    result: dict[str, Any] = {"lerobot_root": str(lerobot_root), "offline": {}, "online": {}}
    selected_tasks = args.tasks if args.tasks is not None else ([] if args.dataset else list(DATASETS))
    selected_datasets = {task: DATASETS[task] for task in selected_tasks}
    for dataset in args.dataset:
        result["offline"][str(dataset)] = _check_dataset(
            _resolved(dataset),
            online=False,
            action_horizon=args.action_horizon,
            expected_fps=args.expected_fps,
            intervention_value=args.intervention_value,
        )
    for task, (dataset_name, _) in selected_datasets.items():
        result["offline"][task] = _check_dataset(
            lerobot_root / dataset_name,
            online=False,
            action_horizon=args.action_horizon,
            expected_fps=args.expected_fps,
            intervention_value=args.intervention_value,
        )
    if args.online_dataset:
        result["online"] = _check_dataset(
            lerobot_root / args.online_dataset,
            online=True,
            action_horizon=args.action_horizon,
            expected_fps=args.expected_fps,
            intervention_value=args.intervention_value,
        )
    if args.checkpoint:
        result["checkpoint"] = _check_checkpoint(_resolved(args.checkpoint))
    result["base_assets"] = _check_base_assets(_resolved(args.asset_root))
    result["norm_stats_candidates"] = {
        task: _find_norm_stats(_resolved(args.checkpoint_root), dataset_name)
        for task, (dataset_name, _) in selected_datasets.items()
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
