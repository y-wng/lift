#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.shared.lerobot_reader import LeRobotDataReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read local LeRobot datasets directly from parquet episodes.")
    parser.add_argument("dataset", help="Dataset path or repo_id such as flexiv/towelv3_100.")
    parser.add_argument("--root", type=Path, default=None, help="Optional LeRobot root when dataset is a repo_id.")
    parser.add_argument("--episode", type=int, default=None, help="Read one episode.")
    parser.add_argument("--frame", type=int, default=None, help="Read one frame from --episode.")
    parser.add_argument("--column", type=str, default=None, help="Read one column, optionally within one episode.")
    parser.add_argument("--columns", nargs="+", default=None, help="Restrict output to these columns.")
    parser.add_argument("--decode-images", action="store_true", help="Decode image columns.")
    parser.add_argument(
        "--image-as-pil",
        action="store_true",
        help="When decoding images, keep them as PIL.Image instead of numpy arrays.",
    )
    parser.add_argument("--show-meta", action="store_true", help="Include episode metadata for --episode.")
    parser.add_argument("--stats", action="store_true", help="Print compact stats for ndarray outputs.")
    parser.add_argument("--preview", type=int, default=5, help="How many rows/items to preview in textual output.")
    return parser.parse_args()


def _describe_value(value: Any, preview: int) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        description = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size > 0:
            description["preview"] = np.asarray(value[:preview]).tolist()
            if np.issubdtype(value.dtype, np.number):
                description["min"] = float(np.min(value))
                description["max"] = float(np.max(value))
                description["mean"] = float(np.mean(value))
        else:
            description["preview"] = []
        return description

    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "preview_types": [type(item).__name__ for item in value[:preview]],
        }

    if hasattr(value, "size") and hasattr(value, "mode"):
        return {
            "type": type(value).__name__,
            "size": list(value.size),
            "mode": getattr(value, "mode", None),
        }

    return {
        "type": type(value).__name__,
        "value": value,
    }


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    reader = LeRobotDataReader(args.dataset, root=args.root)
    image_as_numpy = not args.image_as_pil

    if args.frame is not None:
        if args.episode is None:
            raise ValueError("--frame requires --episode.")
        frame = reader.read_frame(
            args.episode,
            args.frame,
            columns=args.columns,
            decode_images=args.decode_images,
            image_as_numpy=image_as_numpy,
        )
        _print_json({key: _describe_value(value, args.preview) for key, value in frame.items()})
        return

    if args.column is not None:
        value = reader.read_column(
            args.column,
            episode_index=args.episode,
            decode_images=args.decode_images,
            image_as_numpy=image_as_numpy,
        )
        payload = _describe_value(value, args.preview)
        if args.stats and isinstance(value, np.ndarray) and value.size > 0 and np.issubdtype(value.dtype, np.number):
            payload["std"] = float(np.std(value))
            if value.ndim >= 2:
                payload["per_dim_mean"] = np.mean(value, axis=0).tolist()
                payload["per_dim_std"] = np.std(value, axis=0).tolist()
        _print_json(payload)
        return

    if args.episode is not None:
        episode = reader.read_episode(
            args.episode,
            columns=args.columns,
            decode_images=args.decode_images,
            image_as_numpy=image_as_numpy,
        )
        payload = {
            "episode_index": args.episode,
            "columns": {key: _describe_value(value, args.preview) for key, value in episode.items()},
        }
        if args.show_meta:
            payload["episode_meta"] = reader.get_episode_metadata(args.episode)
        _print_json(payload)
        return

    _print_json(reader.summary())


if __name__ == "__main__":
    main()
