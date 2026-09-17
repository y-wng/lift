from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

_DEFAULT_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
_EPISODE_FILE_RE = re.compile(r"episode_(\d+)\.parquet$")


@dataclass(frozen=True)
class LeRobotFeature:
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class LeRobotEpisodeFile:
    episode_index: int
    path: Path


def resolve_lerobot_dataset_root(dataset: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a LeRobot dataset from an absolute path, relative path, or repo_id."""
    dataset_str = os.fspath(dataset)
    direct_path = Path(dataset_str).expanduser()
    if direct_path.exists():
        return direct_path.resolve()

    search_root = Path(root).expanduser().resolve() if root is not None else _DEFAULT_LEROBOT_HOME.resolve()
    candidates = [search_root / dataset_str]
    if "/" not in dataset_str:
        candidates.extend(sorted(search_root.glob(f"*/{dataset_str}")))

    existing = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            existing.append(resolved)

    if not existing:
        raise FileNotFoundError(
            f"Could not resolve dataset '{dataset_str}'. Tried local path and candidates under {search_root}."
        )
    if len(existing) > 1:
        joined = ", ".join(str(path) for path in existing)
        raise ValueError(f"Dataset name '{dataset_str}' is ambiguous under {search_root}: {joined}")
    return existing[0]


class LeRobotDataReader:
    """Direct reader for local LeRobot v2.x datasets stored as parquet episodes."""

    def __init__(self, dataset: str | os.PathLike[str], *, root: str | os.PathLike[str] | None = None) -> None:
        self.dataset_root = resolve_lerobot_dataset_root(dataset, root=root)
        self.meta_dir = self.dataset_root / "meta"
        self.data_dir = self.dataset_root / "data"
        self.info_path = self.meta_dir / "info.json"
        if not self.info_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata file: {self.info_path}")
        self.info = json.loads(self.info_path.read_text())
        self.features = self._parse_features(self.info.get("features", {}))
        self._episode_files = self._scan_episode_files()
        self._episodes_meta = self._load_jsonl_by_episode(self.meta_dir / "episodes.jsonl")
        self._episodes_stats = self._load_jsonl_by_episode(self.meta_dir / "episodes_stats.jsonl")
        self.tasks = self._load_jsonl(self.meta_dir / "tasks.jsonl")

    @classmethod
    def from_repo_id(cls, repo_id: str, *, root: str | os.PathLike[str] | None = None) -> LeRobotDataReader:
        return cls(repo_id, root=root)

    @property
    def repo_id(self) -> str:
        try:
            return str(self.dataset_root.relative_to(_DEFAULT_LEROBOT_HOME.resolve()))
        except ValueError:
            return self.dataset_root.name

    @property
    def fps(self) -> float | None:
        fps = self.info.get("fps")
        return float(fps) if fps is not None else None

    @property
    def robot_type(self) -> str | None:
        return self.info.get("robot_type")

    @property
    def episode_indices(self) -> list[int]:
        return [item.episode_index for item in self._episode_files]

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "repo_id": self.repo_id,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "feature_names": list(self.features),
            "episode_count_from_files": len(self._episode_files),
            "episode_indices": self.episode_indices,
            "info_total_episodes": self.info.get("total_episodes"),
            "info_total_frames": self.info.get("total_frames"),
            "info_total_chunks": self.info.get("total_chunks"),
        }

    def get_episode_path(self, episode_index: int) -> Path:
        for item in self._episode_files:
            if item.episode_index == episode_index:
                return item.path
        raise KeyError(f"Episode {episode_index} not found in {self.dataset_root}")

    def get_episode_metadata(self, episode_index: int) -> dict[str, Any]:
        meta = {}
        if episode_index in self._episodes_meta:
            meta["episode"] = self._episodes_meta[episode_index]
        if episode_index in self._episodes_stats:
            meta["stats"] = self._episodes_stats[episode_index]
        return meta

    def read_table(self, episode_index: int, columns: list[str] | tuple[str, ...] | None = None) -> pa.Table:
        path = self.get_episode_path(episode_index)
        return pq.read_table(path, columns=list(columns) if columns is not None else None)

    def read_episode(
        self,
        episode_index: int,
        *,
        columns: list[str] | tuple[str, ...] | None = None,
        decode_images: bool = False,
        image_as_numpy: bool = True,
    ) -> dict[str, Any]:
        table = self.read_table(episode_index, columns=columns)
        return self._table_to_dict(table, decode_images=decode_images, image_as_numpy=image_as_numpy)

    def iter_episodes(
        self,
        *,
        columns: list[str] | tuple[str, ...] | None = None,
        decode_images: bool = False,
        image_as_numpy: bool = True,
    ):
        for episode_index in self.episode_indices:
            yield (
                episode_index,
                self.read_episode(
                    episode_index,
                    columns=columns,
                    decode_images=decode_images,
                    image_as_numpy=image_as_numpy,
                ),
            )

    def read_column(
        self,
        column: str,
        *,
        episode_index: int | None = None,
        decode_images: bool = False,
        image_as_numpy: bool = True,
    ) -> Any:
        if episode_index is not None:
            episode = self.read_episode(
                episode_index,
                columns=[column],
                decode_images=decode_images,
                image_as_numpy=image_as_numpy,
            )
            return episode[column]

        values = []
        for _, episode in self.iter_episodes(
            columns=[column],
            decode_images=decode_images,
            image_as_numpy=image_as_numpy,
        ):
            values.append(episode[column])

        if not values:
            return np.empty((0,), dtype=np.float32)
        if isinstance(values[0], np.ndarray):
            return np.concatenate(values, axis=0)

        merged = []
        for value in values:
            merged.extend(value)
        return merged

    def read_frame(
        self,
        episode_index: int,
        frame_index: int,
        *,
        columns: list[str] | tuple[str, ...] | None = None,
        decode_images: bool = False,
        image_as_numpy: bool = True,
    ) -> dict[str, Any]:
        table = self.read_table(episode_index, columns=columns)
        if frame_index < 0 or frame_index >= table.num_rows:
            raise IndexError(
                f"Frame {frame_index} is out of range for episode {episode_index} with {table.num_rows} rows."
            )
        row_table = table.slice(frame_index, 1)
        row = self._table_to_dict(row_table, decode_images=decode_images, image_as_numpy=image_as_numpy)
        return {key: self._squeeze_first_dim(value) for key, value in row.items()}

    def decode_image_cell(self, cell: object, *, as_numpy: bool = False) -> Image.Image | np.ndarray:
        if isinstance(cell, dict):
            raw_bytes = cell.get("bytes")
            if raw_bytes:
                image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            else:
                raw_path = cell.get("path")
                if raw_path is None:
                    raise ValueError("Image cell dict must contain either 'bytes' or 'path'.")
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    image_path = self.dataset_root / image_path
                image = Image.open(image_path).convert("RGB")
        elif isinstance(cell, bytes | bytearray):
            image = Image.open(io.BytesIO(cell)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image cell type: {type(cell)}")

        if as_numpy:
            return np.asarray(image)
        return image

    def decode_image_column(
        self,
        column: str,
        *,
        episode_index: int,
        as_numpy: bool = True,
        stack: bool = True,
    ) -> np.ndarray | list[Image.Image] | list[np.ndarray]:
        cells = self.read_column(column, episode_index=episode_index, decode_images=False)
        decoded = [self.decode_image_cell(cell, as_numpy=as_numpy) for cell in cells]
        if as_numpy and stack and decoded:
            return np.stack(decoded, axis=0)
        return decoded

    def _scan_episode_files(self) -> list[LeRobotEpisodeFile]:
        files = []
        if not self.data_dir.is_dir():
            return files
        for path in sorted(self.data_dir.glob("chunk-*/episode_*.parquet")):
            match = _EPISODE_FILE_RE.match(path.name)
            if match is None:
                continue
            files.append(LeRobotEpisodeFile(episode_index=int(match.group(1)), path=path))
        return files

    def _table_to_dict(self, table: pa.Table, *, decode_images: bool, image_as_numpy: bool) -> dict[str, Any]:
        out = {}
        for column_name in table.column_names:
            feature = self.features.get(column_name)
            out[column_name] = self._convert_column(
                table[column_name],
                feature=feature,
                decode_images=decode_images,
                image_as_numpy=image_as_numpy,
            )
        return out

    def _convert_column(
        self,
        column: pa.ChunkedArray,
        *,
        feature: LeRobotFeature | None,
        decode_images: bool,
        image_as_numpy: bool,
    ) -> Any:
        if feature is not None and feature.dtype == "image":
            cells = column.to_pylist()
            if not decode_images:
                return cells
            decoded = [self.decode_image_cell(cell, as_numpy=image_as_numpy) for cell in cells]
            if image_as_numpy and decoded:
                return np.stack(decoded, axis=0)
            return decoded

        dtype = self._feature_numpy_dtype(feature)
        values = column.to_pylist()
        if dtype is None:
            return np.asarray(values)
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def _squeeze_first_dim(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.shape[0] == 1:
            return value[0]
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    @staticmethod
    def _parse_features(raw_features: dict[str, dict[str, Any]]) -> dict[str, LeRobotFeature]:
        features = {}
        for name, spec in raw_features.items():
            names = spec.get("names")
            features[name] = LeRobotFeature(
                dtype=spec.get("dtype", "unknown"),
                shape=tuple(spec.get("shape", [])),
                names=tuple(names) if names is not None else None,
            )
        return features

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @classmethod
    def _load_jsonl_by_episode(cls, path: Path) -> dict[int, dict[str, Any]]:
        out = {}
        for row in cls._load_jsonl(path):
            episode_index = row.get("episode_index")
            if episode_index is None:
                continue
            out[int(episode_index)] = row
        return out

    @staticmethod
    def _feature_numpy_dtype(feature: LeRobotFeature | None) -> np.dtype | None:
        if feature is None:
            return None
        dtype = feature.dtype
        if dtype == "float16":
            return np.float16
        if dtype == "float32":
            return np.float32
        if dtype == "float64":
            return np.float64
        if dtype == "int8":
            return np.int8
        if dtype == "int16":
            return np.int16
        if dtype == "int32":
            return np.int32
        if dtype == "int64":
            return np.int64
        if dtype == "uint8":
            return np.uint8
        if dtype == "bool":
            return np.bool_
        return None
