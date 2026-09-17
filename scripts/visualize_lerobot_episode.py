#!/usr/bin/env python3

import argparse
import io
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import contextlib

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
import pyarrow as pa
import pyarrow.parquet as pq

CONTROL_FLAG_COLORS = {
    1: "#1f77b4",  # blue
    -1: "#f1c40f",  # yellow
    0: "#9e9e9e",  # fallback / unknown
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a local LeRobot episode directory or a dataset root that contains episode_* folders."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Episode directory path, or a parent directory that contains episode_* subdirectories.",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Episode selector when input_path is a parent directory. Supports 0, 000, 0000, episode_0000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save visualizations. Default: <parent>/_viz/<episode_name>.",
    )
    parser.add_argument(
        "--contact-sheet-frames",
        type=int,
        default=16,
        help="How many frames to sample into the contact sheet.",
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=96,
        help="Maximum number of frames to keep in the preview gif.",
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip gif generation.",
    )
    return parser.parse_args()


def is_lerobot_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def resolve_episode_path(input_path: Path, episode_spec: str | None) -> Path:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")
    if is_lerobot_dataset_dir(input_path):
        return input_path
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")

    if episode_spec is None:
        candidates = sorted(path for path in input_path.iterdir() if is_lerobot_dataset_dir(path))
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"{input_path} is not an episode directory. "
            "Please pass --episode when the directory contains multiple episode_* subdirectories."
        )

    names = build_episode_name_candidates(episode_spec)
    for name in names:
        candidate = input_path / name
        if is_lerobot_dataset_dir(candidate):
            return candidate

    raise FileNotFoundError(f"Could not resolve episode '{episode_spec}' under {input_path}. Tried: {', '.join(names)}")


def build_episode_name_candidates(spec: str) -> list[str]:
    spec = spec.strip()
    candidates = [spec]

    numeric = None
    if spec.isdigit():
        numeric = int(spec)
    elif spec.startswith("episode_"):
        tail = spec.split("_", 1)[1]
        if tail.isdigit():
            numeric = int(tail)

    if numeric is not None:
        candidates.extend(
            [
                f"episode_{numeric:04d}",
                f"episode_{numeric:03d}",
                f"ep_{numeric:03d}",
            ]
        )

    deduped = []
    seen = set()
    for item in candidates:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def load_episode_info(episode_path: Path) -> dict:
    info_path = episode_path / "meta" / "info.json"
    return json.loads(info_path.read_text())


def load_episode_table(episode_path: Path) -> pa.Table:
    parquet_files = sorted((episode_path / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {episode_path / 'data'}")
    tables = [pq.read_table(path) for path in parquet_files]
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables)


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = max(1, min(length, count))
    if count == 1:
        return [0]
    return sorted({round(x) for x in np.linspace(0, length - 1, count)})


def decode_image_cell(cell: object, episode_path: Path) -> Image.Image:
    if isinstance(cell, dict):
        raw_bytes = cell.get("bytes")
        if raw_bytes:
            with Image.open(io.BytesIO(raw_bytes)) as image:
                return image.convert("RGB")
        raw_path = cell.get("path")
        if raw_path:
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = episode_path / image_path
            with Image.open(image_path) as image:
                return image.convert("RGB")
    if isinstance(cell, bytes | bytearray):
        with Image.open(io.BytesIO(cell)) as image:
            return image.convert("RGB")
    raise ValueError(f"Unsupported image cell type: {type(cell)}")


def draw_frame_label(image: Image.Image, label: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 126, 28), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    return image


def save_contact_sheet(
    image_cells: list[object],
    episode_path: Path,
    indices: list[int],
    timestamps: np.ndarray | None,
    output_path: Path,
) -> None:
    if not indices:
        return

    first_image = decode_image_cell(image_cells[indices[0]], episode_path)
    tile_width, tile_height = first_image.size
    cols = math.ceil(math.sqrt(len(indices)))
    rows = math.ceil(len(indices) / cols)
    margin = 8

    canvas = Image.new(
        "RGB",
        (
            cols * tile_width + (cols + 1) * margin,
            rows * tile_height + (rows + 1) * margin,
        ),
        color=(24, 24, 24),
    )

    for slot, frame_index in enumerate(indices):
        row, col = divmod(slot, cols)
        image = decode_image_cell(image_cells[frame_index], episode_path)
        image = ImageOps.pad(image, (tile_width, tile_height), color=(0, 0, 0))

        label = f"#{frame_index}"
        if timestamps is not None and frame_index < len(timestamps):
            label += f"  t={timestamps[frame_index]:.2f}s"
        image = draw_frame_label(image, label)

        x0 = margin + col * (tile_width + margin)
        y0 = margin + row * (tile_height + margin)
        canvas.paste(image, (x0, y0))

    canvas.save(output_path)


def save_preview_gif(
    image_cells: list[object],
    episode_path: Path,
    indices: list[int],
    timestamps: np.ndarray | None,
    fps: float | None,
    output_path: Path,
) -> None:
    if len(indices) < 2:
        return

    frames = []
    for frame_index in indices:
        image = decode_image_cell(image_cells[frame_index], episode_path)
        image = ImageOps.pad(image, (224, 224), color=(0, 0, 0))
        label = f"#{frame_index}"
        if timestamps is not None and frame_index < len(timestamps):
            label += f"  t={timestamps[frame_index]:.2f}s"
        frames.append(draw_frame_label(image, label))

    if timestamps is not None and len(indices) > 1:
        sampled_timestamps = timestamps[indices]
        frame_gap_s = float(np.mean(np.diff(sampled_timestamps)))
    else:
        step = max(1, round(len(image_cells) / len(indices)))
        base_fps = fps if fps and fps > 0 else 10.0
        frame_gap_s = step / base_fps

    duration_ms = max(20, round(frame_gap_s * 1000.0))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def infer_dim_labels(key: str, feature_info: dict, dim: int) -> list[str]:
    if key in {"state", "actions"} and dim == 7:
        return ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    if key.endswith("wrench") and dim == 6:
        return ["fx", "fy", "fz", "tx", "ty", "tz"]
    names = feature_info.get("names")
    if isinstance(names, list) and len(names) == dim:
        return [str(name) for name in names]
    return [f"dim_{idx}" for idx in range(dim)]


def normalize_control_flags(control_flags: np.ndarray | None, length: int) -> np.ndarray | None:
    if control_flags is None:
        return None
    flags = np.asarray(control_flags, dtype=np.float64).reshape(-1)
    if len(flags) != length:
        return None
    normalized = np.zeros(len(flags), dtype=np.int8)
    normalized[np.isclose(flags, 1.0)] = 1
    normalized[np.isclose(flags, -1.0)] = -1
    return normalized


def control_flag_to_color(flag: int) -> str:
    return CONTROL_FLAG_COLORS.get(int(flag), CONTROL_FLAG_COLORS[0])


def iter_control_runs(control_flags: np.ndarray | None, length: int) -> list[tuple[int, int, int]]:
    if length <= 0:
        return []
    if control_flags is None:
        return [(0, length, 0)]

    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(control_flags[0])
    for idx in range(1, length):
        value = int(control_flags[idx])
        if value != current:
            runs.append((start, idx, current))
            start = idx
            current = value
    runs.append((start, length, current))
    return runs


def make_control_legend_handles(
    *, include_styles: bool = False, style_labels: tuple[str, str] | None = None
) -> list[Line2D]:
    handles = []
    if include_styles and style_labels is not None:
        handles.extend(
            [
                Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label=style_labels[0]),
                Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label=style_labels[1]),
            ]
        )
    handles.extend(
        [
            Line2D([0], [0], color=CONTROL_FLAG_COLORS[1], linestyle="-", linewidth=2.0, label="control=1"),
            Line2D([0], [0], color=CONTROL_FLAG_COLORS[-1], linestyle="-", linewidth=2.0, label="control=-1"),
        ]
    )
    return handles


def plot_control_colored_line(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    control_flags: np.ndarray | None,
    *,
    linewidth: float = 1.2,
    linestyle: str = "-",
) -> None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) == 0:
        return
    if len(x) == 1:
        axis.plot(x, y, color=control_flag_to_color(0), linewidth=linewidth, linestyle=linestyle)
        return

    for start, end, flag in iter_control_runs(control_flags, len(x)):
        if end - start <= 0:
            continue
        axis.plot(
            x[start:end],
            y[start:end],
            color=control_flag_to_color(flag),
            linewidth=linewidth,
            linestyle=linestyle,
        )


def build_colored_3d_collection(
    coords: np.ndarray,
    control_flags: np.ndarray | None,
    *,
    linewidth: float,
    linestyle: str,
) -> Line3DCollection | None:
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) < 2:
        return None
    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    if control_flags is None:
        colors = [control_flag_to_color(0)] * len(segments)
    else:
        colors = [control_flag_to_color(flag) for flag in control_flags[:-1]]
    return Line3DCollection(segments, colors=colors, linewidths=linewidth, linestyles=linestyle)


def plot_multidim_series(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    x_label: str,
    control_flags: np.ndarray | None = None,
) -> None:
    if y.ndim == 1:
        y = y[:, None]

    dim = y.shape[1]
    cols = 1 if dim <= 3 else 2
    rows = math.ceil(dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3 * rows, 4)), sharex=True, squeeze=False)

    for idx, axis in enumerate(axes.flat):
        if idx >= dim:
            axis.axis("off")
            continue
        plot_control_colored_line(axis, x, y[:, idx], control_flags, linewidth=1.2, linestyle="-")
        axis.set_ylabel(labels[idx])
        axis.grid(alpha=0.3)
        if control_flags is not None:
            axis.legend(handles=make_control_legend_handles(), loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.supxlabel(x_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_series(
    x: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    x_label: str,
    label_a: str,
    label_b: str,
    control_flags: np.ndarray | None = None,
) -> None:
    if y_a.ndim == 1:
        y_a = y_a[:, None]
    if y_b.ndim == 1:
        y_b = y_b[:, None]

    dim = y_a.shape[1]
    cols = 1 if dim <= 3 else 2
    rows = math.ceil(dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3 * rows, 4)), sharex=True, squeeze=False)

    for idx, axis in enumerate(axes.flat):
        if idx >= dim:
            axis.axis("off")
            continue
        plot_control_colored_line(axis, x, y_a[:, idx], control_flags, linewidth=1.2, linestyle="-")
        plot_control_colored_line(axis, x, y_b[:, idx], control_flags, linewidth=1.2, linestyle="--")
        axis.set_ylabel(labels[idx])
        axis.grid(alpha=0.3)
        axis.legend(
            handles=make_control_legend_handles(include_styles=True, style_labels=(label_a, label_b)),
            loc="upper right",
            fontsize=8,
        )

    fig.suptitle(title)
    fig.supxlabel(x_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def extract_xyz_trajectories(array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    if array.ndim != 2 or array.shape[1] < 3:
        return []
    if array.shape[1] >= 10:
        return [
            ("left_arm", array[:, :3]),
            ("right_arm", array[:, 7:10]),
        ]
    return [("arm", array[:, :3])]


def set_equalish_3d_aspect(axis: plt.Axes, coords: np.ndarray) -> None:
    extent = np.ptp(coords, axis=0).astype(np.float64)
    extent = np.where(extent < 1e-9, 1.0, extent)
    with contextlib.suppress(AttributeError):
        axis.set_box_aspect(extent)


def plot_3d_trajectory(
    array: np.ndarray,
    title: str,
    output_path: Path,
    line_label: str,
    control_flags: np.ndarray | None = None,
) -> None:
    trajectories = extract_xyz_trajectories(array)
    if not trajectories:
        return

    fig = plt.figure(figsize=(7 * len(trajectories), 6))
    for idx, (name, coords) in enumerate(trajectories, start=1):
        axis = fig.add_subplot(1, len(trajectories), idx, projection="3d")
        collection = build_colored_3d_collection(coords, control_flags, linewidth=1.5, linestyle="-")
        if collection is not None:
            axis.add_collection3d(collection)
        axis.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="green", s=36, label="start")
        axis.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color="red", s=36, label="end")
        axis.auto_scale_xyz(coords[:, 0], coords[:, 1], coords[:, 2])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(name)
        axis.grid(visible=True, alpha=0.3)
        axis.legend(
            handles=[
                Line2D([0], [0], color=CONTROL_FLAG_COLORS[1], linewidth=2.0, label=f"{line_label}, control=1"),
                Line2D([0], [0], color=CONTROL_FLAG_COLORS[-1], linewidth=2.0, label=f"{line_label}, control=-1"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=7, label="start"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=7, label="end"),
            ],
            loc="best",
            fontsize=8,
        )
        set_equalish_3d_aspect(axis, coords)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_3d_trajectory(
    array_a: np.ndarray,
    array_b: np.ndarray,
    title: str,
    output_path: Path,
    label_a: str,
    label_b: str,
    control_flags: np.ndarray | None = None,
) -> None:
    trajectories_a = extract_xyz_trajectories(array_a)
    trajectories_b = extract_xyz_trajectories(array_b)
    if not trajectories_a or not trajectories_b:
        return

    num_plots = min(len(trajectories_a), len(trajectories_b))
    fig = plt.figure(figsize=(7 * num_plots, 6))

    for idx in range(num_plots):
        name_a, coords_a = trajectories_a[idx]
        name_b, coords_b = trajectories_b[idx]
        coords_a = coords_a.astype(np.float64)
        coords_b = coords_b.astype(np.float64)

        axis = fig.add_subplot(1, num_plots, idx + 1, projection="3d")
        collection_a = build_colored_3d_collection(coords_a, control_flags, linewidth=1.5, linestyle="-")
        collection_b = build_colored_3d_collection(coords_b, control_flags, linewidth=1.5, linestyle="--")
        if collection_a is not None:
            axis.add_collection3d(collection_a)
        if collection_b is not None:
            axis.add_collection3d(collection_b)
        axis.scatter(coords_b[0, 0], coords_b[0, 1], coords_b[0, 2], color="green", s=36, label=f"{label_b} start")
        axis.scatter(coords_b[-1, 0], coords_b[-1, 1], coords_b[-1, 2], color="red", s=36, label=f"{label_b} end")
        all_coords = np.vstack([coords_a, coords_b])
        axis.auto_scale_xyz(all_coords[:, 0], all_coords[:, 1], all_coords[:, 2])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(name_b if name_a == name_b else f"{name_a} / {name_b}")
        axis.grid(visible=True, alpha=0.3)
        axis.legend(
            handles=[
                Line2D([0], [0], color="black", linewidth=1.5, linestyle="-", label=label_a),
                Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label=label_b),
                Line2D([0], [0], color=CONTROL_FLAG_COLORS[1], linewidth=2.0, label="control=1"),
                Line2D([0], [0], color=CONTROL_FLAG_COLORS[-1], linewidth=2.0, label="control=-1"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=7, label="start"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=7, label="end"),
            ],
            loc="best",
            fontsize=8,
        )
        set_equalish_3d_aspect(axis, all_coords)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize_array(array: np.ndarray) -> dict:
    summary = {
        "shape": list(array.shape),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }

    if array.ndim == 2 and array.shape[1] <= 16:
        summary["per_dim_mean"] = [float(x) for x in np.mean(array, axis=0)]
        summary["per_dim_std"] = [float(x) for x in np.std(array, axis=0)]
        summary["first_row"] = [float(x) for x in array[0]]
        summary["last_row"] = [float(x) for x in array[-1]]
    elif array.ndim == 1:
        summary["first_value"] = float(array[0])
        summary["last_value"] = float(array[-1])

    return summary


def build_summary(
    episode_path: Path,
    output_dir: Path,
    info: dict,
    raw_data: dict,
    numeric_data: dict[str, np.ndarray],
    image_keys: list[str],
) -> dict:
    summary = {
        "episode_path": str(episode_path),
        "output_dir": str(output_dir),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "num_frames": len(next(iter(raw_data.values()))) if raw_data else 0,
        "image_keys": image_keys,
        "numeric_keys": sorted(numeric_data.keys()),
        "features": info.get("features", {}),
        "arrays": {},
    }

    for key, array in numeric_data.items():
        summary["arrays"][key] = summarize_array(array)

    timestamps = numeric_data.get("timestamp")
    if timestamps is not None and len(timestamps) > 1:
        deltas = np.diff(timestamps)
        summary["timestamps"] = {
            "start_s": float(timestamps[0]),
            "end_s": float(timestamps[-1]),
            "duration_s": float(timestamps[-1] - timestamps[0]),
            "delta_mean_s": float(np.mean(deltas)),
            "delta_std_s": float(np.std(deltas)),
            "delta_min_s": float(np.min(deltas)),
            "delta_max_s": float(np.max(deltas)),
        }

    return summary


def main() -> None:
    args = parse_args()

    episode_path = resolve_episode_path(args.input_path, args.episode)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = episode_path.parent / "_viz" / episode_path.name
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = load_episode_info(episode_path)
    table = load_episode_table(episode_path)
    raw_data = table.to_pydict()
    feature_info = info.get("features", {})

    image_keys = []
    numeric_data: dict[str, np.ndarray] = {}

    for key, values in raw_data.items():
        dtype = feature_info.get(key, {}).get("dtype")
        if dtype == "image":
            image_keys.append(key)
            continue
        numeric_data[key] = np.asarray(values)

    control_flags = None
    if "control_flag" in numeric_data:
        control_flags = normalize_control_flags(numeric_data["control_flag"], len(numeric_data["control_flag"]))

    x = numeric_data.get("timestamp")
    if x is not None and len(x) == len(next(iter(raw_data.values()))):
        x = x.astype(np.float64)
        x_label = "timestamp (s)"
    else:
        x = np.arange(len(next(iter(raw_data.values()))), dtype=np.int64)
        x_label = "frame index"

    summary = build_summary(episode_path, output_dir, info, raw_data, numeric_data, image_keys)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if "timestamp" in numeric_data and len(numeric_data["timestamp"]) > 1:
        timestamp_deltas = np.diff(numeric_data["timestamp"])
        plot_multidim_series(
            np.arange(len(timestamp_deltas)),
            timestamp_deltas,
            ["delta_t"],
            "Timestamp Deltas",
            output_dir / "timestamp_deltas.png",
            "frame step",
        )

    for key, array in numeric_data.items():
        if key in {"timestamp", "frame_index", "episode_index", "index", "task_index"}:
            continue
        feature = feature_info.get(key, {})
        labels = infer_dim_labels(key, feature, array.shape[1] if array.ndim > 1 else 1)
        plot_multidim_series(
            x,
            array.astype(np.float64),
            labels,
            f"{key} over time",
            output_dir / f"{key}.png",
            x_label,
            control_flags=control_flags if key != "control_flag" else None,
        )

    if "actions" in numeric_data:
        plot_3d_trajectory(
            numeric_data["actions"].astype(np.float64),
            "Actions 3D Trajectory",
            output_dir / "actions_3d.png",
            "actions",
            control_flags=control_flags,
        )

    if "state" in numeric_data and "actions" in numeric_data:
        state = numeric_data["state"].astype(np.float64)
        actions = numeric_data["actions"].astype(np.float64)
        if state.shape == actions.shape:
            labels = infer_dim_labels("actions", feature_info.get("actions", {}), actions.shape[1])
            plot_overlay_series(
                x,
                state,
                actions,
                labels,
                "State vs Actions",
                output_dir / "state_vs_actions.png",
                x_label,
                "state",
                "actions",
                control_flags=control_flags,
            )
            plot_multidim_series(
                x,
                actions - state,
                labels,
                "Actions - State",
                output_dir / "actions_minus_state.png",
                x_label,
                control_flags=control_flags,
            )
            plot_overlay_3d_trajectory(
                state,
                actions,
                "State vs Actions 3D Trajectory",
                output_dir / "state_vs_actions_3d.png",
                "state",
                "actions",
                control_flags=control_flags,
            )

    timestamps = numeric_data.get("timestamp")
    fps = info.get("fps")
    for image_key in image_keys:
        image_cells = raw_data[image_key]
        contact_sheet_indices = evenly_spaced_indices(len(image_cells), args.contact_sheet_frames)
        save_contact_sheet(
            image_cells,
            episode_path,
            contact_sheet_indices,
            timestamps,
            output_dir / f"{image_key}_contact_sheet.png",
        )

        if not args.skip_gif:
            gif_indices = evenly_spaced_indices(len(image_cells), args.gif_frames)
            save_preview_gif(
                image_cells,
                episode_path,
                gif_indices,
                timestamps,
                float(fps) if fps is not None else None,
                output_dir / f"{image_key}_preview.gif",
            )

    print(f"[VISUALIZED] episode={episode_path}")
    print(f"[VISUALIZED] output_dir={output_dir}")
    print(f"[VISUALIZED] summary={summary_path}")


if __name__ == "__main__":
    main()
