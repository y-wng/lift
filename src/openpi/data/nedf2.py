"""
Convert NEDF2 episodes into LeRobot datasets for Flexiv TDK recordings.

This converter is configuration-driven and supports:
- pi-style features (`left_wrist_img`, `state`, `actions`)
- extra aligned features such as `left_wrench`
- optional aligned control flags with configurable frame dropping for switching states
- per-feature multi-stream composition for single-arm and future dual-arm data
- per-episode or merged-dataset export layouts
- anomaly checks and timestamp alignment against the shortest image stream
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Any

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)


DEFAULT_REPO_NAME = Path("flexiv/book_insertion_v3_100")
REQUIRED_FEATURES = ("left_wrist_img", "state", "actions")
DEFAULT_OUTPUT_PATH = Path("dataset_lerobot")
DEFAULT_SOURCE_DIR = Path("dataset_nedf2")
CONTROL_FLAG_FEATURE_NAME = "control_flag"
EPISODE_DONE_MARKER_NAMES = (".done", ".down")
EPISODE_DONE_MARKER_NAME = EPISODE_DONE_MARKER_NAMES[0]
DEFAULT_FPS = 30
DEFAULT_FRAME_DOWNSAMPLE = 3
OUTPUT_LAYOUT_PER_EPISODE = "per-episode"
OUTPUT_LAYOUT_MERGED = "merged"
OUTPUT_LAYOUT_CHOICES = (OUTPUT_LAYOUT_PER_EPISODE, OUTPUT_LAYOUT_MERGED)
DEFAULT_OUTPUT_LAYOUT = OUTPUT_LAYOUT_PER_EPISODE


@dataclass(frozen=True)
class FeatureStreamSpec:
    feature_name: str
    dtype: str
    shape: tuple[int, ...]
    source_aliases: tuple[str, ...]
    source_keys: tuple[str, ...]
    source_latencies_s: tuple[float, ...]
    transform: str
    slice_range: tuple[int, int] | None

    @property
    def device_key(self) -> str:
        return self.source_keys[0]


@dataclass(frozen=True)
class EpisodeJob:
    episode_dir: Path
    template: dict[str, Any]
    task: str
    success: bool | None


@dataclass(frozen=True)
class FrameAlignmentStats:
    reference_frames: int
    aligned_frames: int
    dropped_control_flag_zero_frames: int
    saved_frames: int


def resolve_dataset_output_path(repo_id: Path, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path).expanduser()
    return HF_LEROBOT_HOME / repo_id


def parse_episode_index(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


def select_data_template(config_data: dict[str, Any], data_key: str | None) -> tuple[str, dict[str, Any]]:
    data_section = config_data.get("data")
    if not isinstance(data_section, dict) or not data_section:
        raise ValueError("Config must contain a non-empty 'data' section")

    if data_key is not None:
        if data_key not in data_section:
            raise ValueError(f"data_key '{data_key}' not found in config")
        return data_key, data_section[data_key]

    if len(data_section) != 1:
        raise ValueError("Config contains multiple data templates; please provide --data-key")

    selected_key = next(iter(data_section))
    return selected_key, data_section[selected_key]


def normalize_features(features: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, spec in features.items():
        normalized_spec = dict(spec)
        normalized_spec["shape"] = tuple(spec["shape"])
        normalized[key] = normalized_spec
    return normalized


def build_dataset_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dataset_features: dict[str, dict[str, Any]] = {}
    for key, spec in features.items():
        dataset_features[key] = dict(spec)
    return dataset_features


def create_episode_dataset(
    *,
    repo_id: str,
    root: Path,
    robot_type: str,
    fps: int,
    features: dict[str, dict[str, Any]],
) -> LeRobotDataset:
    root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type=robot_type,
        fps=fps,
        features=features,
        use_videos=False,
        image_writer_threads=0,
        image_writer_processes=0,
    )


def clear_dataset_episode_buffer(dataset: LeRobotDataset) -> None:
    clear_episode_buffer = getattr(dataset, "clear_episode_buffer", None)
    if callable(clear_episode_buffer):
        clear_episode_buffer()


def downsample_reference_timestamps(reference_timestamps: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError(f"frame_downsample must be >= 1, got {factor}")
    if factor == 1 or len(reference_timestamps) == 0:
        return np.ascontiguousarray(reference_timestamps)
    return np.ascontiguousarray(reference_timestamps[::factor])


def episode_done_marker_hint() -> str:
    return " or ".join(EPISODE_DONE_MARKER_NAMES)


def episode_done_marker_paths(episode_dir: Path) -> list[Path]:
    return [episode_dir / marker_name for marker_name in EPISODE_DONE_MARKER_NAMES]


def has_episode_done_marker(episode_dir: Path) -> bool:
    return any(marker_path.exists() for marker_path in episode_done_marker_paths(episode_dir))


def episode_done_marker_path(episode_dir: Path) -> Path:
    for marker_path in episode_done_marker_paths(episode_dir):
        if marker_path.exists():
            return marker_path
    return episode_dir / EPISODE_DONE_MARKER_NAME


def infer_robot_type(features: dict[str, dict[str, Any]]) -> str:
    state_size = int(np.prod(features["state"]["shape"]))
    return "single_flexiv_tdk" if state_size <= 7 else "bimanual_flexiv_tdk"


def validate_slice_range(feature_name: str, slice_range: Any) -> tuple[int, int]:
    if not isinstance(slice_range, list | tuple) or len(slice_range) != 2:
        raise ValueError(f"Feature '{feature_name}' slice must be [start, end]")
    start, end = int(slice_range[0]), int(slice_range[1])
    if start < 0 or end <= start:
        raise ValueError(f"Feature '{feature_name}' slice is invalid: {slice_range}")
    return start, end


def resolve_builder_source_aliases(feature_name: str, builder: dict[str, Any] | None) -> tuple[str, ...]:
    if builder is None:
        return (feature_name,)

    if "source" in builder:
        return (str(builder["source"]),)

    sources = builder.get("sources")
    if not isinstance(sources, list | tuple) or not sources:
        raise ValueError(f"Feature builder for '{feature_name}' must define 'source' or non-empty 'sources'")
    return tuple(str(source_name) for source_name in sources)


def validate_transform(feature_name: str, dtype: str, transform: str) -> None:
    supported = {
        "identity",
        "pose_wxyz_gripper_to_state",
        "pose_xyzw_gripper_to_state",
        "pose_wxyz_and_gripper_to_state",
        "pose_xyzw_and_gripper_to_state",
        "bimanual_pose_wxyz_and_gripper_to_state",
        "bimanual_pose_xyzw_and_gripper_to_state",
    }
    if transform not in supported:
        raise ValueError(f"Unsupported transform '{transform}' for feature '{feature_name}'")
    if dtype == "image" and transform != "identity":
        raise ValueError(f"Image feature '{feature_name}' only supports the identity transform")


def build_feature_stream_specs(
    features: dict[str, dict[str, Any]],
    feat_map: dict[str, str],
    latency_map: dict[str, float] | None = None,
    slice_map: dict[str, Any] | None = None,
    feature_builders: dict[str, Any] | None = None,
) -> dict[str, FeatureStreamSpec]:
    latency_map = latency_map or {}
    slice_map = slice_map or {}
    feature_builders = feature_builders or {}

    missing_required = [name for name in REQUIRED_FEATURES if name not in features]
    if missing_required:
        raise ValueError(f"Missing required features: {missing_required}")

    specs: dict[str, FeatureStreamSpec] = {}
    for feature_name, feature_cfg in features.items():
        dtype = str(feature_cfg["dtype"])
        shape = tuple(feature_cfg["shape"])
        if not shape:
            raise ValueError(f"Feature '{feature_name}' must define a non-empty shape")
        if dtype == "image" and (len(shape) != 3 or shape[2] != 3):
            raise ValueError(f"Feature '{feature_name}' image shape must be [H, W, 3], got {shape}")

        builder = feature_builders.get(feature_name)
        source_aliases = resolve_builder_source_aliases(feature_name, builder)
        source_keys: list[str] = []
        source_latencies: list[float] = []
        for source_alias in source_aliases:
            if source_alias not in feat_map:
                raise ValueError(f"Feature '{feature_name}' references unknown source alias '{source_alias}'")
            source_keys.append(str(feat_map[source_alias]))
            source_latencies.append(float(latency_map.get(source_alias, 0.0)))

        transform = "identity" if builder is None else str(builder.get("transform", "identity"))
        validate_transform(feature_name, dtype, transform)

        slice_range = None
        slice_value = None
        if builder is not None and "slice" in builder:
            slice_value = builder["slice"]
        elif feature_name in slice_map:
            slice_value = slice_map[feature_name]
        if slice_value is not None:
            slice_range = validate_slice_range(feature_name, slice_value)

        specs[feature_name] = FeatureStreamSpec(
            feature_name=feature_name,
            dtype=dtype,
            shape=shape,
            source_aliases=source_aliases,
            source_keys=tuple(source_keys),
            source_latencies_s=tuple(source_latencies),
            transform=transform,
            slice_range=slice_range,
        )
    return specs


def configure_actions_builder_from_state(prepared: dict[str, Any]) -> None:
    feature_builders = dict(prepared.get("feature_builders", {}))
    state_builder = feature_builders.get("state")

    if state_builder is not None:
        feature_builders["actions"] = dict(state_builder)
        prepared["feature_builders"] = feature_builders
        return

    if "state" not in prepared["feat"]:
        raise ValueError("Cannot use NEDF state as LeRobot action: no 'state' feature builder or feat mapping found")

    feature_builders["actions"] = {"source": "state"}
    prepared["feature_builders"] = feature_builders


def configure_actions_builder_from_command(prepared: dict[str, Any]) -> None:
    feature_builders = dict(prepared.get("feature_builders", {}))
    command_builder = feature_builders.get("command")

    if command_builder is not None:
        feature_builders["actions"] = dict(command_builder)
        prepared["feature_builders"] = feature_builders
        return

    if "command" not in prepared["feat"]:
        raise ValueError(
            "Cannot use NEDF command as LeRobot action: no 'command' feature builder or feat mapping found"
        )

    feature_builders["actions"] = {"source": "command"}
    prepared["feature_builders"] = feature_builders


def prepare_template(
    template: dict[str, Any],
    features: dict[str, dict[str, Any]],
    *,
    use_state_as_action: bool = False,
    use_command_as_action: bool = True,
) -> dict[str, Any]:
    prepared = dict(template)
    prepared["feat"] = dict(prepared.get("feat", {}))
    prepared["latency"] = dict(prepared.get("latency", {}))
    prepared["slices"] = dict(prepared.get("slices", {}))
    prepared["feature_builders"] = dict(prepared.get("feature_builders", {}))
    prepared["invalid_id"] = set(prepared.get("invalid_id", []))
    prepared["drop_control_flag_zero_frames"] = bool(prepared.get("drop_control_flag_zero_frames", False))

    if not prepared.get("task"):
        raise ValueError("Each dataset template must define a non-empty task")

    if use_command_as_action:
        configure_actions_builder_from_command(prepared)
    elif use_state_as_action:
        configure_actions_builder_from_state(prepared)

    build_feature_stream_specs(
        features,
        prepared["feat"],
        prepared["latency"],
        prepared["slices"],
        prepared["feature_builders"],
    )
    return prepared


def iter_unique_sources(feature_specs: dict[str, FeatureStreamSpec]) -> list[tuple[str, str, bool]]:
    unique_sources: list[tuple[str, str, bool]] = []
    seen_keys: set[str] = set()
    for spec in feature_specs.values():
        is_image = spec.dtype == "image"
        for source_alias, source_key in zip(spec.source_aliases, spec.source_keys, strict=True):
            if source_key in seen_keys:
                continue
            seen_keys.add(source_key)
            unique_sources.append((source_alias, source_key, is_image))
    return unique_sources


def resolve_lowdim_topic_candidates(source_key: str) -> tuple[str, ...]:
    topic = source_key if source_key.startswith("/") else f"/lowdim/{source_key.lower()}"

    lowercase_topic = topic.lower()
    if lowercase_topic == topic:
        return (topic,)
    return (topic, lowercase_topic)


def read_image_stream(reader: Any, source_key: str) -> tuple[np.ndarray, np.ndarray]:
    if source_key.startswith("/"):
        raise ValueError(f"Image source must be a camera id, not a topic path: {source_key}")

    images: list[np.ndarray] = []
    timestamps: list[float] = []
    for msg in reader.iter_color(source_key.lower()):
        jpg_bytes = np.frombuffer(msg.decoded_message.data, dtype=np.uint8)
        image = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        timestamps.append(msg.message.log_time / 1e9)

    if not images:
        return np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.float64)

    return np.asarray(images), np.asarray(timestamps, dtype=np.float64)


def read_lowdim_stream(reader: Any, source_key: str) -> tuple[np.ndarray, np.ndarray]:
    for topic in resolve_lowdim_topic_candidates(source_key):
        data: list[np.ndarray] = []
        timestamps: list[float] = []

        for msg in reader.get_lowdim(topic):
            sample = np.asarray(msg.decoded_message.data, dtype=np.float32)
            if sample.ndim == 0:
                sample = sample.reshape(1)
            data.append(sample)
            timestamps.append(msg.message.log_time / 1e9)

        if data:
            return np.stack(data), np.asarray(timestamps, dtype=np.float64)

    return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float64)


def load_episode_with_timestamps(
    episode_dir: Path,
    feature_specs: dict[str, FeatureStreamSpec],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"metadata.json not found: {metadata_path}")

    try:
        from nmx_nedf_api import NEDFFactory
        from nmx_nedf_api import NEDFReaderConfig
    except ImportError as exc:
        raise RuntimeError(
            "NEDF2 conversion requires the optional Flexiv NEDF reader package "
            "(`nmx_nedf_api`), which is not included in the public installation."
        ) from exc

    reader = NEDFFactory.get_reader(NEDFReaderConfig(metadata_file_path=str(metadata_path)))
    data_dict: dict[str, np.ndarray] = {}
    timestamp_dict: dict[str, np.ndarray] = {}

    source_is_image: dict[str, bool] = {}
    source_latency_s: dict[str, float] = {}
    for spec in feature_specs.values():
        is_image = spec.dtype == "image"
        for source_key, latency_s in zip(spec.source_keys, spec.source_latencies_s, strict=True):
            previous_is_image = source_is_image.get(source_key)
            if previous_is_image is not None and previous_is_image != is_image:
                raise ValueError(f"Source '{source_key}' is used by both image and lowdim features")
            source_is_image[source_key] = is_image

            previous_latency = source_latency_s.get(source_key)
            if previous_latency is not None and abs(previous_latency - latency_s) > 1e-9:
                raise ValueError(f"Conflicting latency configured for source '{source_key}'")
            source_latency_s[source_key] = latency_s

    for _, source_key, is_image in iter_unique_sources(feature_specs):
        if is_image:
            data, timestamps = read_image_stream(reader, source_key)
        else:
            data, timestamps = read_lowdim_stream(reader, source_key)

        if len(timestamps) == 0:
            continue

        data_dict[source_key] = data
        timestamp_dict[source_key] = timestamps

    if not timestamp_dict:
        return data_dict, timestamp_dict

    min_timestamp = min(ts[0] for ts in timestamp_dict.values() if len(ts) > 0)
    for source_key, timestamps in timestamp_dict.items():
        timestamp_dict[source_key] = timestamps - min_timestamp

    for source_key, latency_s in source_latency_s.items():
        if source_key in timestamp_dict:
            timestamp_dict[source_key] = timestamp_dict[source_key] - latency_s

    min_timestamp_after_latency = min(ts[0] for ts in timestamp_dict.values() if len(ts) > 0)
    if min_timestamp_after_latency < 0:
        offset = -min_timestamp_after_latency
        for source_key, timestamps in timestamp_dict.items():
            timestamp_dict[source_key] = timestamps + offset

    return data_dict, timestamp_dict


def center_crop_and_resize_image(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    target_h, target_w, target_c = target_shape
    if target_c != 3:
        raise ValueError(f"Only 3-channel images are supported, got {target_shape}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {image.shape}")

    if image.shape[:2] == (target_h, target_w):
        return np.ascontiguousarray(image)

    height, width = image.shape[:2]
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    cropped = image[top : top + crop_size, left : left + crop_size]
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def align_timestamp_to_index(
    target_timestamp: float,
    device_timestamps: np.ndarray,
    max_time_diff_s: float = 0.05,
) -> int:
    if len(device_timestamps) == 0:
        return -1

    insert_idx = np.searchsorted(device_timestamps, target_timestamp, side="left")
    left_idx = max(insert_idx - 1, 0)
    right_idx = min(insert_idx, len(device_timestamps) - 1)

    left_diff = abs(device_timestamps[left_idx] - target_timestamp)
    right_diff = abs(device_timestamps[right_idx] - target_timestamp)

    if right_diff < left_diff:
        best_idx = right_idx
        best_diff = right_diff
    else:
        best_idx = left_idx
        best_diff = left_diff

    return best_idx if best_diff <= max_time_diff_s else -1


def apply_feature_slice(array: np.ndarray, spec: FeatureStreamSpec) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if spec.slice_range is not None:
        start, end = spec.slice_range
        if end > flat.size:
            raise ValueError(f"Feature '{spec.feature_name}' slice {spec.slice_range} exceeds source size {flat.size}")
        flat = flat[start:end]

    expected_size = int(np.prod(spec.shape))
    if flat.size != expected_size:
        raise ValueError(
            f"Feature '{spec.feature_name}' resolved size {flat.size} does not match expected shape {spec.shape}"
        )

    return np.ascontiguousarray(flat.astype(np.float32, copy=False).reshape(spec.shape))


def quaternion_to_rotvec(quaternion: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"Quaternion must have 4 values, got {quat.size}")
    if order == "wxyz":
        quat_xyzw = quat[[1, 2, 3, 0]]
    elif order == "xyzw":
        quat_xyzw = quat
    else:
        raise ValueError(f"Unsupported quaternion order: {order}")
    return Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)


def convert_pose_sample_to_state(
    pose_sample: np.ndarray,
    gripper_sample: np.ndarray | None,
    quat_order: str,
) -> np.ndarray:
    pose = np.asarray(pose_sample, dtype=np.float32).reshape(-1)
    if gripper_sample is None:
        if pose.size != 8:
            raise ValueError(f"Expected pose+gripper vector with 8 values, got {pose.size}")
        pose_values = pose[:7]
        gripper = pose[7:8]
    else:
        if pose.size != 7:
            raise ValueError(f"Expected pose vector with 7 values, got {pose.size}")
        gripper = np.asarray(gripper_sample, dtype=np.float32).reshape(-1)
        if gripper.size != 1:
            raise ValueError(f"Expected gripper vector with 1 value, got {gripper.size}")
        pose_values = pose

    position = pose_values[:3]
    rotation = quaternion_to_rotvec(pose_values[3:7], quat_order)
    return np.concatenate([position, rotation, gripper.astype(np.float32, copy=False)], axis=0)


def transform_lowdim_samples(samples: list[np.ndarray], spec: FeatureStreamSpec) -> np.ndarray:
    transform = spec.transform
    if transform == "identity":
        if len(samples) != 1:
            raise ValueError(f"Identity transform for '{spec.feature_name}' expects 1 source")
        return apply_feature_slice(samples[0], spec)

    if transform == "pose_wxyz_gripper_to_state":
        if len(samples) == 1:
            result = convert_pose_sample_to_state(samples[0], None, "wxyz")
        elif len(samples) == 2:
            result = convert_pose_sample_to_state(samples[0], samples[1], "wxyz")
        else:
            raise ValueError(f"Transform '{transform}' expects [pose+gripper] or [pose, gripper]")
        return apply_feature_slice(result, spec)

    if transform == "pose_xyzw_gripper_to_state":
        if len(samples) == 1:
            result = convert_pose_sample_to_state(samples[0], None, "xyzw")
        elif len(samples) == 2:
            result = convert_pose_sample_to_state(samples[0], samples[1], "xyzw")
        else:
            raise ValueError(f"Transform '{transform}' expects [pose+gripper] or [pose, gripper]")
        return apply_feature_slice(result, spec)

    if transform == "pose_wxyz_and_gripper_to_state":
        if len(samples) != 2:
            raise ValueError(f"Transform '{transform}' expects [pose, gripper]")
        result = convert_pose_sample_to_state(samples[0], samples[1], "wxyz")
        return apply_feature_slice(result, spec)

    if transform == "pose_xyzw_and_gripper_to_state":
        if len(samples) != 2:
            raise ValueError(f"Transform '{transform}' expects [pose, gripper]")
        result = convert_pose_sample_to_state(samples[0], samples[1], "xyzw")
        return apply_feature_slice(result, spec)

    if transform in {
        "bimanual_pose_wxyz_and_gripper_to_state",
        "bimanual_pose_xyzw_and_gripper_to_state",
    }:
        if len(samples) != 4:
            raise ValueError(f"Transform '{transform}' expects [left_pose, left_gripper, right_pose, right_gripper]")
        quat_order = "wxyz" if "wxyz" in transform else "xyzw"
        left_state = convert_pose_sample_to_state(samples[0], samples[1], quat_order)
        right_state = convert_pose_sample_to_state(samples[2], samples[3], quat_order)
        result = np.concatenate([left_state, right_state], axis=0)
        return apply_feature_slice(result, spec)

    raise ValueError(f"Unsupported transform '{transform}'")


def select_reference_image_feature(
    feature_specs: dict[str, FeatureStreamSpec],
    timestamp_dict: dict[str, np.ndarray],
) -> FeatureStreamSpec:
    image_specs = [spec for spec in feature_specs.values() if spec.dtype == "image"]
    if not image_specs:
        raise ValueError("At least one image feature is required for timestamp alignment")

    available_specs = [
        spec for spec in image_specs if spec.device_key in timestamp_dict and len(timestamp_dict[spec.device_key]) > 0
    ]
    if not available_specs:
        raise ValueError("No configured image streams contain data")

    return min(available_specs, key=lambda spec: len(timestamp_dict[spec.device_key]))


def build_frame_dict_at_timestamp(
    reference_timestamp: float,
    feature_specs: dict[str, FeatureStreamSpec],
    data_dict: dict[str, np.ndarray],
    timestamp_dict: dict[str, np.ndarray],
    max_time_diff_s: float,
) -> dict[str, np.ndarray] | None:
    frame_dict: dict[str, np.ndarray] = {}

    for spec in feature_specs.values():
        aligned_samples: list[np.ndarray] = []
        for source_key in spec.source_keys:
            data = data_dict.get(source_key)
            timestamps = timestamp_dict.get(source_key)
            if data is None or timestamps is None:
                return None

            aligned_idx = align_timestamp_to_index(reference_timestamp, timestamps, max_time_diff_s)
            if aligned_idx < 0:
                return None
            aligned_samples.append(data[aligned_idx])

        if spec.dtype == "image":
            frame_dict[spec.feature_name] = center_crop_and_resize_image(
                aligned_samples[0],
                spec.shape,
            )
        else:
            frame_dict[spec.feature_name] = transform_lowdim_samples(aligned_samples, spec)

    return frame_dict


def is_control_flag_feature(feature_name: str) -> bool:
    return CONTROL_FLAG_FEATURE_NAME in feature_name.lower()


def frame_has_zero_control_flag(frame_dict: dict[str, np.ndarray]) -> bool:
    control_flag = frame_dict.get(CONTROL_FLAG_FEATURE_NAME)
    if control_flag is None:
        return False

    values = np.asarray(control_flag, dtype=np.float32).reshape(-1)
    return bool(values.size > 0 and np.isclose(values[0], 0.0))


def add_aligned_frames_to_dataset(
    dataset: LeRobotDataset,
    *,
    task: str,
    reference_timestamps: np.ndarray,
    feature_specs: dict[str, FeatureStreamSpec],
    data_dict: dict[str, np.ndarray],
    timestamp_dict: dict[str, np.ndarray],
    max_time_diff_s: float,
    drop_control_flag_zero_frames: bool = False,
) -> FrameAlignmentStats:
    if drop_control_flag_zero_frames and CONTROL_FLAG_FEATURE_NAME not in feature_specs:
        raise ValueError(
            f"'{CONTROL_FLAG_FEATURE_NAME}' feature is required when drop_control_flag_zero_frames is enabled"
        )

    aligned_frames = 0
    dropped_control_flag_zero_frames_count = 0
    saved_frames = 0

    for reference_timestamp in reference_timestamps:
        frame_dict = build_frame_dict_at_timestamp(
            reference_timestamp,
            feature_specs,
            data_dict,
            timestamp_dict,
            max_time_diff_s,
        )
        if frame_dict is None:
            continue

        aligned_frames += 1
        if drop_control_flag_zero_frames and frame_has_zero_control_flag(frame_dict):
            dropped_control_flag_zero_frames_count += 1
            continue

        frame_dict["task"] = task
        dataset.add_frame(frame_dict)
        saved_frames += 1

    return FrameAlignmentStats(
        reference_frames=len(reference_timestamps),
        aligned_frames=aligned_frames,
        dropped_control_flag_zero_frames=dropped_control_flag_zero_frames_count,
        saved_frames=saved_frames,
    )


def log_control_flag_filter_summary(
    episode_name: str,
    frame_stats: FrameAlignmentStats,
    *,
    filter_enabled: bool,
) -> None:
    if not filter_enabled:
        return

    logger.info(
        "Episode %s removed %d/%d aligned frames with control_flag==0 (%d reference frames, %d saved)",
        episode_name,
        frame_stats.dropped_control_flag_zero_frames,
        frame_stats.aligned_frames,
        frame_stats.reference_frames,
        frame_stats.saved_frames,
    )


class NEDF2EpisodeAnomalyDetector:
    def __init__(
        self,
        episode_idx: int,
        episode_path: Path,
        data_dict: dict[str, np.ndarray],
        timestamp_dict: dict[str, np.ndarray],
        feat: dict[str, str],
        config: dict[str, Any],
        feature_specs: dict[str, FeatureStreamSpec],
    ):
        self.episode_idx = episode_idx
        self.episode_path = episode_path
        self.data_dict = data_dict
        self.timestamp_dict = timestamp_dict
        self.feat = feat
        self.config = config or {}
        self.feature_specs = feature_specs
        self.anomalies: list[dict[str, Any]] = []
        self._sanitized_timestamps_cache: dict[str, tuple[np.ndarray, int]] = {}

    def _sanitize_timestamps_for_anomaly_checks(
        self,
        source_alias: str,
        source_key: str,
        timestamps: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        cached = self._sanitized_timestamps_cache.get(source_key)
        if cached is not None:
            return cached

        if len(timestamps) < 3:
            result = (timestamps, 0)
            self._sanitized_timestamps_cache[source_key] = result
            return result

        max_gap = float(self.config.get("max_timestamp_gap", 1.0))
        sanitized = timestamps
        trimmed = 0

        while len(sanitized) >= 3 and trimmed < 3:
            diffs = np.diff(sanitized)
            first_gap = float(diffs[0])
            if first_gap <= 0:
                break

            tail_diffs = diffs[1 : 1 + min(10, len(diffs) - 1)]
            if len(tail_diffs) == 0:
                break

            positive_tail_diffs = tail_diffs[tail_diffs > 0]
            if len(positive_tail_diffs) == 0:
                break

            typical_gap = float(np.median(positive_tail_diffs))
            if typical_gap <= 0:
                break

            if first_gap <= max(max_gap * 10.0, typical_gap * 20.0):
                break

            sanitized = sanitized[1:]
            trimmed += 1

        if trimmed > 0:
            self._add_anomaly(
                "warning",
                f"Source '{source_alias}' ignored {trimmed} disconnected leading timestamp sample(s)",
                {
                    "source_alias": source_alias,
                    "source_key": source_key,
                    "trimmed_samples": trimmed,
                },
            )

        result = (sanitized, trimmed)
        self._sanitized_timestamps_cache[source_key] = result
        return result

    def _add_anomaly(
        self,
        severity: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        anomaly = {
            "severity": severity,
            "message": message,
            "episode_idx": self.episode_idx,
            "episode_path": str(self.episode_path),
        }
        if details:
            anomaly["details"] = details
        self.anomalies.append(anomaly)

    def detect_missing_streams(self) -> None:
        for spec in self.feature_specs.values():
            for source_alias, source_key in zip(spec.source_aliases, spec.source_keys, strict=True):
                data = self.data_dict.get(source_key)
                timestamps = self.timestamp_dict.get(source_key)
                if data is None or timestamps is None or len(data) == 0 or len(timestamps) == 0:
                    self._add_anomaly(
                        "error",
                        f"Missing stream for source '{source_alias}' ({source_key})",
                        {"feature": spec.feature_name, "source_alias": source_alias, "source_key": source_key},
                    )

    def detect_frequency_anomalies(self) -> None:
        expected_fps = self.config.get("expected_fps", {})
        tolerance = float(self.config.get("fps_tolerance", 0.3))

        for source_alias, source_key, is_image in iter_unique_sources(self.feature_specs):
            timestamps = self.timestamp_dict.get(source_key)
            if timestamps is None or len(timestamps) < 2:
                continue
            timestamps, _ = self._sanitize_timestamps_for_anomaly_checks(source_alias, source_key, timestamps)
            if len(timestamps) < 2:
                continue

            duration = timestamps[-1] - timestamps[0]
            if duration <= 0:
                continue

            expected_range = expected_fps.get(source_alias) or expected_fps.get("image" if is_image else "lowdim")
            if not expected_range:
                continue

            actual_fps = (len(timestamps) - 1) / duration
            min_fps = float(expected_range[0]) * (1 - tolerance)
            max_fps = float(expected_range[1]) * (1 + tolerance)
            if actual_fps < min_fps or actual_fps > max_fps:
                self._add_anomaly(
                    "error",
                    f"Source '{source_alias}' frequency anomaly: {actual_fps:.1f}Hz",
                    {
                        "source_alias": source_alias,
                        "source_key": source_key,
                        "actual_fps": float(actual_fps),
                        "expected_range": list(expected_range),
                    },
                )

    def detect_length_anomalies(self) -> None:
        min_frames = int(self.config.get("min_frames", 10))

        for source_alias, source_key, _ in iter_unique_sources(self.feature_specs):
            timestamps = self.timestamp_dict.get(source_key)
            if timestamps is None:
                continue
            if len(timestamps) < min_frames:
                self._add_anomaly(
                    "error",
                    f"Source '{source_alias}' has only {len(timestamps)} frames",
                    {
                        "source_alias": source_alias,
                        "source_key": source_key,
                        "frames": len(timestamps),
                        "min_frames": min_frames,
                    },
                )

    def detect_timestamp_discontinuity(self) -> None:
        max_gap = float(self.config.get("max_timestamp_gap", 1.0))

        for source_alias, source_key, _ in iter_unique_sources(self.feature_specs):
            timestamps = self.timestamp_dict.get(source_key)
            if timestamps is None or len(timestamps) < 2:
                continue
            timestamps, _ = self._sanitize_timestamps_for_anomaly_checks(source_alias, source_key, timestamps)
            if len(timestamps) < 2:
                continue

            time_diffs = np.diff(timestamps)
            if np.any(time_diffs <= 0):
                self._add_anomaly(
                    "error",
                    f"Source '{source_alias}' has non-monotonic timestamps",
                    {
                        "source_alias": source_alias,
                        "source_key": source_key,
                        "violations": int(np.sum(time_diffs <= 0)),
                    },
                )

            max_gap_found = float(np.max(time_diffs))
            if max_gap_found > max_gap:
                self._add_anomaly(
                    "error",
                    f"Source '{source_alias}' has timestamp gap {max_gap_found:.3f}s",
                    {
                        "source_alias": source_alias,
                        "source_key": source_key,
                        "max_gap": max_gap_found,
                    },
                )

    def detect_black_images(self) -> None:
        min_brightness = float(self.config.get("min_brightness", 10))
        check_ratio = float(self.config.get("image_check_ratio", 0.1))

        for source_alias, source_key, is_image in iter_unique_sources(self.feature_specs):
            if not is_image:
                continue

            data = self.data_dict.get(source_key)
            if data is None or len(data) == 0:
                continue

            num_check = max(1, int(len(data) * check_ratio))
            sample_indices = np.linspace(0, len(data) - 1, num_check, dtype=int)
            dark_frames = 0
            for idx in sample_indices:
                gray = cv2.cvtColor(data[idx], cv2.COLOR_RGB2GRAY)
                if float(np.mean(gray)) < min_brightness:
                    dark_frames += 1

            if dark_frames > 0:
                self._add_anomaly(
                    "error",
                    f"Source '{source_alias}' contains dark/black images",
                    {
                        "source_alias": source_alias,
                        "source_key": source_key,
                        "dark_frames": dark_frames,
                        "checked_frames": int(num_check),
                    },
                )

    def detect_zero_or_static_numeric_streams(self) -> None:
        zero_threshold = float(self.config.get("zero_data_threshold", 1.0))
        frozen_ratio_threshold = float(self.config.get("frozen_data_threshold", 0.95))
        frozen_diff_threshold = float(self.config.get("frozen_frame_diff_threshold", 1e-6))

        for source_alias, source_key, is_image in iter_unique_sources(self.feature_specs):
            if is_image:
                continue
            if is_control_flag_feature(source_alias):
                continue

            data = self.data_dict.get(source_key)
            if data is None or len(data) == 0:
                continue

            flattened = np.asarray(data, dtype=np.float32).reshape(len(data), -1)
            zero_ratio = float(np.mean(np.abs(flattened) < 1e-8))
            if zero_ratio > zero_threshold:
                severity = "warning" if ("wrench" in source_alias or "gripper" in source_alias) else "error"
                self._add_anomaly(
                    severity,
                    f"Source '{source_alias}' is mostly zero",
                    {
                        "source_alias": source_alias,
                        "source_key": source_key,
                        "zero_ratio": zero_ratio,
                    },
                )

            if len(flattened) < 2:
                continue

            frozen_ratio = float(np.mean(np.all(np.abs(np.diff(flattened, axis=0)) < frozen_diff_threshold, axis=1)))
            if frozen_ratio > frozen_ratio_threshold:
                severity = "warning" if ("wrench" in source_alias or "gripper" in source_alias) else "error"
                self._add_anomaly(
                    severity,
                    f"Source '{source_alias}' is nearly static",
                    {
                        "source_alias": source_alias,
                        "source_key": source_key,
                        "frozen_ratio": frozen_ratio,
                    },
                )

    def run_all_detections(self) -> tuple[list[dict[str, Any]], bool]:
        self.detect_missing_streams()
        self.detect_frequency_anomalies()
        self.detect_length_anomalies()
        self.detect_timestamp_discontinuity()
        self.detect_black_images()
        self.detect_zero_or_static_numeric_streams()
        has_errors = any(anomaly["severity"] == "error" for anomaly in self.anomalies)
        return self.anomalies, has_errors

    def get_summary(self) -> str:
        if not self.anomalies:
            return f"Episode {self.episode_idx}: no anomalies"

        errors = sum(1 for anomaly in self.anomalies if anomaly["severity"] == "error")
        warnings = sum(1 for anomaly in self.anomalies if anomaly["severity"] == "warning")
        return f"Episode {self.episode_idx}: {errors} errors, {warnings} warnings"


def extract_episode_jobs(
    config_data: dict[str, Any],
    features: dict[str, dict[str, Any]],
    *,
    use_state_as_action: bool,
    use_command_as_action: bool,
    data_key: str | None = None,
    source_dir: Path | None = None,
) -> list[EpisodeJob]:
    data_section = config_data.get("data")
    if not isinstance(data_section, dict) or not data_section:
        raise ValueError("Config must contain a non-empty 'data' mapping")

    template_items: list[tuple[str, dict[str, Any]]]
    if data_key is not None or source_dir is not None:
        template_items = [select_data_template(config_data, data_key)]
    else:
        template_items = list(data_section.items())

    jobs: list[EpisodeJob] = []
    for dataset_root, template in template_items:
        prepared = prepare_template(
            template,
            features,
            use_state_as_action=use_state_as_action,
            use_command_as_action=use_command_as_action,
        )
        dataset_path = source_dir if source_dir is not None else Path(dataset_root)
        episode_dirs = sorted(
            [path for path in dataset_path.glob("episode_*") if path.is_dir()],
            key=parse_episode_index,
        )

        for episode_dir in episode_dirs:
            if not has_episode_done_marker(episode_dir):
                logger.info(
                    "Skipping %s: waiting for completion marker (%s)",
                    episode_dir,
                    episode_done_marker_hint(),
                )
                continue
            if parse_episode_index(episode_dir) in prepared["invalid_id"]:
                continue
            jobs.append(
                EpisodeJob(
                    episode_dir=episode_dir,
                    template=prepared,
                    task=str(prepared["task"]),
                    success=prepared.get("success"),
                )
            )

    return jobs


class Converter:
    def __init__(
        self,
        config: dict[str, Any],
        repo_id: str,
        output_path: Path,
        *,
        robot_type: str | None = None,
        fps: int = DEFAULT_FPS,
        frame_downsample: int = DEFAULT_FRAME_DOWNSAMPLE,
        enable_anomaly_detection: bool = True,
        push_to_hub: bool = False,
        max_time_diff_ms: float = 50.0,
        output_layout: str = DEFAULT_OUTPUT_LAYOUT,
        use_state_as_action: bool = False,
        use_command_as_action: bool = True,
        data_key: str | None = None,
        source_dir: Path | None = None,
    ):
        self.config = config
        self.repo_id = str(repo_id)
        self.output_path = output_path
        self.features = normalize_features(self.config["features"])
        self.dataset_features = build_dataset_features(self.features)
        self.robot_type = robot_type or infer_robot_type(self.features)
        self.fps = fps
        self.frame_downsample = frame_downsample
        self.enable_anomaly_detection = enable_anomaly_detection
        self.push_to_hub = push_to_hub
        self.max_time_diff_ms = max_time_diff_ms
        self.output_layout = output_layout
        self.use_state_as_action = use_state_as_action
        self.use_command_as_action = use_command_as_action
        self.data_key = data_key
        self.source_dir = source_dir
        self.anomaly_detection_config = self.config.get("anomaly_detection", {})

    def process_episode(
        self,
        dataset: LeRobotDataset,
        job: EpisodeJob,
    ) -> int:
        feature_specs = build_feature_stream_specs(
            self.features,
            job.template["feat"],
            job.template["latency"],
            job.template["slices"],
            job.template["feature_builders"],
        )
        data_dict, timestamp_dict = load_episode_with_timestamps(job.episode_dir, feature_specs)
        if not data_dict or not timestamp_dict:
            raise ValueError(f"No valid NEDF2 data found in {job.episode_dir}")

        if self.enable_anomaly_detection:
            detector = NEDF2EpisodeAnomalyDetector(
                episode_idx=parse_episode_index(job.episode_dir),
                episode_path=job.episode_dir,
                data_dict=data_dict,
                timestamp_dict=timestamp_dict,
                feat=job.template["feat"],
                config=self.anomaly_detection_config,
                feature_specs=feature_specs,
            )
            anomalies, has_errors = detector.run_all_detections()
            if anomalies:
                logger.info(detector.get_summary())
                for anomaly in anomalies:
                    log_fn = logger.error if anomaly["severity"] == "error" else logger.warning
                    log_fn("  %s", anomaly["message"])
            if has_errors:
                raise ValueError(f"Episode {job.episode_dir.name} failed anomaly checks")

        reference_spec = select_reference_image_feature(feature_specs, timestamp_dict)
        reference_timestamps = downsample_reference_timestamps(
            timestamp_dict[reference_spec.device_key],
            self.frame_downsample,
        )
        max_time_diff_s = self.max_time_diff_ms / 1000.0

        frame_stats = add_aligned_frames_to_dataset(
            dataset,
            task=job.task,
            reference_timestamps=reference_timestamps,
            feature_specs=feature_specs,
            data_dict=data_dict,
            timestamp_dict=timestamp_dict,
            max_time_diff_s=max_time_diff_s,
            drop_control_flag_zero_frames=job.template["drop_control_flag_zero_frames"],
        )
        log_control_flag_filter_summary(
            job.episode_dir.name,
            frame_stats,
            filter_enabled=job.template["drop_control_flag_zero_frames"],
        )

        if frame_stats.saved_frames == 0:
            raise ValueError(f"No aligned frames produced for {job.episode_dir}")

        dataset.save_episode()
        return frame_stats.saved_frames

    def run(self) -> None:
        if self.push_to_hub:
            raise ValueError("This converter does not support --push-to-hub")

        jobs = extract_episode_jobs(
            self.config,
            self.features,
            use_state_as_action=self.use_state_as_action,
            use_command_as_action=self.use_command_as_action,
            data_key=self.data_key,
            source_dir=self.source_dir,
        )
        if self.output_layout == OUTPUT_LAYOUT_MERGED:
            dataset = create_episode_dataset(
                repo_id=self.repo_id,
                root=self.output_path,
                robot_type=self.robot_type,
                fps=self.fps,
                features=self.dataset_features,
            )
            try:
                for job in tqdm(jobs, desc="Processing episodes"):
                    try:
                        saved_frames = self.process_episode(dataset, job)
                        logger.info(
                            "Episode %s converted with %d frames",
                            job.episode_dir.name,
                            saved_frames,
                        )
                    except Exception as exc:
                        clear_dataset_episode_buffer(dataset)
                        logger.error("Failed to process %s: %s", job.episode_dir, exc)
            finally:
                finalize = getattr(dataset, "finalize", None)
                if callable(finalize):
                    finalize()
            return

        for job in tqdm(jobs, desc="Processing episodes"):
            dataset_root = self.output_path / job.episode_dir.name
            if dataset_root.exists():
                shutil.rmtree(dataset_root)

            dataset = create_episode_dataset(
                repo_id=self.repo_id,
                root=dataset_root,
                robot_type=self.robot_type,
                fps=self.fps,
                features=self.dataset_features,
            )
            failed = False
            try:
                saved_frames = self.process_episode(dataset, job)
                logger.info(
                    "Episode %s converted with %d frames",
                    job.episode_dir.name,
                    saved_frames,
                )
            except Exception as exc:
                failed = True
                logger.error("Failed to process %s: %s", job.episode_dir, exc)
            finally:
                finalize = getattr(dataset, "finalize", None)
                if callable(finalize):
                    finalize()
                if failed and dataset_root.exists():
                    shutil.rmtree(dataset_root)


def main(args: argparse.Namespace) -> None:
    with args.config_path.open("r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    output_path = resolve_dataset_output_path(args.repo_name, args.output_path)
    if output_path.exists():
        shutil.rmtree(output_path)

    source_dir = Path(args.source_dir).expanduser() if args.source_dir is not None else None
    Converter(
        config=config_data,
        repo_id=str(args.repo_name),
        output_path=output_path,
        robot_type=args.robot_type,
        fps=args.fps,
        frame_downsample=args.frame_downsample,
        enable_anomaly_detection=not args.disable_anomaly_detection,
        push_to_hub=args.push_to_hub,
        max_time_diff_ms=args.max_time_diff_ms,
        output_layout=args.output_layout,
        use_state_as_action=args.use_state_as_action,
        use_command_as_action=args.use_command_as_action,
        data_key=args.data_key,
        source_dir=source_dir,
    ).run()


def cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Convert NEDF2 Flexiv TDK episodes to LeRobot dataset")
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "repo_name",
        type=Path,
        nargs="?",
        default=DEFAULT_REPO_NAME,
        help="Dataset repo id under HF_LEROBOT_HOME when --output-path is not set",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Override the source directory containing episode_* folders",
    )
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH, help="Explicit dataset output path")
    parser.add_argument(
        "--output-layout",
        type=str,
        choices=OUTPUT_LAYOUT_CHOICES,
        default=DEFAULT_OUTPUT_LAYOUT,
        help="Write one dataset per episode or append all episodes into one dataset root",
    )
    parser.add_argument("--data-key", type=str, default=None, help="Select one entry from config.data")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Dataset FPS metadata")
    parser.add_argument(
        "--frame-downsample",
        type=int,
        default=DEFAULT_FRAME_DOWNSAMPLE,
        help="Keep every Nth aligned image frame",
    )
    parser.add_argument("--robot-type", type=str, default=None, help="Override robot type")
    parser.add_argument(
        "--max-time-diff-ms",
        type=float,
        default=50.0,
        help="Maximum timestamp delta for alignment in milliseconds",
    )
    parser.add_argument(
        "--disable-anomaly-detection",
        action="store_true",
        help="Disable anomaly checks before conversion",
    )
    parser.add_argument(
        "--use-state-as-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use NEDF state to populate LeRobot actions instead of NEDF action",
    )
    parser.add_argument(
        "--use-command-as-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NEDF command to populate LeRobot actions instead of NEDF action",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push the dataset after conversion")
    main(parser.parse_args())
