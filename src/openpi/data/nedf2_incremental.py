"""
Incrementally convert NEDF2 Flexiv TDK episodes into LeRobot datasets.

Each source episode is processed only after a completion marker appears. `.done` is the
preferred marker and legacy `.down` is accepted for compatibility. Converted output can
either be written into one dataset directory per episode under the chosen output root or
appended into a single merged LeRobot dataset root.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

from openpi.data.nedf2 import DEFAULT_FPS
from openpi.data.nedf2 import DEFAULT_FRAME_DOWNSAMPLE
from openpi.data.nedf2 import DEFAULT_OUTPUT_LAYOUT
from openpi.data.nedf2 import OUTPUT_LAYOUT_CHOICES
from openpi.data.nedf2 import OUTPUT_LAYOUT_MERGED
from openpi.data.nedf2 import OUTPUT_LAYOUT_PER_EPISODE
from openpi.data.nedf2 import LeRobotDataset
from openpi.data.nedf2 import NEDF2EpisodeAnomalyDetector
from openpi.data.nedf2 import add_aligned_frames_to_dataset
from openpi.data.nedf2 import build_dataset_features
from openpi.data.nedf2 import build_feature_stream_specs
from openpi.data.nedf2 import clear_dataset_episode_buffer
from openpi.data.nedf2 import create_episode_dataset
from openpi.data.nedf2 import downsample_reference_timestamps
from openpi.data.nedf2 import infer_robot_type
from openpi.data.nedf2 import load_episode_with_timestamps
from openpi.data.nedf2 import log_control_flag_filter_summary
from openpi.data.nedf2 import normalize_features
from openpi.data.nedf2 import parse_episode_index
from openpi.data.nedf2 import prepare_template
from openpi.data.nedf2 import resolve_dataset_output_path
from openpi.data.nedf2 import select_reference_image_feature

logger = logging.getLogger(__name__)


STATE_VERSION = 2
DEFAULT_STATE_FILE_NAME = "nedf_incremental_state.json"
EPISODE_DONE_MARKER_NAMES = (".done", ".down")
EPISODE_DONE_MARKER_NAME = EPISODE_DONE_MARKER_NAMES[0]
OUTPUT_DATASET_DONE_MARKER_NAME = ".done"
DEFAULT_CONFIG_PATH = Path("preprocess_data/configs/book_insertion_v3_online.yaml")
DEFAULT_REPO_NAME = Path("flexiv/book_insertion_v3_100_online")
DEFAULT_SOURCE_DIR = Path("dataset_nedf2")
DEFAULT_OUTPUT_PATH = Path("dataset_lerobot")
MERGED_OUTPUT_DONE_MARKER_DIR_NAME = ".episodes_done"
MERGED_DATASET_KEY = "__merged__"


@dataclass(frozen=True)
class EpisodeSignature:
    file_count: int
    total_size_bytes: int
    max_mtime_ns: int


@dataclass(frozen=True)
class ScanDecision:
    ready: bool
    reason: str
    signature: EpisodeSignature | None


@dataclass(frozen=True)
class ConversionResult:
    saved_frames: int
    task: str
    episode_dir: Path
    stage_timings: EpisodeStageTimings


@dataclass
class EpisodeStageTimings:
    load_episode_data_s: float = 0.0
    anomaly_detection_s: float = 0.0
    align_and_package_frames_s: float = 0.0
    save_episode_s: float = 0.0
    cleanup_after_failure_s: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "load_episode_data_s": self.load_episode_data_s,
            "anomaly_detection_s": self.anomaly_detection_s,
            "align_and_package_frames_s": self.align_and_package_frames_s,
            "save_episode_s": self.save_episode_s,
            "cleanup_after_failure_s": self.cleanup_after_failure_s,
        }


class EpisodeProcessingError(RuntimeError):
    def __init__(self, message: str, stage_timings: EpisodeStageTimings) -> None:
        super().__init__(message)
        self.stage_timings = stage_timings


def format_episode_stage_timings(stage_timings: EpisodeStageTimings) -> str:
    values = [f"{name}={duration:.3f}s" for name, duration in stage_timings.to_dict().items() if duration > 0.0]
    return ", ".join(values) if values else "no timings recorded"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name.lstrip('.')}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def compute_episode_signature(episode_dir: Path) -> EpisodeSignature:
    file_count = 0
    total_size_bytes = 0
    max_mtime_ns = 0

    for file_path in sorted(episode_dir.rglob("*")):
        if not file_path.is_file():
            continue
        stat = file_path.stat()
        file_count += 1
        total_size_bytes += stat.st_size
        max_mtime_ns = max(max_mtime_ns, stat.st_mtime_ns)

    return EpisodeSignature(file_count, total_size_bytes, max_mtime_ns)


def required_episode_files(episode_dir: Path) -> list[Path]:
    return [
        episode_dir / "metadata.json",
        episode_dir / "data" / "mcap_index.json",
    ]


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


def is_episode_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path.name.startswith("episode_") or path.name.startswith("ep_"))


def is_episode_structurally_complete(episode_dir: Path) -> tuple[bool, str]:
    if not has_episode_done_marker(episode_dir):
        return False, f"waiting for completion marker ({episode_done_marker_hint()})"

    missing = [str(path.relative_to(episode_dir)) for path in required_episode_files(episode_dir) if not path.exists()]
    if missing:
        return False, f"missing required files: {', '.join(missing)}"

    if not any((episode_dir / "data").glob("*.mcap")):
        return False, "no .mcap files found"

    return True, "complete"


def signatures_match(previous_signature: dict[str, Any] | None, current_signature: EpisodeSignature) -> bool:
    if previous_signature is None:
        return False
    return previous_signature == asdict(current_signature)


def decide_episode_readiness(
    episode_dir: Path,
    previous_entry: dict[str, Any] | None,
) -> ScanDecision:
    complete, reason = is_episode_structurally_complete(episode_dir)
    if not complete:
        return ScanDecision(ready=False, reason=reason, signature=None)

    signature = compute_episode_signature(episode_dir)
    if previous_entry and previous_entry.get("status") == "converted":
        return ScanDecision(ready=False, reason="already converted", signature=signature)

    if (
        previous_entry
        and previous_entry.get("status") == "failed"
        and signatures_match(
            previous_entry.get("signature"),
            signature,
        )
    ):
        return ScanDecision(ready=False, reason="failed and unchanged", signature=signature)

    return ScanDecision(ready=True, reason="ready", signature=signature)


def read_total_episodes(info_path: Path) -> int | None:
    if not info_path.exists():
        return None
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    return int(payload.get("total_episodes", 0))


def merged_state_path(output_path: Path, state_file_name: str) -> Path:
    return output_path.parent / f".{output_path.name}.{state_file_name}"


def merged_done_marker_dir(output_path: Path) -> Path:
    return output_path.parent / f".{output_path.name}{MERGED_OUTPUT_DONE_MARKER_DIR_NAME}"


def validate_output_root(output_path: Path, state_file_name: str, output_layout: str) -> Path:
    if output_layout == OUTPUT_LAYOUT_MERGED:
        state_path = merged_state_path(output_path, state_file_name)
        if not output_path.exists():
            return state_path

        if output_path.is_file():
            raise ValueError(f"Output path must be a directory: {output_path}")

        info_path = output_path / "meta" / "info.json"
        if info_path.exists():
            return state_path

        entries = list(output_path.iterdir())
        if entries:
            raise ValueError(
                f"Output path {output_path} already contains data but is not a merged LeRobot dataset root"
            )
        return state_path

    if not output_path.exists():
        return output_path / state_file_name

    if output_path.is_file():
        raise ValueError(f"Output path must be a directory: {output_path}")

    info_path = output_path / "meta" / "info.json"
    if info_path.exists():
        raise ValueError(
            f"Output path {output_path} is a single LeRobot dataset root; "
            "incremental mode expects a parent directory for per-episode datasets"
        )

    state_path = output_path / state_file_name
    if state_path.exists():
        return state_path

    entries = list(output_path.iterdir())
    if entries and all(is_episode_dataset_dir(entry) for entry in entries):
        return state_path

    if entries:
        raise ValueError(f"Output path {output_path} already contains data but is missing {state_file_name}")

    return state_path


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


class IncrementalStateStore:
    def __init__(
        self,
        *,
        path: Path,
        repo_name: str,
        source_dir: Path,
        config_path: Path,
        data_key: str,
        output_layout: str,
        use_state_as_action: bool,
        use_command_as_action: bool,
        reset_state: bool = False,
    ) -> None:
        self.path = path
        self.payload = {
            "version": STATE_VERSION,
            "repo_name": repo_name,
            "source_dir": str(source_dir.resolve()),
            "config_path": str(config_path.resolve()),
            "data_key": data_key,
            "output_layout": output_layout,
            "use_state_as_action": use_state_as_action,
            "use_command_as_action": use_command_as_action,
            "episodes": {},
            "last_scan_at": None,
        }
        if reset_state and self.path.exists():
            if not self.path.is_file():
                raise ValueError(f"State path must be a file: {self.path}")
            logger.warning("Resetting incremental state file: %s", self.path)
            self.path.unlink()
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if loaded.get("version") != STATE_VERSION:
            raise ValueError(f"Unsupported state version in {self.path}")

        for key in ("repo_name", "source_dir", "config_path", "data_key", "output_layout"):
            if loaded.get(key) != self.payload[key]:
                raise ValueError(
                    f"State file {self.path} does not match current {key}: "
                    f"saved={loaded.get(key)!r}, current={self.payload[key]!r}. "
                    "Use --reset-state to rebuild the state file."
                )
        loaded_use_state_as_action = loaded.get("use_state_as_action", False)
        if loaded_use_state_as_action != self.payload["use_state_as_action"]:
            raise ValueError(
                f"State file {self.path} does not match current use_state_as_action: "
                f"saved={loaded_use_state_as_action!r}, current={self.payload['use_state_as_action']!r}. "
                "Use --reset-state to rebuild the state file."
            )
        loaded_use_command_as_action = loaded.get("use_command_as_action", False)
        if loaded_use_command_as_action != self.payload["use_command_as_action"]:
            raise ValueError(
                f"State file {self.path} does not match current use_command_as_action: "
                f"saved={loaded_use_command_as_action!r}, current={self.payload['use_command_as_action']!r}. "
                "Use --reset-state to rebuild the state file."
            )
        loaded["use_state_as_action"] = loaded_use_state_as_action
        loaded["use_command_as_action"] = loaded_use_command_as_action
        self.payload = loaded

    def maybe_persist(self) -> None:
        self.payload["last_scan_at"] = utc_now_iso()
        atomic_write_json(self.path, self.payload)

    def get_episode(self, episode_name: str) -> dict[str, Any] | None:
        return self.payload["episodes"].get(episode_name)

    def update_episode(self, episode_name: str, **fields: Any) -> None:
        record = dict(self.payload["episodes"].get(episode_name, {}))
        record.update(fields)
        record["updated_at"] = utc_now_iso()
        self.payload["episodes"][episode_name] = record
        self.maybe_persist()


class DatasetManager:
    def __init__(
        self,
        *,
        repo_id: str,
        output_path: Path,
        dataset_features: dict[str, Any],
        fps: int,
        robot_type: str,
        output_layout: str,
        state_file_name: str,
    ) -> None:
        self.repo_id = repo_id
        self.output_path = output_path
        self.dataset_features = dataset_features
        self.fps = fps
        self.robot_type = robot_type
        self.output_layout = output_layout
        self.state_file_name = state_file_name
        self.dataset: LeRobotDataset | None = None
        self.current_dataset_key: str | None = None

    def dataset_key_for_episode(self, episode_name: str) -> str:
        if self.output_layout == OUTPUT_LAYOUT_MERGED:
            return MERGED_DATASET_KEY
        return episode_name

    def dataset_root_for_episode(self, episode_name: str) -> Path:
        if self.output_layout == OUTPUT_LAYOUT_MERGED:
            return self.output_path
        return self.output_path / episode_name

    def done_marker_path_for_episode(self, episode_name: str) -> Path:
        if self.output_layout == OUTPUT_LAYOUT_MERGED:
            return merged_done_marker_dir(self.output_path) / f"{episode_name}.done"
        return self.dataset_root_for_episode(episode_name) / OUTPUT_DATASET_DONE_MARKER_NAME

    def _read_total_episodes(self, info_path: Path) -> int | None:
        return read_total_episodes(info_path)

    def _merged_ignored_entry_names(self) -> set[str]:
        return {self.state_file_name, MERGED_OUTPUT_DONE_MARKER_DIR_NAME}

    def _cleanup_empty_dataset_root(self, dataset_root: Path) -> None:
        if self.output_layout != OUTPUT_LAYOUT_MERGED:
            shutil.rmtree(dataset_root)
            return

        for child in dataset_root.iterdir():
            if child.name in self._merged_ignored_entry_names():
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def get_existing_episode_count(self, episode_name: str) -> int | None:
        return self._read_total_episodes(self.dataset_root_for_episode(episode_name) / "meta" / "info.json")

    def has_completion_marker(self, episode_name: str) -> bool:
        return self.done_marker_path_for_episode(episode_name).exists()

    def write_completion_marker(self, episode_name: str) -> None:
        marker_path = self.done_marker_path_for_episode(episode_name)
        atomic_write_text(marker_path, f"completed_at={utc_now_iso()}\n")

    def remove_completion_marker(self, episode_name: str) -> None:
        marker_path = self.done_marker_path_for_episode(episode_name)
        if marker_path.exists():
            marker_path.unlink()

    def remove_dataset_root(self, episode_name: str) -> None:
        if self.output_layout == OUTPUT_LAYOUT_MERGED:
            return
        dataset_root = self.dataset_root_for_episode(episode_name)
        if dataset_root.exists():
            shutil.rmtree(dataset_root)

    def ensure_open(self, episode_name: str) -> LeRobotDataset:
        dataset_key = self.dataset_key_for_episode(episode_name)
        if self.dataset is not None and self.current_dataset_key == dataset_key:
            return self.dataset

        if self.dataset is not None and self.current_dataset_key != dataset_key:
            self.finalize_and_close()

        dataset_root = self.dataset_root_for_episode(episode_name)
        info_path = dataset_root / "meta" / "info.json"

        if info_path.exists():
            total_episodes = self._read_total_episodes(info_path)
            if total_episodes and total_episodes > 0:
                self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_root)
                self.current_dataset_key = dataset_key
                return self.dataset
            self._cleanup_empty_dataset_root(dataset_root)

        dataset_root.parent.mkdir(parents=True, exist_ok=True)
        if dataset_root.exists():
            if dataset_root.is_file():
                raise ValueError(f"Output path must be a directory: {dataset_root}")
            if self.output_layout == OUTPUT_LAYOUT_MERGED:
                remaining_entries = [
                    entry for entry in dataset_root.iterdir() if entry.name not in self._merged_ignored_entry_names()
                ]
            else:
                remaining_entries = list(dataset_root.iterdir())
            if remaining_entries:
                raise ValueError(f"Output path exists but is not a clean dataset root: {dataset_root}")
            if self.output_layout != OUTPUT_LAYOUT_MERGED:
                dataset_root.rmdir()

        self.dataset = create_episode_dataset(
            repo_id=self.repo_id,
            root=dataset_root,
            robot_type=self.robot_type,
            fps=self.fps,
            features=self.dataset_features,
        )
        self.current_dataset_key = dataset_key
        return self.dataset

    def clear_episode_buffer(self) -> None:
        if self.dataset is None:
            return
        try:
            clear_dataset_episode_buffer(self.dataset)
        except Exception:
            logger.exception("Failed to clear dataset episode buffer")

    def finalize_and_close(self) -> None:
        if self.dataset is None:
            return
        finalize = getattr(self.dataset, "finalize", None)
        if callable(finalize):
            finalize()
        self.dataset = None
        self.current_dataset_key = None


class IncrementalConverter:
    def __init__(
        self,
        *,
        config_data: dict[str, Any],
        config_path: Path,
        repo_id: str,
        output_path: Path,
        source_dir: Path,
        data_key: str,
        fps: int,
        frame_downsample: int,
        robot_type: str | None,
        max_time_diff_ms: float,
        state_file_name: str,
        reset_state: bool,
        poll_interval: float,
        enable_anomaly_detection: bool,
        output_layout: str,
        use_state_as_action: bool,
        use_command_as_action: bool,
    ) -> None:
        self.config_data = config_data
        self.config_path = config_path
        self.repo_id = repo_id
        self.output_path = output_path
        self.source_dir = source_dir
        self.data_key = data_key
        self.fps = fps
        self.frame_downsample = frame_downsample
        self.max_time_diff_ms = max_time_diff_ms
        self.poll_interval = poll_interval
        self.enable_anomaly_detection = enable_anomaly_detection
        self.output_layout = output_layout
        self.use_state_as_action = use_state_as_action
        self.use_command_as_action = use_command_as_action
        self.features = normalize_features(self.config_data["features"])
        self.dataset_features = build_dataset_features(self.features)
        self.robot_type = robot_type or infer_robot_type(self.features)
        self.anomaly_detection_config = self.config_data.get("anomaly_detection", {})

        template = prepare_template(
            self.config_data["data"][self.data_key],
            self.features,
            use_state_as_action=self.use_state_as_action,
            use_command_as_action=self.use_command_as_action,
        )
        self.template = template
        self.feature_specs = build_feature_stream_specs(
            self.features,
            template["feat"],
            template["latency"],
            template["slices"],
            template["feature_builders"],
        )

        self.state_path = validate_output_root(self.output_path, state_file_name, self.output_layout)
        existing_total_episodes = read_total_episodes(self.output_path / "meta" / "info.json")
        if (
            self.output_layout == OUTPUT_LAYOUT_MERGED
            and existing_total_episodes is not None
            and existing_total_episodes > 0
        ):
            if reset_state:
                raise ValueError(
                    "Cannot use --reset-state with a non-empty merged dataset output. "
                    "Use a new output path if you need to rebuild it."
                )
            if not self.state_path.exists():
                raise ValueError(
                    "Merged output dataset already exists but the incremental state file is missing. "
                    "Cannot safely infer which source episodes were already converted."
                )
        self.state = IncrementalStateStore(
            path=self.state_path,
            repo_name=self.repo_id,
            source_dir=self.source_dir,
            config_path=self.config_path,
            data_key=self.data_key,
            output_layout=self.output_layout,
            use_state_as_action=self.use_state_as_action,
            use_command_as_action=self.use_command_as_action,
            reset_state=reset_state,
        )
        self.dataset_manager = DatasetManager(
            repo_id=self.repo_id,
            output_path=self.output_path,
            dataset_features=self.dataset_features,
            fps=self.fps,
            robot_type=self.robot_type,
            output_layout=self.output_layout,
            state_file_name=state_file_name,
        )

    def process_episode(self, episode_dir: Path, dataset: LeRobotDataset) -> ConversionResult:
        stage_timings = EpisodeStageTimings()
        task = str(self.template["task"])
        try:
            stage_started_at = time.perf_counter()
            data_dict, timestamp_dict = load_episode_with_timestamps(episode_dir, self.feature_specs)
            stage_timings.load_episode_data_s = time.perf_counter() - stage_started_at
            if not data_dict or not timestamp_dict:
                raise ValueError(f"No valid NEDF2 data found in {episode_dir}")

            if self.enable_anomaly_detection:
                stage_started_at = time.perf_counter()
                detector = NEDF2EpisodeAnomalyDetector(
                    episode_idx=parse_episode_index(episode_dir),
                    episode_path=episode_dir,
                    data_dict=data_dict,
                    timestamp_dict=timestamp_dict,
                    feat=self.template["feat"],
                    config=self.anomaly_detection_config,
                    feature_specs=self.feature_specs,
                )
                anomalies, has_errors = detector.run_all_detections()
                stage_timings.anomaly_detection_s = time.perf_counter() - stage_started_at
                if anomalies:
                    logger.info(detector.get_summary())
                    for anomaly in anomalies:
                        log_fn = logger.error if anomaly["severity"] == "error" else logger.warning
                        log_fn("  %s", anomaly["message"])
                if has_errors:
                    raise ValueError("; ".join(a["message"] for a in anomalies if a["severity"] == "error"))

            reference_spec = select_reference_image_feature(self.feature_specs, timestamp_dict)
            reference_timestamps = downsample_reference_timestamps(
                timestamp_dict[reference_spec.device_key],
                self.frame_downsample,
            )
            max_time_diff_s = self.max_time_diff_ms / 1000.0

            stage_started_at = time.perf_counter()
            frame_stats = add_aligned_frames_to_dataset(
                dataset,
                task=task,
                reference_timestamps=reference_timestamps,
                feature_specs=self.feature_specs,
                data_dict=data_dict,
                timestamp_dict=timestamp_dict,
                max_time_diff_s=max_time_diff_s,
                drop_control_flag_zero_frames=self.template["drop_control_flag_zero_frames"],
            )
            stage_timings.align_and_package_frames_s = time.perf_counter() - stage_started_at
            log_control_flag_filter_summary(
                episode_dir.name,
                frame_stats,
                filter_enabled=self.template["drop_control_flag_zero_frames"],
            )

            if frame_stats.saved_frames == 0:
                raise ValueError(f"No aligned frames produced for {episode_dir}")

            stage_started_at = time.perf_counter()
            dataset.save_episode()
            stage_timings.save_episode_s = time.perf_counter() - stage_started_at

            return ConversionResult(
                saved_frames=frame_stats.saved_frames,
                task=task,
                episode_dir=episode_dir,
                stage_timings=stage_timings,
            )
        except Exception as exc:
            raise EpisodeProcessingError(str(exc), stage_timings) from exc

    def discover_episode_dirs(self) -> list[Path]:
        if not self.source_dir.exists():
            return []
        episode_dirs = [path for path in self.source_dir.glob("episode_*") if path.is_dir()]
        return sorted(episode_dirs, key=parse_episode_index)

    def _record_deferred(self, episode_dir: Path, decision: ScanDecision) -> None:
        payload: dict[str, Any] = {"status": "deferred", "reason": decision.reason}
        if decision.signature is not None:
            payload["signature"] = asdict(decision.signature)
        self.state.update_episode(episode_dir.name, **payload)

    def _record_failure(
        self,
        episode_dir: Path,
        signature: EpisodeSignature,
        error_message: str,
        processing_duration_s: float | None,
        stage_timings: EpisodeStageTimings | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "status": "failed",
            "signature": asdict(signature),
            "error_message": error_message,
            "reason": None,
        }
        if processing_duration_s is not None:
            payload["processing_duration_seconds"] = processing_duration_s
        if stage_timings is not None:
            payload["stage_timings_seconds"] = stage_timings.to_dict()
        self.state.update_episode(episode_dir.name, **payload)

    def _record_success(
        self,
        episode_dir: Path,
        signature: EpisodeSignature,
        dataset_episode_index: int,
        saved_frames: int,
        processing_duration_s: float | None,
        stage_timings: EpisodeStageTimings | None = None,
    ) -> None:
        dataset_root = self.dataset_manager.dataset_root_for_episode(episode_dir.name)
        payload: dict[str, Any] = {
            "status": "converted",
            "signature": asdict(signature),
            "dataset_episode_index": dataset_episode_index,
            "converted_frames": saved_frames,
            "converted_at": utc_now_iso(),
            "dataset_path": str(dataset_root),
            "error_message": None,
            "reason": None,
        }
        if processing_duration_s is not None:
            payload["processing_duration_seconds"] = processing_duration_s
        if stage_timings is not None:
            payload["stage_timings_seconds"] = stage_timings.to_dict()
        self.state.update_episode(episode_dir.name, **payload)

    def process_ready_episode(self, episode_dir: Path, signature: EpisodeSignature) -> bool:
        if self.output_layout == OUTPUT_LAYOUT_PER_EPISODE:
            existing_episode_count = self.dataset_manager.get_existing_episode_count(episode_dir.name)
            if existing_episode_count is not None and existing_episode_count > 0:
                self.dataset_manager.write_completion_marker(episode_dir.name)
                self._record_success(
                    episode_dir=episode_dir,
                    signature=signature,
                    dataset_episode_index=max(0, existing_episode_count - 1),
                    saved_frames=0,
                    processing_duration_s=0.0,
                )
                return True

        self.dataset_manager.remove_completion_marker(episode_dir.name)
        dataset = self.dataset_manager.ensure_open(episode_dir.name)
        dataset_episode_index = dataset.meta.total_episodes
        started_at = time.perf_counter()

        try:
            result = self.process_episode(episode_dir, dataset)
            processing_duration_s = time.perf_counter() - started_at
            self.dataset_manager.finalize_and_close()
            self.dataset_manager.write_completion_marker(episode_dir.name)
            self._record_success(
                episode_dir=episode_dir,
                signature=signature,
                dataset_episode_index=dataset_episode_index,
                saved_frames=result.saved_frames,
                processing_duration_s=processing_duration_s,
                stage_timings=result.stage_timings,
            )
            logger.info(
                "Episode %s converted: %d frames, %s",
                episode_dir.name,
                result.saved_frames,
                format_episode_stage_timings(result.stage_timings),
            )
            return True
        except EpisodeProcessingError as exc:
            stage_timings = exc.stage_timings
            cleanup_started_at = time.perf_counter()
            self.dataset_manager.clear_episode_buffer()
            self.dataset_manager.finalize_and_close()
            self.dataset_manager.remove_completion_marker(episode_dir.name)
            self.dataset_manager.remove_dataset_root(episode_dir.name)
            stage_timings.cleanup_after_failure_s = time.perf_counter() - cleanup_started_at
            processing_duration_s = time.perf_counter() - started_at
            self._record_failure(
                episode_dir=episode_dir,
                signature=signature,
                error_message=str(exc),
                processing_duration_s=processing_duration_s,
                stage_timings=stage_timings,
            )
            logger.error(
                "Episode %s failed: %s (%s)",
                episode_dir.name,
                exc,
                format_episode_stage_timings(stage_timings),
            )
            return False

    def scan_once(self) -> None:
        discovered = self.discover_episode_dirs()
        if not discovered:
            self.state.maybe_persist()
            return

        for episode_dir in discovered:
            previous_entry = self.state.get_episode(episode_dir.name)
            if previous_entry and previous_entry.get("status") == "converted":
                existing_episode_count = self.dataset_manager.get_existing_episode_count(episode_dir.name)
                if self.output_layout == OUTPUT_LAYOUT_PER_EPISODE:
                    if existing_episode_count is None or existing_episode_count <= 0:
                        logger.warning(
                            "State marks %s converted but output dataset is missing or empty at %s; reprocessing",
                            episode_dir.name,
                            self.dataset_manager.dataset_root_for_episode(episode_dir.name),
                        )
                        previous_entry = None
                    elif not self.dataset_manager.has_completion_marker(episode_dir.name):
                        self.dataset_manager.write_completion_marker(episode_dir.name)
                else:
                    dataset_episode_index = previous_entry.get("dataset_episode_index")
                    if dataset_episode_index is None:
                        logger.warning(
                            "State marks %s converted but is missing dataset_episode_index; reprocessing",
                            episode_dir.name,
                        )
                        previous_entry = None
                    elif existing_episode_count is None or existing_episode_count <= int(dataset_episode_index):
                        logger.warning(
                            "State marks %s converted but merged output dataset at %s is missing episode index %s; reprocessing",
                            episode_dir.name,
                            self.dataset_manager.dataset_root_for_episode(episode_dir.name),
                            dataset_episode_index,
                        )
                        previous_entry = None
                    elif not self.dataset_manager.has_completion_marker(episode_dir.name):
                        self.dataset_manager.write_completion_marker(episode_dir.name)
            decision = decide_episode_readiness(episode_dir, previous_entry)
            if not decision.ready:
                if previous_entry and decision.reason in {"already converted", "failed and unchanged"}:
                    continue
                self._record_deferred(episode_dir, decision)
                continue
            if decision.signature is None:
                raise ValueError(f"Ready episode {episode_dir} is missing a signature")
            self.process_ready_episode(episode_dir, decision.signature)

        self.state.maybe_persist()

    def run(self) -> None:
        logger.info(
            "Watching %s for completed episodes. Output root: %s",
            self.source_dir,
            self.output_path,
        )
        try:
            while True:
                self.scan_once()
                time.sleep(self.poll_interval)
        finally:
            self.dataset_manager.finalize_and_close()


def main(args: argparse.Namespace) -> None:
    with args.config_path.open("r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    data_key, _ = select_data_template(config_data, args.data_key)
    output_path = resolve_dataset_output_path(args.repo_name, args.output_path)

    converter = IncrementalConverter(
        config_data=config_data,
        config_path=args.config_path,
        repo_id=str(args.repo_name),
        output_path=output_path,
        source_dir=Path(args.source_dir).expanduser(),
        data_key=data_key,
        fps=args.fps,
        frame_downsample=args.frame_downsample,
        robot_type=args.robot_type,
        max_time_diff_ms=args.max_time_diff_ms,
        state_file_name=args.state_file_name,
        reset_state=args.reset_state,
        poll_interval=args.poll_interval,
        enable_anomaly_detection=not args.disable_anomaly_detection,
        output_layout=args.output_layout,
        use_state_as_action=args.use_state_as_action,
        use_command_as_action=args.use_command_as_action,
    )
    converter.run()


def cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Incrementally convert NEDF2 Flexiv TDK episodes to LeRobot datasets")
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
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing episode_* folders",
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
        "--state-file",
        "--state-file-name",
        dest="state_file_name",
        default=DEFAULT_STATE_FILE_NAME,
        help="Filename used to track incremental conversion state",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Discard the existing incremental state file and rebuild it for the current source/output",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between directory scans",
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
    main(parser.parse_args())
