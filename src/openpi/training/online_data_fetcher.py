"""
Online data fetcher for local or service-backed episode ingestion.
"""

from collections import deque
import hashlib
import logging
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import requests

from openpi.shared.archive import extract_recording
from openpi.shared.episode_schema import CONTROL_FLAG_FEATURE_NAME
from openpi.shared.episode_schema import CONTROL_FLAG_FEATURE_SPEC

INCREMENTAL_EPISODE_PREFIXES = ("episode_", "ep_")
DATASET_DONE_MARKER_NAME = ".done"


class AdaptiveSampler:
    """
    Implements SOP-style adaptive sampling between online and offline data.

    Formula: ω_on = exp(alpha·l̄_on) / (exp(alpha·l̄_on) + exp(l̄_off))
    where alpha > 1 is a boost factor to prioritize online data.
    """

    def __init__(
        self,
        window_size: int = 200,
        boost_factor: float = 1.5,
        min_online_ratio: float = 0.2,
        max_online_ratio: float = 0.8,
        initial_online_weight: float = 0.5,
    ):
        if window_size <= 0 or not np.isfinite(boost_factor) or boost_factor <= 0:
            raise ValueError("Sampler window_size and boost_factor must be positive.")
        if not 0 <= min_online_ratio <= initial_online_weight <= max_online_ratio <= 1:
            raise ValueError("Sampler ratios must satisfy 0 <= min <= initial <= max <= 1.")
        self.window_size = window_size
        self.boost_factor = boost_factor
        self.min_online_ratio = min_online_ratio
        self.max_online_ratio = max_online_ratio

        self.online_losses: deque = deque(maxlen=window_size)
        self.offline_losses: deque = deque(maxlen=window_size)

        self._online_weight = initial_online_weight

    def add_loss(self, loss: float, *, is_online: bool):
        if is_online:
            self.online_losses.append(loss)
        else:
            self.offline_losses.append(loss)

    def update_weights(self) -> float:
        if len(self.online_losses) < 10 or len(self.offline_losses) < 10:
            return self._online_weight

        l_on = np.mean(list(self.online_losses))
        l_off = np.mean(list(self.offline_losses))

        exp_on = np.exp(np.clip(self.boost_factor * l_on, -50, 50))
        exp_off = np.exp(np.clip(l_off, -50, 50))

        omega_on = exp_on / (exp_on + exp_off)
        omega_on = np.clip(omega_on, self.min_online_ratio, self.max_online_ratio)

        self._online_weight = omega_on
        return self._online_weight

    @property
    def online_weight(self) -> float:
        return self._online_weight

    @property
    def offline_weight(self) -> float:
        return 1.0 - self._online_weight

    def get_stats(self) -> dict[str, Any]:
        return {
            "online_weight": self._online_weight,
            "offline_weight": 1.0 - self._online_weight,
            "online_loss_mean": np.mean(list(self.online_losses)) if self.online_losses else 0.0,
            "offline_loss_mean": np.mean(list(self.offline_losses)) if self.offline_losses else 0.0,
            "online_loss_count": len(self.online_losses),
            "offline_loss_count": len(self.offline_losses),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "online_losses": list(self.online_losses),
            "offline_losses": list(self.offline_losses),
            "_online_weight": self._online_weight,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.online_losses = deque(state_dict["online_losses"], maxlen=self.window_size)
        self.offline_losses = deque(state_dict["offline_losses"], maxlen=self.window_size)
        self._online_weight = state_dict["_online_weight"]


class OnlineDataFetcher:
    """
    Fetches new episodes from cloud storage and/or local LeRobot datasets.

    Cloud recordings are downloaded and converted via convert_data_to_zarr.
    Local datasets are expected to be immediate subdirectories under
    ``local_lerobot_data_root`` and are loaded directly as LeRobot datasets.
    Incremental per-episode exports are only consumed after their output-side
    `.done` marker is present.

    Local LeRobot directory scanning supports online data ingestion without a
    data-cloud service.
    """

    def __init__(
        self,
        datacloud_endpoint: str,
        identifier: str,
        *,
        query_filter: dict | None = None,
        local_lerobot_data_root: str = "",
        robot_type: str = "single_iphone_flexiv",
        fps: int = 10,
        task_description: str = "",
        features: dict[str, dict] | None = None,
        # convert_data_to_zarr parameters
        use_absolute_action: bool = True,
        action_type: str = "left_arm_6DOF_gripper_width",
        temporal_downsample_ratio: int = 0,
        use_dino: bool = False,
        episode_clip_head_seconds: float = 0.0,
        episode_clip_tail_seconds: float = 0.0,
        gripper_width_bias: float = 0.0,
        gripper_width_scale: float = 1.0,
        require_control_flag: bool = False,
    ):
        if datacloud_endpoint:
            from openpi.data.dependencies import require_legacy_dependencies

            require_legacy_dependencies()
        self.datacloud_endpoint = datacloud_endpoint
        self.identifier = identifier
        self.query_filter = query_filter or {}
        self.local_lerobot_data_root = (
            Path(local_lerobot_data_root).expanduser().resolve() if local_lerobot_data_root else None
        )
        self.robot_type = robot_type
        self.fps = fps
        self.task_description = task_description
        self.features = dict(features) if features is not None else self._default_features()
        self.require_control_flag = require_control_flag
        if require_control_flag:
            self.features.setdefault(CONTROL_FLAG_FEATURE_NAME, dict(CONTROL_FLAG_FEATURE_SPEC))

        # convert_data_to_zarr parameters
        self.use_absolute_action = use_absolute_action
        # Keep the enum lookup lazy so local-only mode does not require cloud conversion deps.
        self.action_type = action_type
        self.temporal_downsample_ratio = temporal_downsample_ratio
        self.use_dino = use_dino
        self.episode_clip_head_seconds = episode_clip_head_seconds
        self.episode_clip_tail_seconds = episode_clip_tail_seconds
        self.gripper_width_bias = gripper_width_bias
        self.gripper_width_scale = gripper_width_scale

        self._fetched_uuids: set = set()
        self._processed_local_datasets: set[str] = set()

    def _default_features(self) -> dict[str, dict]:
        if self.robot_type == "single_iphone_flexiv":
            return {
                "left_wrist_img": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
                "left_wrench": {"dtype": "float32", "shape": (6,), "names": ["force_torque"]},
            }
        # bimanual
        return {
            "left_wrist_img": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "right_wrist_img": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (14,), "names": ["actions"]},
        }

    def fetch_new_episodes(self) -> list[dict[str, np.ndarray]] | None:
        """
        Fetch new episodes from all configured online data sources.

        Returns list of episode data dicts or None if no new data.
        Each episode dict contains arrays for each feature key with shape (T, ...).
        """
        episodes_data = []

        cloud_episodes = self._fetch_new_cloud_episodes()
        if cloud_episodes:
            episodes_data.extend(cloud_episodes)

        local_episodes = self._fetch_new_local_episodes()
        if local_episodes:
            episodes_data.extend(local_episodes)

        self._validate_control_flags(episodes_data)
        return episodes_data or None

    def _validate_control_flags(self, episodes_data: list[dict[str, np.ndarray]]) -> None:
        if not self.require_control_flag:
            return
        missing = [index for index, episode in enumerate(episodes_data) if CONTROL_FLAG_FEATURE_NAME not in episode]
        if missing:
            raise ValueError(
                f"Online data filtering requires {CONTROL_FLAG_FEATURE_NAME!r}; missing from episodes {missing}."
            )

    def _fetch_new_cloud_episodes(self) -> list[dict[str, np.ndarray]] | None:
        """Fetch new episodes from datacloud if cloud fetching is configured."""
        if not self.datacloud_endpoint or not self.identifier:
            return None

        try:
            list_recordings_request = {
                "identifier": self.identifier,
                "query_filter": self.query_filter,
                "limit": 10000,
                "skip": 0,
            }
            url = f"{self.datacloud_endpoint}/v1/logs"
            response = requests.post(
                url,
                json=list_recordings_request,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code != 200:
                logging.warning(f"Failed to list recordings: {response.text}")
                return None

            records = response.json().get("data", [])
            all_uuids = {record["uuid"] for record in records}
            new_uuids = all_uuids - self._fetched_uuids

            if not new_uuids:
                return None

            logging.info(f"Found {len(new_uuids)} new episodes to fetch")

            episodes_data = self._download_and_process(list(new_uuids))

            if episodes_data:
                self._fetched_uuids.update(new_uuids)

            return episodes_data

        except Exception as e:
            logging.error(f"Error fetching new episodes: {e}")
            return None

    def _fetch_new_local_episodes(self) -> list[dict[str, np.ndarray]] | None:
        """Scan a local directory for newly added LeRobot datasets."""
        if self.local_lerobot_data_root is None:
            return None
        if not self.local_lerobot_data_root.exists():
            logging.warning(f"Local LeRobot data root does not exist: {self.local_lerobot_data_root}")
            return None
        if not self.local_lerobot_data_root.is_dir():
            logging.warning(f"Local LeRobot data root is not a directory: {self.local_lerobot_data_root}")
            return None

        episodes_data: list[dict[str, np.ndarray]] = []
        # Each immediate child directory is treated as one LeRobot dataset source.
        for dataset_root in sorted(self.local_lerobot_data_root.iterdir()):
            if not dataset_root.is_dir():
                continue
            dataset_key = str(dataset_root.resolve())
            if dataset_key in self._processed_local_datasets:
                continue
            if not self._is_lerobot_dataset_dir(dataset_root):
                continue

            dataset_episodes = self._load_local_lerobot_dataset(dataset_root)
            if dataset_episodes:
                episodes_data.extend(dataset_episodes)
                self._processed_local_datasets.add(dataset_key)

        if episodes_data:
            logging.info(
                f"Loaded {len(episodes_data)} episodes from local LeRobot datasets under {self.local_lerobot_data_root}"
            )
            return episodes_data

        return None

    def _is_lerobot_dataset_dir(self, dataset_root: Path) -> bool:
        if not (dataset_root / "meta").is_dir() or not (dataset_root / "data").is_dir():
            return False
        if self._requires_completion_marker(dataset_root):
            return (dataset_root / DATASET_DONE_MARKER_NAME).is_file()
        return True

    def _requires_completion_marker(self, dataset_root: Path) -> bool:
        return dataset_root.name.startswith(INCREMENTAL_EPISODE_PREFIXES)

    def _load_local_lerobot_dataset(self, dataset_root: Path) -> list[dict[str, np.ndarray]] | None:
        """Load a local LeRobot dataset and convert it to per-episode arrays."""
        try:
            dataset = LeRobotDataset(dataset_root.name, root=dataset_root)
            if len(dataset) == 0:
                logging.warning(f"Local LeRobot dataset is empty: {dataset_root}")
                return None
            if self.require_control_flag and CONTROL_FLAG_FEATURE_NAME not in dataset.meta.features:
                raise ValueError(
                    f"Residual online dataset {dataset_root} is missing required feature {CONTROL_FLAG_FEATURE_NAME!r}."
                )

            feature_keys = [feature for feature in self.features if feature in dataset.meta.features]
            if not feature_keys:
                logging.warning(f"No matching features found in local LeRobot dataset: {dataset_root}")
                return None

            episodes: list[dict[str, np.ndarray]] = []
            current_episode_idx: int | None = None
            current_task = ""
            feature_buffers: dict[str, list[np.ndarray]] = {feature: [] for feature in feature_keys}

            for idx in range(len(dataset)):
                item = dataset[idx]
                episode_idx = int(np.asarray(item["episode_index"]).item())

                if current_episode_idx is None:
                    current_episode_idx = episode_idx
                    current_task = item.get("task", "")
                elif episode_idx != current_episode_idx:
                    episodes.append(self._finalize_local_episode(feature_buffers, current_task))
                    current_episode_idx = episode_idx
                    current_task = item.get("task", "")
                    feature_buffers = {feature: [] for feature in feature_keys}

                for feature in feature_keys:
                    feature_buffers[feature].append(np.asarray(item[feature]))

            episodes.append(self._finalize_local_episode(feature_buffers, current_task))
            logging.info(f"Loaded {len(episodes)} episodes from local LeRobot dataset {dataset_root}")
            return episodes
        except ValueError:
            raise
        except Exception as e:
            logging.error(f"Error loading local LeRobot dataset {dataset_root}: {e}")
            import traceback

            logging.error(traceback.format_exc())
            return None

    def _finalize_local_episode(self, feature_buffers: dict[str, list[np.ndarray]], task: str) -> dict[str, np.ndarray]:
        episode_data = {feature: np.stack(frames, axis=0) for feature, frames in feature_buffers.items() if frames}
        episode_data["task"] = task
        return episode_data

    def _download_and_process(self, uuids: list[str]) -> list[dict[str, np.ndarray]] | None:
        """Download and process episodes into LeRobot-compatible format using convert_data_to_zarr."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "downloaded_records.tar.lz4")

            try:
                from openpi.data.iphone import ActionType
                from openpi.data.iphone import convert_data_to_zarr

                data_request = {
                    "identifier": self.identifier,
                    "uuids": uuids,
                }
                response = requests.post(
                    f"{self.datacloud_endpoint}/v1/download_records",
                    json=data_request,
                    stream=True,
                    timeout=300,
                )

                if response.status_code != 200:
                    logging.error(f"Failed to download records: {response.text}")
                    return None

                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                server_sha256sum = response.headers.get("X-File-SHA256")
                if server_sha256sum:
                    sha256_hash = hashlib.sha256()
                    with open(filename, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(chunk)
                    file_sha256sum = sha256_hash.hexdigest()
                    if file_sha256sum != server_sha256sum:
                        logging.error("SHA256 checksum mismatch")
                        return None

                extract_dir = os.path.join(temp_dir, "downloaded_records")
                os.makedirs(extract_dir, exist_ok=True)

                import lz4.frame

                with (
                    lz4.frame.open(filename, "rb") as lz4_file,
                    tarfile.open(fileobj=lz4_file, mode="r|") as tar,
                ):
                    extract_recording(tar, extract_dir)

                # Use convert_data_to_zarr to process downloaded data
                zarr_output_dir = os.path.join(temp_dir, "zarr_output")
                zarr_path = convert_data_to_zarr(
                    input_dir=extract_dir,
                    output_dir=zarr_output_dir,
                    temporal_downsample_ratio=self.temporal_downsample_ratio,
                    use_absolute_action=self.use_absolute_action,
                    action_type=ActionType[self.action_type.upper()],
                    use_dino=self.use_dino,
                    episode_clip_head_seconds=self.episode_clip_head_seconds,
                    episode_clip_tail_seconds=self.episode_clip_tail_seconds,
                    gripper_width_bias=self.gripper_width_bias,
                    gripper_width_scale=self.gripper_width_scale,
                )

                if zarr_path and os.path.exists(zarr_path):
                    return self._load_zarr_to_episodes(zarr_path)

                return None

            except Exception as e:
                logging.error(f"Error downloading/processing episodes: {e}")
                import traceback

                logging.error(traceback.format_exc())
                return None

    def _load_zarr_to_episodes(self, zarr_path: str) -> list[dict[str, np.ndarray]] | None:
        """Load zarr data and convert to list of episode dicts in LeRobot format."""
        try:
            import zarr

            from openpi.data.iphone_lerobot import convert_episode_to_lerobot_format

            src_root = zarr.group(zarr_path)

            # Load meta data
            episode_ends = src_root["meta"]["episode_ends"][:]

            # Load data
            data = {}
            for key in src_root["data"]:
                data[key] = src_root["data"][key][:]

            # Split into episodes
            episodes = []
            episode_starts = [0, *list(episode_ends[:-1])]

            for start_idx, end_idx in zip(episode_starts, episode_ends, strict=True):
                raw_episode_data = {key: data[key][start_idx:end_idx] for key in data}

                # Convert to LeRobot format
                episode_data = convert_episode_to_lerobot_format(
                    episode_data=raw_episode_data,
                    robot_type=self.robot_type,
                    task=self.task_description,
                )
                episodes.append(episode_data)

            logging.info(f"Loaded {len(episodes)} episodes from zarr data")
            return episodes

        except Exception as e:
            logging.error(f"Error loading zarr data: {e}")
            import traceback

            logging.error(traceback.format_exc())
            return None

    @property
    def fetched_count(self) -> int:
        return len(self._fetched_uuids)
