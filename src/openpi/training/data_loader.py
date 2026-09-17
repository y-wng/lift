from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import shutil
import threading
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_delta_indices
from lerobot.common.datasets.utils import get_episode_data_index
import numpy as np
import torch

import openpi.models.model as _model
from openpi.shared.episode_schema import CONTROL_FLAG_FEATURE_NAME
from openpi.shared.episode_schema import CONTROL_FLAG_FEATURE_SPEC
from openpi.shared.episode_schema import DEFAULT_INTERVENTION_VALUE
from openpi.shared.episode_schema import is_intervention_chunk
import openpi.training.config as _config
from openpi.training.online_data_fetcher import AdaptiveSampler
import openpi.training.residual_target_cache as _residual_targets
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    available_features = set(dataset_meta.features)
    action_sequence_keys = tuple(key for key in data_config.action_sequence_keys if key in available_features)
    if action_sequence_keys != tuple(data_config.action_sequence_keys):
        missing_keys = tuple(key for key in data_config.action_sequence_keys if key not in available_features)
        logging.info(
            "Dataset %s does not contain optional sequence keys %s; loading the available keys only.",
            repo_id,
            missing_keys,
        )
    if "actions" not in action_sequence_keys:
        raise ValueError(f"Dataset {repo_id} does not contain the required 'actions' feature.")
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={key: [t / dataset_meta.fps for t in range(action_horizon)] for key in action_sequence_keys},
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    action_horizon = (
        config.model.async_action_horizon
        if getattr(config.model, "async_action_horizon", -1) > 0
        else config.model.action_horizon
    )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]


# ============================================================================
# HybridDataset and HybridDataLoader for Online DAgger Training
# ============================================================================


class HybridDataset(Dataset):
    """
    A dataset that combines offline and online LeRobot datasets with adaptive sampling.

    Features:
    - Loads an existing offline LeRobot dataset
    - Dynamically maintains an online LeRobot dataset
    - Integrates AdaptiveSampler for loss-based sampling weight adjustment
    - Provides append_episodes() interface for adding new data
    - Supports sampling by source (online/offline)
    """

    def __init__(
        self,
        offline_repo_id: str,
        online_repo_id: str,
        action_horizon: int,
        *,
        action_sequence_keys: Sequence[str] = ("actions",),
        online_action_sequence_keys: Sequence[str] = ("actions",),
        prompt_from_task: bool = False,
        # Adaptive sampling parameters
        window_size: int = 200,
        boost_factor: float = 1.5,
        min_online_ratio: float = 0.2,
        max_online_ratio: float = 0.8,
        initial_online_weight: float = 0.5,
        # Online dataset creation parameters
        robot_type: str = "single_iphone_flexiv",
        fps: int = 10,
        features: dict | None = None,
        online_intervention_only: bool = False,
        intervention_value: float = DEFAULT_INTERVENTION_VALUE,
        allow_offline_warm_start: bool = True,
        residual_target_builder: _residual_targets.ResidualTargetBuilder | None = None,
        # Deprecated compatibility arguments. Residual targets are no longer cached.
        residual_model_state: typing.Any | None = None,
        residual_cache_key: str = "",
        residual_cache_dir: str | None = None,
        seed: int = 0,
    ):
        del residual_model_state, residual_cache_key, residual_cache_dir
        if not online_repo_id.strip():
            raise ValueError("online_repo_id must be non-empty; use a fresh repo id for each run.")
        if online_repo_id == offline_repo_id:
            raise ValueError(
                "online_repo_id must differ from offline_repo_id because the online dataset is recreated at startup."
            )
        self._offline_repo_id = offline_repo_id
        self._online_repo_id = online_repo_id
        self._action_horizon = action_horizon
        self._action_sequence_keys = ("actions",)
        self._prompt_from_task = prompt_from_task
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # Load offline dataset
        offline_meta = LeRobotDatasetMetadata(offline_repo_id)
        offline_action_sequence_keys = tuple(key for key in action_sequence_keys if "wrench" not in key)
        if not offline_action_sequence_keys:
            offline_action_sequence_keys = ("actions",)
        if offline_action_sequence_keys != tuple(action_sequence_keys):
            logging.info("Skipping wrench sequence keys for offline hybrid dataset reads")
        self._offline_dataset = LeRobotDataset(
            offline_repo_id,
            delta_timestamps={
                key: [t / offline_meta.fps for t in range(action_horizon)] for key in offline_action_sequence_keys
            },
        )
        self._offline_tasks = offline_meta.tasks if prompt_from_task else None

        # Store features from offline dataset or use provided features
        if features is None:
            features = self._get_features_from_offline_dataset()
        self._residual_target_builder = residual_target_builder
        self._uses_control_flag = online_intervention_only or residual_target_builder is not None
        if self._uses_control_flag and CONTROL_FLAG_FEATURE_NAME not in features:
            features = dict(features)
            features[CONTROL_FLAG_FEATURE_NAME] = dict(CONTROL_FLAG_FEATURE_SPEC)
        self._features = features
        self._robot_type = robot_type
        self._fps = fps
        self._online_intervention_only = online_intervention_only
        self._intervention_value = intervention_value
        self._allow_offline_warm_start = allow_offline_warm_start
        self._online_sample_remainder = 0.0
        self._valid_online_indices: list[int] = []

        # Build delta_timestamps for online dataset (same as offline)
        self._delta_timestamps = {key: [t / fps for t in range(action_horizon)] for key in online_action_sequence_keys}

        # Delete existing online dataset if it exists (online dataset should be built fresh during training)
        online_path = HF_LEROBOT_HOME / online_repo_id
        if online_path.exists():
            logging.info(f"Removing existing online dataset at {online_path}")
            shutil.rmtree(online_path)

        # Create online dataset (initially empty)
        self._online_dataset = LeRobotDataset.create(
            repo_id=online_repo_id,
            robot_type=robot_type,
            fps=fps,
            features=features,
            image_writer_threads=4,
            image_writer_processes=2,
        )

        # Initialize adaptive sampler
        self._adaptive_sampler = AdaptiveSampler(
            window_size=window_size,
            boost_factor=boost_factor,
            min_online_ratio=min_online_ratio,
            max_online_ratio=max_online_ratio,
            initial_online_weight=initial_online_weight,
        )

        self._online_episodes_count = 0
        self._online_residual_targets: list[np.ndarray] | None = None
        if residual_target_builder is not None:
            self._online_residual_targets = []
            self._zero_residual_target = np.zeros(residual_target_builder.target_shape, dtype=np.float32)

    def _get_features_from_offline_dataset(self) -> dict:
        """Extract features configuration from offline dataset metadata."""
        # Default features based on common patterns
        return {
            "left_wrist_img": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "left_wrench": {"dtype": "float32", "shape": (6,), "names": ["force_torque"]},
        }

    @staticmethod
    def _drop_wrench_keys(item: dict) -> dict:
        for key in ("left_wrench", "left_wrench_is_pad", "wrench", "wrench_is_pad"):
            item.pop(key, None)
        return item

    def _get_offline_item(self, idx: int) -> dict:
        item = self._drop_wrench_keys(self._offline_dataset[idx])
        if self._prompt_from_task and self._offline_tasks:
            item = _transforms.PromptFromLeRobotTask(self._offline_tasks)(item)
        return item

    def _get_online_item_by_actual_index(self, idx: int) -> dict:
        item = self._online_dataset[idx]
        if self._prompt_from_task and self._offline_tasks:
            item = _transforms.PromptFromLeRobotTask(self._offline_tasks)(item)
        return item

    def append_episodes(
        self,
        episodes_data: list[dict[str, np.ndarray]],
        task: str = "",
        *,
        residual_model_state: typing.Any | None = None,
    ) -> None:
        with self._lock:
            self._append_episodes(
                episodes_data,
                task=task,
                residual_model_state=residual_model_state,
            )

    def _append_episodes(
        self,
        episodes_data: list[dict[str, np.ndarray]],
        task: str = "",
        *,
        residual_model_state: typing.Any | None = None,
    ) -> None:
        """
        Append new episodes to the online dataset.

        Args:
            episodes_data: List of episode data dicts, each containing
                          arrays for each key with shape (T, ...)
            task: Task description for these episodes
        """
        old_online_frame_count = len(self._online_dataset)
        for episode_data in episodes_data:
            if self._residual_target_builder is not None and CONTROL_FLAG_FEATURE_NAME not in episode_data:
                raise ValueError(f"Residual online episode is missing required feature {CONTROL_FLAG_FEATURE_NAME!r}.")
            num_frames = None
            for key in self._features:
                if key in episode_data:
                    if num_frames is None:
                        num_frames = episode_data[key].shape[0]
                    break

            if num_frames is None:
                logging.warning("No valid data found in episode, skipping")
                continue
            if self._residual_target_builder is not None:
                control_flag_frames = np.asarray(episode_data[CONTROL_FLAG_FEATURE_NAME])
                if control_flag_frames.shape[0] != num_frames:
                    raise ValueError(
                        f"Online control_flag has {control_flag_frames.shape[0]} frames; expected {num_frames}."
                    )

            for step in range(num_frames):
                frame_dict = {}
                for feat in self._features:
                    if feat in episode_data:
                        frame_dict[feat] = self._prepare_feature_value_for_storage(feat, episode_data[feat][step])
                frame_dict["task"] = task or episode_data.get("task", "")
                self._online_dataset.add_frame(frame_dict)

            self._online_dataset.save_episode()
            self._online_episodes_count += 1

        # Update delta_timestamps, delta_indices and episode_data_index for online dataset
        self._online_dataset.delta_timestamps = self._delta_timestamps
        self._online_dataset.delta_indices = get_delta_indices(self._delta_timestamps, self._fps)
        self._online_dataset.episode_data_index = get_episode_data_index(
            self._online_dataset.meta.episodes, self._online_dataset.episodes
        )

        new_online_frame_count = len(self._online_dataset)
        if self._residual_target_builder is not None and new_online_frame_count > old_online_frame_count:
            if residual_model_state is None:
                raise ValueError(
                    "residual_model_state is required when appending episodes to a residual target dataset."
                )
            if self._online_residual_targets is None or len(self._online_residual_targets) != old_online_frame_count:
                raise RuntimeError("Online residual target cache is not aligned with the online dataset.")
            new_targets = self._residual_target_builder.build_online(
                residual_model_state,
                self._get_online_item_by_actual_index,
                range(old_online_frame_count, new_online_frame_count),
                source_name=f"online:{self._online_repo_id}",
            )
            self._online_residual_targets.extend(new_targets)

        self._refresh_valid_online_indices()

        logging.info(
            "Online dataset updated: %d episodes, %d samples",
            self._online_episodes_count,
            len(self._online_dataset),
        )

    @property
    def lock(self) -> threading.RLock:
        """Serialize batch reads with online dataset appends."""

        return self._lock

    def _prepare_feature_value_for_storage(self, feature_name: str, value: np.ndarray) -> np.ndarray:
        """Normalize feature layout before writing a new online frame.

        Local LeRobot datasets are read back as channel-first tensors by `LeRobotDataset.__getitem__`,
        but `LeRobotDataset.add_frame` expects raw image arrays to follow the feature metadata layout.
        For the common `(height, width, channel)` metadata used in our datasets, convert `CHW -> HWC`
        before appending to the online dataset.
        """
        array = np.asarray(value)
        feature = self._features.get(feature_name, {})
        expected_shape = tuple(feature.get("shape", ()))
        if (
            expected_shape
            and feature.get("dtype") not in {"image", "video"}
            and array.shape != expected_shape
            and array.size == int(np.prod(expected_shape))
        ):
            return np.asarray(array, dtype=array.dtype).reshape(expected_shape)

        if feature.get("dtype") not in {"image", "video"} or array.ndim != 3:
            return array

        feature_names = tuple(feature.get("names", ()))
        if len(expected_shape) != 3 or len(feature_names) != 3:
            return array

        if feature_names[-1] in {"channel", "channels"}:
            expected_chw = (expected_shape[2], expected_shape[0], expected_shape[1])
            if array.shape == expected_chw:
                return np.ascontiguousarray(np.transpose(array, (1, 2, 0)))
        elif feature_names[0] in {"channel", "channels"}:
            expected_hwc = (expected_shape[1], expected_shape[2], expected_shape[0])
            if array.shape == expected_hwc:
                return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))

        return array

    def _online_item_has_valid_control_flag_chunk(self, item: dict) -> bool:
        if not self._online_intervention_only:
            return True
        control_flag = item.get(CONTROL_FLAG_FEATURE_NAME)
        if control_flag is None:
            return False

        values = np.asarray(control_flag, dtype=np.float32).reshape(-1)
        return is_intervention_chunk(values, self._intervention_value)

    def _refresh_valid_online_indices(self) -> None:
        if not self._online_intervention_only:
            self._valid_online_indices = list(range(len(self._online_dataset)))
            return

        valid_indices = []
        for idx in range(len(self._online_dataset)):
            item = self._online_dataset[idx]
            if self._online_item_has_valid_control_flag_chunk(item):
                valid_indices.append(idx)
        self._valid_online_indices = valid_indices
        logging.info(
            "Online control_flag filter kept %d/%d chunks with control_flag==-1",
            len(self._valid_online_indices),
            len(self._online_dataset),
        )

    @staticmethod
    def _drop_control_flag_key(item: dict) -> dict:
        item.pop(CONTROL_FLAG_FEATURE_NAME, None)
        return item

    def _get_filtered_online_item(self, idx: int) -> dict:
        if self.online_len == 0:
            raise IndexError("Online dataset is empty.")

        actual_idx = self._resolve_online_index(idx)
        item = self._get_online_item_by_actual_index(actual_idx)
        if self._uses_control_flag:
            item = self._drop_control_flag_key(item)
        return item

    def _resolve_online_index(self, idx: int) -> int:
        if self._online_intervention_only:
            return self._valid_online_indices[idx % self.online_len]
        return idx % len(self._online_dataset)

    def get_residual_target(self, *, is_online: bool, idx: int) -> np.ndarray:
        if self._residual_target_builder is None:
            raise RuntimeError("Residual targets are not configured for this dataset.")
        if is_online:
            if self.online_len == 0 or self._online_residual_targets is None:
                raise IndexError("Online residual targets are not available.")
            return self._online_residual_targets[self._resolve_online_index(idx)]
        return self._zero_residual_target

    @property
    def has_residual_targets(self) -> bool:
        return self._residual_target_builder is not None

    @property
    def offline_len(self) -> int:
        return len(self._offline_dataset)

    @property
    def online_len(self) -> int:
        return len(self._valid_online_indices) if self._online_intervention_only else len(self._online_dataset)

    def __len__(self) -> int:
        return self.offline_len + self.online_len

    def __getitem__(self, index: SupportsIndex) -> dict:
        """Standard getitem - uses unified indexing across both datasets."""
        idx = index.__index__()
        if idx < self.offline_len:
            item = self._get_offline_item(idx)
        else:
            online_idx = idx - self.offline_len
            if online_idx < self.online_len:
                item = self._get_filtered_online_item(online_idx)
            else:
                item = self._get_offline_item(idx % self.offline_len)

        return item

    def get_item_by_source(self, *, is_online: bool, idx: int) -> dict:
        """Get item from specific source (online or offline)."""
        if is_online and self.online_len > 0:
            item = self._get_filtered_online_item(idx)
        else:
            item = self._get_offline_item(idx % self.offline_len)

        return item

    def sample_batch_indices(self, batch_size: int, *, replace: bool = False) -> list[tuple[bool, int]]:
        """
        Sample batch indices using adaptive sampling weights.

        Returns list of (is_online, idx) tuples.
        """
        if self.online_len == 0:
            if not self._allow_offline_warm_start:
                raise ValueError("No eligible online samples; this run disables offline warm starts.")
            # Only offline data available
            if not replace and batch_size <= self.offline_len:
                indices = self._rng.choice(self.offline_len, size=batch_size, replace=False)
            else:
                indices = self._rng.integers(0, self.offline_len, size=batch_size)
            return [(False, int(idx)) for idx in indices]

        online_weight = self._adaptive_sampler.online_weight

        online_quota = batch_size * online_weight + self._online_sample_remainder
        n_online = min(batch_size, round(online_quota))
        self._online_sample_remainder = online_quota - n_online
        n_offline = batch_size - n_online

        # Sample online indices
        if n_online > 0:
            if not replace and n_online <= self.online_len:
                online_indices = self._rng.choice(self.online_len, size=n_online, replace=False)
            else:
                online_indices = self._rng.integers(0, self.online_len, size=n_online)
        else:
            online_indices = np.array([], dtype=np.int64)

        # Sample offline indices
        if n_offline > 0:
            if not replace and n_offline <= self.offline_len:
                offline_indices = self._rng.choice(self.offline_len, size=n_offline, replace=False)
            else:
                offline_indices = self._rng.integers(0, self.offline_len, size=n_offline)
        else:
            offline_indices = np.array([], dtype=np.int64)

        batch = [(True, int(idx)) for idx in online_indices] + [(False, int(idx)) for idx in offline_indices]
        self._rng.shuffle(batch)
        return batch

    @property
    def adaptive_sampler(self):
        return self._adaptive_sampler

    @property
    def online_episodes_count(self) -> int:
        return self._online_episodes_count

    @property
    def offline_episodes_count(self) -> int:
        return self._offline_dataset.meta.total_episodes

    def get_sampling_stats(self) -> dict:
        return self._adaptive_sampler.get_stats()


class HybridDataLoader:
    """
    Data loader for HybridDataset that supports adaptive sampling between online and offline data.
    """

    def __init__(
        self,
        dataset: HybridDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        seed: int = 0,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._local_batch_size = local_batch_size
        self._num_batches = num_batches
        self._rng = np.random.default_rng(seed)

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding

    def __iter__(self):
        num_items = 0
        while True:
            if self._num_batches is not None and num_items >= self._num_batches:
                return

            # Keep reads and online appends mutually exclusive. The producer thread
            # may be preparing a batch while the training loop ingests new episodes.
            with self._dataset.lock:
                # Sample batch indices using adaptive sampling
                batch_indices = self._dataset.sample_batch_indices(self._local_batch_size, replace=False)

                # Collect batch data and track sources
                batch_data = []
                batch_is_online = []
                wrench_mask = []
                for is_online, idx in batch_indices:
                    item = self._dataset.get_item_by_source(is_online=is_online, idx=idx)
                    item = self._transform(item)
                    if self._dataset.has_residual_targets:
                        # Residual targets are already in the post-transform normalized action space.
                        item["actions"] = np.asarray(
                            self._dataset.get_residual_target(is_online=is_online, idx=idx), dtype=np.float32
                        )
                    if item.get("wrench") is None:
                        item["wrench"] = np.zeros((item["actions"].shape[0], 6), dtype=np.float32)
                        wrench_mask.append(False)
                    else:
                        wrench_mask.append(True)
                    batch_data.append(item)
                    batch_is_online.append(is_online)

                # Stack batch
                batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *batch_data)
                batch["_is_online"] = np.array(batch_is_online, dtype=np.bool_)
                batch["wrench_mask"] = np.array(wrench_mask, dtype=np.bool_)
                batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
            num_items += 1
            yield batch

    @property
    def dataset(self) -> HybridDataset:
        return self._dataset


class HybridDataLoaderImpl(DataLoader):
    """Wrapper for HybridDataLoader that implements the DataLoader protocol."""

    def __init__(self, data_config: _config.DataConfig, data_loader: HybridDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    @property
    def dataset(self) -> HybridDataset:
        return self._data_loader.dataset

    def __iter__(self):
        for batch in self._data_loader:
            is_online = batch.pop("_is_online", None)
            observation = _model.Observation.from_dict(batch)
            actions = batch["actions"]
            if is_online is not None:
                yield observation, actions, is_online
            else:
                yield observation, actions


def create_hybrid_data_loader(
    config: _config.TrainConfig,
    online_repo_id: str,
    *,
    sharding: jax.sharding.Sharding | None = None,
    num_batches: int | None = None,
    # Adaptive sampling parameters
    window_size: int = 200,
    boost_factor: float = 1.5,
    min_online_ratio: float = 0.2,
    max_online_ratio: float = 0.8,
    initial_online_weight: float = 0.5,
    # Online dataset parameters
    robot_type: str = "single_iphone_flexiv",
    fps: int = 10,
    features: dict | None = None,
    online_intervention_only: bool = False,
    intervention_value: float = DEFAULT_INTERVENTION_VALUE,
    allow_offline_warm_start: bool = True,
    residual_model_def: typing.Any | None = None,
    residual_model_state: typing.Any | None = None,
    residual_mesh: jax.sharding.Mesh | None = None,
    # Deprecated compatibility arguments. Residual targets are generated only for new online episodes.
    residual_target_cache_key: str = "",
    residual_target_cache_dir: str | None = None,
    residual_target_num_steps: int = 10,
    residual_target_batch_size: int | None = None,
) -> HybridDataLoaderImpl:
    """
    Create a hybrid data loader for online DAgger training.

    Args:
        config: The training configuration (contains offline repo_id).
        online_repo_id: The repo ID for the online dataset.
        sharding: The sharding to use for the data loader.
        num_batches: Number of batches to return.
        window_size: Window size for adaptive sampling.
        boost_factor: Boost factor for online data.
        min_online_ratio: Minimum online sampling ratio.
        max_online_ratio: Maximum online sampling ratio.
        initial_online_weight: Initial online sampling weight.
        robot_type: Robot type for online dataset.
        fps: FPS for online dataset.
        features: Features configuration for online dataset.
    """
    del residual_target_cache_key, residual_target_cache_dir
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"Creating hybrid data loader with offline repo: {data_config.repo_id}, online repo: {online_repo_id}")

    action_horizon = (
        config.model.async_action_horizon
        if getattr(config.model, "async_action_horizon", -1) > 0
        else config.model.action_horizon
    )

    # Create hybrid dataset
    online_action_sequence_keys = ["actions"]
    if config.online_use_wrench:
        online_action_sequence_keys.append("left_wrench")
    residual_policy_in_use = getattr(config.model, "residual_policy_in_use", False)
    if residual_policy_in_use:
        if online_intervention_only:
            logging.warning("Ignoring online control_flag chunk filtering for residual training.")
        online_intervention_only = False
    if online_intervention_only or residual_policy_in_use:
        online_action_sequence_keys.append(CONTROL_FLAG_FEATURE_NAME)

    # The base policy still receives its original normalized observation/action coordinates. The residual target
    # builder applies its separate static scale only to the action difference; image and wrench are not part of it.
    norm_stats = {}
    if data_config.repo_id != "fake":
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    if residual_policy_in_use:
        norm_stats = _transforms.residual_policy_norm_stats(norm_stats)

    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    residual_target_builder = None
    if residual_policy_in_use:
        if residual_model_def is None or residual_model_state is None:
            raise ValueError("Residual target generation requires residual_model_def and residual_model_state.")
        rollout = _residual_targets.BaseActionRollout(
            residual_model_def,
            mesh=residual_mesh,
            input_sharding=sharding,
            num_steps=residual_target_num_steps,
            seed=config.seed,
        )
        residual_target_builder = _residual_targets.ResidualTargetBuilder(
            _transforms.compose(transforms),
            rollout,
            batch_size=residual_target_batch_size or config.batch_size,
            action_horizon=action_horizon,
            action_dim=config.model.action_dim,
            action_scale=config.model.residual_action_scale,
            intervention_value=config.model.residual_intervention_value,
        )

    dataset = HybridDataset(
        offline_repo_id=data_config.repo_id,
        online_repo_id=online_repo_id,
        action_horizon=action_horizon,
        action_sequence_keys=data_config.action_sequence_keys,
        online_action_sequence_keys=tuple(online_action_sequence_keys),
        prompt_from_task=data_config.prompt_from_task,
        window_size=window_size,
        boost_factor=boost_factor,
        min_online_ratio=min_online_ratio,
        max_online_ratio=max_online_ratio,
        initial_online_weight=initial_online_weight,
        robot_type=robot_type,
        fps=fps,
        features=features,
        online_intervention_only=online_intervention_only,
        intervention_value=intervention_value,
        allow_offline_warm_start=allow_offline_warm_start,
        residual_target_builder=residual_target_builder,
        seed=config.seed,
    )

    local_batch_size = config.batch_size // jax.process_count()
    logging.info(f"Hybrid data loader local_batch_size: {local_batch_size}")

    data_loader = HybridDataLoader(
        dataset=dataset,
        transforms=transforms,
        local_batch_size=local_batch_size,
        sharding=sharding,
        num_batches=num_batches,
        seed=config.seed,
    )

    return HybridDataLoaderImpl(data_config, data_loader)
