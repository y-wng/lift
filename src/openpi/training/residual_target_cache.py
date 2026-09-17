from collections.abc import Callable, Sequence
import contextlib
import logging
import time
from typing import Protocol

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.shared.episode_schema import CONTROL_FLAG_FEATURE_NAME
from openpi.shared.episode_schema import DEFAULT_INTERVENTION_VALUE
from openpi.shared.episode_schema import intervention_mask as select_intervention_steps
import openpi.training.sharding as _sharding


class BaseActionRolloutFn(Protocol):
    def __call__(self, model_state: nnx.State, observation: _model.Observation) -> np.ndarray: ...


class BaseActionRollout:
    """Jitted base-policy sampler that borrows, but does not retain, model parameters."""

    def __init__(
        self,
        model_def: nnx.GraphDef,
        *,
        mesh: jax.sharding.Mesh | None,
        input_sharding: jax.sharding.Sharding | None,
        num_steps: int,
        seed: int,
    ):
        self._mesh = mesh
        self._input_sharding = input_sharding
        self._num_steps = num_steps
        self._rng = jax.random.key(seed)

        def sample(model_state: nnx.State, rng: jax.Array, observation: _model.Observation):
            model = nnx.merge(model_def, model_state)
            if not hasattr(model, "sample_base_actions"):
                raise TypeError("Residual target generation requires a model with sample_base_actions().")
            return model.sample_base_actions(rng, observation, num_steps=self._num_steps)

        self._sample = jax.jit(sample)

    def __call__(self, model_state: nnx.State, observation: _model.Observation) -> np.ndarray:
        if self._input_sharding is None:
            observation = jax.tree.map(jnp.asarray, observation)
        else:
            observation = jax.tree.map(
                lambda value: jax.make_array_from_process_local_data(self._input_sharding, value),
                observation,
            )

        self._rng, sample_rng = jax.random.split(self._rng)
        mesh_context = _sharding.set_mesh(self._mesh) if self._mesh is not None else contextlib.nullcontext()
        with mesh_context:
            actions = self._sample(model_state, sample_rng, observation)
        return np.asarray(jax.device_get(actions), dtype=np.float32)


class ResidualTargetBuilder:
    """Build normalized online residual targets at human-intervention timesteps."""

    def __init__(
        self,
        transform: Callable[[dict], dict],
        rollout: BaseActionRolloutFn,
        *,
        batch_size: int,
        action_horizon: int,
        action_dim: int,
        action_scale: Sequence[float] | None = None,
        intervention_value: float = DEFAULT_INTERVENTION_VALUE,
    ):
        if batch_size <= 0:
            raise ValueError("Residual target batch size must be positive.")
        self._transform = transform
        self._rollout = rollout
        self._batch_size = batch_size
        self._target_shape = (action_horizon, action_dim)
        self._action_scale = np.asarray(action_scale if action_scale is not None else (1.0,), dtype=np.float32)
        if self._action_scale.ndim != 1 or not 0 < self._action_scale.size <= action_dim:
            raise ValueError(
                f"Residual action scale has shape {self._action_scale.shape}; expected 1 to {action_dim} values."
            )
        if np.any(self._action_scale <= 0):
            raise ValueError("Residual action scale values must be positive.")
        self._action_scale = np.pad(
            self._action_scale,
            (0, action_dim - self._action_scale.size),
            constant_values=1.0,
        )
        self._intervention_value = intervention_value

    @property
    def target_shape(self) -> tuple[int, int]:
        return self._target_shape

    def build_online(
        self,
        model_state: nnx.State,
        item_getter: Callable[[int], dict],
        indices: Sequence[int],
        *,
        destination: np.ndarray | None = None,
        source_name: str = "dataset",
    ) -> np.ndarray:
        num_items = len(indices)
        if destination is None:
            destination = np.zeros((num_items, *self._target_shape), dtype=np.float32)
        if destination.shape != (num_items, *self._target_shape):
            raise ValueError(
                f"Residual target destination has shape {destination.shape}; "
                f"expected {(num_items, *self._target_shape)}."
            )
        destination.fill(0.0)
        if num_items == 0:
            return destination

        started_at = time.monotonic()
        total_batches = (num_items + self._batch_size - 1) // self._batch_size
        for batch_number, start in enumerate(range(0, num_items, self._batch_size), start=1):
            stop = min(start + self._batch_size, num_items)
            prepared = [self._prepare_online_item(item_getter(int(indices[offset]))) for offset in range(start, stop)]
            intervention_items = [(item, mask) for item, mask in prepared if np.any(mask)]
            if not intervention_items:
                continue

            items = [item for item, _ in intervention_items]
            intervention_masks = [mask for _, mask in intervention_items]
            actual_batch_size = len(items)
            if actual_batch_size < self._batch_size:
                padding = self._batch_size - actual_batch_size
                items.extend([items[-1]] * padding)
                intervention_masks.extend([intervention_masks[-1]] * padding)

            batch = jax.tree.map(
                lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
                *items,
            )
            ground_truth_actions = np.asarray(batch["actions"], dtype=np.float32)
            base_actions = self._rollout(model_state, _model.Observation.from_dict(batch))
            if base_actions.shape != ground_truth_actions.shape:
                raise ValueError(
                    f"Base rollout returned shape {base_actions.shape}; expected {ground_truth_actions.shape}."
                )
            residual_targets = (ground_truth_actions - base_actions) / self._action_scale
            residual_targets = np.where(
                np.asarray(intervention_masks, dtype=np.bool_)[..., None],
                residual_targets,
                0.0,
            )
            intervention_offset = 0
            for output_offset, (_, intervention_mask) in enumerate(prepared):
                if np.any(intervention_mask):
                    destination[start + output_offset] = residual_targets[intervention_offset]
                    intervention_offset += 1

            if batch_number in (1, total_batches) or batch_number % 10 == 0:
                elapsed = max(time.monotonic() - started_at, 1e-6)
                logging.info(
                    "Online residual targets %s: %d/%d samples (%.1f samples/s)",
                    source_name,
                    stop,
                    num_items,
                    stop / elapsed,
                )

        return destination

    def _prepare_online_item(self, item: dict) -> tuple[dict, np.ndarray]:
        raw_item = dict(item)
        control_flag = raw_item.pop(CONTROL_FLAG_FEATURE_NAME, None)
        if control_flag is None:
            raise ValueError(f"Online residual target generation requires the {CONTROL_FLAG_FEATURE_NAME!r} feature.")
        control_flag = np.asarray(control_flag, dtype=np.float32).reshape(-1)
        if control_flag.shape != (self._target_shape[0],):
            raise ValueError(
                f"Online control_flag chunk has shape {control_flag.shape}; expected {(self._target_shape[0],)}."
            )
        intervention_steps = select_intervention_steps(control_flag, self._intervention_value)

        transformed = self._transform(raw_item)
        actions = np.asarray(transformed["actions"])
        if actions.shape != self._target_shape:
            raise ValueError(f"Transformed actions have shape {actions.shape}; expected {self._target_shape}.")

        if transformed.get("wrench") is None:
            transformed["wrench"] = np.zeros((self._target_shape[0], 6), dtype=np.float32)
            transformed["wrench_mask"] = np.asarray(0, dtype=np.bool_)
        else:
            transformed["wrench_mask"] = np.asarray(1, dtype=np.bool_)
        return transformed, intervention_steps

    # Kept as a narrow compatibility alias for callers that used the old builder
    # entry point. It still requires online control flags and never builds offline targets.
    build = build_online
