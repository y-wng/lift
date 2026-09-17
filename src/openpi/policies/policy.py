from collections.abc import Callable, Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        rtc_rebase: Callable[[np.ndarray, np.ndarray | None, dict], dict] | None = None,
        rtc_state_key: str | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
        """
        self._rtc_rebase = rtc_rebase
        self._rtc_state_key = rtc_state_key
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._rng = rng or jax.random.key(0)
        if hasattr(model, "sample_actions_rtc"):
            self._sample_actions_rtc = nnx_utils.module_jit(model.sample_actions_rtc)
        else:
            self._sample_actions_rtc = None
        self._reactive_replan = (
            nnx_utils.module_jit(model.reactive_replan) if hasattr(model, "reactive_replan") else None
        )
        self._reactive_get_k_actions = (
            nnx_utils.module_jit(model.reactive_get_k_actions) if hasattr(model, "reactive_get_k_actions") else None
        )

        # Store previous actions for prefix_actions functionality
        self._prev_actions: np.ndarray | None = None
        # Store the observation/state (absolute pose in umi coord system) that the previous actions were conditioned on.
        # This is critical for RTC: previous actions are relative (pos+rpy) w.r.t. the *previous* base state, not the
        # current state. If we rebase them using the wrong state, the reconstructed absolute trajectory can drift
        # significantly and show counter-intuitive trends (e.g., z monotonicity flipping).
        self._prev_state_abs: np.ndarray | None = None

    def reset(self) -> None:
        """Reset RTC internal state for a new episode.

        Clears cached previous actions and the absolute state they were conditioned
        on, so the next inference starts from a clean slate without stale prefix
        information from the previous episode.
        """
        self._prev_actions = None
        self._prev_state_abs = None

    def _shift_prefix_actions(self, action_chunk: np.ndarray, time_base: int) -> np.ndarray:
        """
        Shift the unexecuted prefix and pad it to the model action horizon.
        This function converts all actions to relative increments from time_base,
        extracts the remainder of actions starting from time_base,
        and pads with zeros to maintain original length.

        Args:
            action_chunk: Action chunk with shape (action_horizon, action_dim)
            time_base: Index in the original chunk that corresponds to current state (max(k_robot_orig, k_gripper_orig))

        Returns:
            Shifted action chunk with same shape as input, where all actions are relative increments from time_base
        """
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with 2 dims, got shape {action_chunk.shape}")

        time_base = max(time_base, 0)
        if time_base >= action_chunk.shape[0]:
            raise ValueError(f"time_base ({time_base}) must be less than action chunk length ({action_chunk.shape[0]})")

        # Convert all actions to relative increments from time_base
        # Extract remainder starting from time_base
        remainder = action_chunk[time_base + 1 :]
        # Pad zeros on the right side to maintain the original length
        pad_length = action_chunk.shape[0] - remainder.shape[0]
        if pad_length > 0:
            pad = np.zeros((pad_length, action_chunk.shape[1]), dtype=action_chunk.dtype)
            return np.concatenate([remainder, pad], axis=0)
        return remainder

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        time_base: int | None = None,
        rtc_config: dict[str, float | int] | None = None,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        # Store original obs for later use in prefix_actions processing
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to JAX arrays.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        # Prepare kwargs for sample_actions or sample_actions_rtc
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        # Determine which method to use based on time_base and rtc_config
        use_rtc = time_base is not None and rtc_config is not None

        # Prepare prefix_actions if time_base is provided and we have previous actions
        prefix_actions: np.ndarray | None = None
        if use_rtc and self._prev_actions is not None:
            prefix_actions = np.asarray(self._prev_actions)
            if prefix_actions.ndim == 1:
                prefix_actions = prefix_actions[None, ...]

            if self._rtc_rebase is None:
                raise ValueError("The selected data config does not define an RTC pose adapter.")
            rtc_inputs = self._rtc_rebase(prefix_actions, self._prev_state_abs, obs)
            rtc_inputs = self._input_transform(rtc_inputs)
            prefix_actions = rtc_inputs["actions"]
            prefix_actions = self._shift_prefix_actions(prefix_actions, time_base)

            if self._sample_actions_rtc is None:
                raise ValueError("Model does not support sample_actions_rtc")

            # Convert prefix_actions to appropriate format
            prefix_actions_tensor = jnp.asarray(prefix_actions)

            # Extract RTC parameters from rtc_config
            inference_delay = rtc_config.get("inference_delay")
            prefix_attention_horizon = rtc_config.get("prefix_attention_horizon")
            max_guidance_weight = rtc_config.get("max_guidance_weight")

            if inference_delay is None or prefix_attention_horizon is None or max_guidance_weight is None:
                raise ValueError(
                    "rtc_config must contain 'inference_delay', 'prefix_attention_horizon', and 'max_guidance_weight'"
                )

            # Add RTC-specific parameters to sample_kwargs
            sample_kwargs_rtc = dict(sample_kwargs)
            sample_kwargs_rtc["prefix_actions"] = prefix_actions_tensor
            sample_kwargs_rtc["inference_delay"] = inference_delay
            sample_kwargs_rtc["prefix_attention_horizon"] = prefix_attention_horizon
            sample_kwargs_rtc["max_guidance_weight"] = max_guidance_weight

            # Call both methods for comparison
            start_time = time.monotonic()
            actions_rtc = self._sample_actions_rtc(sample_rng, observation, **sample_kwargs_rtc)
            model_time = time.monotonic() - start_time
            outputs = {
                "state": inputs["state"],
                "actions": actions_rtc,  # Use RTC actions as the main output
            }
            outputs2 = {
                "state": inputs["state"],
                "actions": prefix_actions,
            }

        else:
            sample_method = self._sample_actions
            start_time = time.monotonic()
            outputs = {
                "state": inputs["state"],
                "actions": sample_method(sample_rng, observation, **sample_kwargs),
            }
            model_time = time.monotonic() - start_time

        # Include prev_state if it exists in inputs
        if "prev_state" in inputs:
            outputs["prev_state"] = inputs["prev_state"]

        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Save actions before output transform for next iteration's prefix_actions

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if use_rtc and prefix_actions is not None and time_base != 0:
            outputs2 = self._output_transform(outputs2)
            outputs["prefix_actions"] = outputs2["actions"]
        if "actions" in outputs:
            actions_to_save = outputs["actions"]
            # Ensure 2D shape (action_horizon, action_dim)
            self._prev_actions = actions_to_save.copy()
            # Cache the absolute state that these actions were conditioned on (for RTC rebasing).
            state = obs.get(self._rtc_state_key) if self._rtc_state_key is not None else None
            self._prev_state_abs = None if state is None else np.asarray(state).copy()

        return outputs

    def reactive_replan(self, obs: dict, *, num_steps: int = 10) -> dict:
        if self._reactive_replan is None:
            raise ValueError("Reactive replan is not supported by this policy model.")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        action_horizon, prefix_tokens, prefix_mask, prefix_ar_mask, dt, kv_cache, action_dim = self._reactive_replan(
            observation, num_steps=num_steps
        )
        self._rng, noise_rng = jax.random.split(self._rng)
        noise = jax.random.normal(noise_rng, (1, int(action_horizon), int(action_dim)))
        self._rng, reac_noise_rng = jax.random.split(self._rng)
        reac_noise = jax.random.normal(reac_noise_rng, (1, int(action_horizon), int(action_dim)))
        return {
            "action_horizon": int(action_horizon),
            "action_dim": int(action_dim),
            "prefix_tokens": prefix_tokens,
            "prefix_mask": prefix_mask,
            "prefix_ar_mask": prefix_ar_mask,
            "dt": dt,
            "kv_cache": kv_cache,
            "noise": noise,
            "reac_noise": reac_noise,
            "sequence_id": int(obs.get("sequence_id", 0)),
        }

    def reactive_get_k_actions(
        self,
        *,
        obs: dict,
        k: int,
        wrench: np.ndarray | None = None,
        noise,
        reac_noise,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        dt,
        kv_cache,
        current_step: int,
        sequence_id: int = 0,
    ) -> dict:
        if self._reactive_get_k_actions is None:
            raise ValueError("Reactive get_k_actions is not supported by this policy model.")
        if k <= 0:
            raise ValueError("k must be positive.")
        action_horizon = int(noise.shape[1])
        next_step = min(current_step + k, action_horizon)
        step_size = next_step - current_step
        if current_step >= action_horizon:
            return {"actions": None, "sequence_id": sequence_id, "next_step": current_step}

        obs = dict(obs)
        if wrench is not None:
            obs["wrench"] = wrench
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        actions = self._reactive_get_k_actions(
            observation=observation,
            noise=noise,
            reac_noise=reac_noise,
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            prefix_ar_mask=prefix_ar_mask,
            dt=dt,
            kv_cache=kv_cache,
        )
        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["sequence_id"] = sequence_id
        outputs["next_step"] = next_step
        outputs["actions"] = outputs["actions"][current_step:next_step, :]
        last_action = np.array([outputs["actions"][-1]])
        for _ in range(action_horizon - step_size):
            outputs["actions"] = np.concatenate([outputs["actions"], last_action], axis=0)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
