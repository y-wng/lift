import logging
import pathlib
from typing import Any

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp

import openpi.models.model as _model
from openpi.policies import flexiv_transforms
import openpi.policies.policy as _policy
import openpi.shared.array_typing as at
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    if getattr(train_config.model, "residual_policy_in_use", False):
        model = _load_residual_policy_model(train_config, checkpoint_dir / "params")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    if getattr(train_config.model, "residual_policy_in_use", False):
        norm_stats = transforms.residual_policy_norm_stats(norm_stats)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        rtc_rebase=(
            flexiv_transforms.rebase_prefix_observation
            if isinstance(train_config.data, _config.LeRobotSingleiPhoneFlexivDataConfig)
            and train_config.data.extra_delta_transform
            else None
        ),
        rtc_state_key=flexiv_transforms.STATE_KEY,
    )


def _drop_abstract_leaves(params: at.Params) -> at.Params:
    flat_params = traverse_util.flatten_dict(params)
    return traverse_util.unflatten_dict(
        {key: value for key, value in flat_params.items() if not isinstance(value, jax.ShapeDtypeStruct)}
    )


def _assert_no_abstract_leaves(params: at.Params) -> None:
    flat_params = traverse_util.flatten_dict(params, sep="/")
    missing = [key for key, value in flat_params.items() if isinstance(value, jax.ShapeDtypeStruct)]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Residual policy load left abstract params unresolved: {preview}")


def _load_residual_policy_model(train_config: _config.TrainConfig, params_path: pathlib.Path) -> _model.BaseModel:
    model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    graphdef, state = nnx.split(model)

    base_params = train_config.weight_loader.load(state.to_pure_dict())
    if any(isinstance(value, jax.ShapeDtypeStruct) for value in traverse_util.flatten_dict(base_params).values()):
        raise ValueError(
            "Residual policy loading requires train_config.weight_loader to provide a concrete fixed base policy."
        )
    base_params = _drop_abstract_leaves(base_params)
    state.replace_by_pure_dict(base_params)

    residual_params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    state.replace_by_pure_dict(residual_params)
    _assert_no_abstract_leaves(state.to_pure_dict())

    return nnx.merge(graphdef, state)
