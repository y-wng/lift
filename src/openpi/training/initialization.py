"""Train-state initialization shared by offline training, DAgger, and evaluation."""

import dataclasses
import logging
from typing import Any

from flax import traverse_util
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def maybe_copy_reactive_expert_params(state: training_utils.TrainState, *, enabled: bool) -> training_utils.TrainState:
    """Optionally copy action expert (_1) params to reactive expert (_2)."""
    if not enabled:
        return state

    def _clone_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _clone_value(v) for k, v in value.items()}
        return value.copy()

    def _count_tensors(value: Any) -> int:
        return 1

    def _align_reactive_params(tree: dict[str, Any], *, track_count: bool) -> tuple[dict[str, Any], int]:
        copied = 0

        if not isinstance(tree, dict):
            return tree, copied

        for key in list(tree.keys()):
            if key.endswith("_2"):
                src_key = f"{key[:-2]}_1"
                if src_key in tree:
                    tree[key] = _clone_value(tree[src_key])
                    if track_count:
                        copied += _count_tensors(tree[src_key])

        for key, value in tree.items():
            if isinstance(value, dict):
                tree[key], child_copied = _align_reactive_params(value, track_count=track_count)
                copied += child_copied
        return tree, copied

    def _copy_state(params: nnx.State | None, *, track_count: bool) -> tuple[nnx.State | None, int]:
        if params is None:
            return None, 0

        model = nnx.merge(state.model_def, params)
        graphdef, model_state = nnx.split(model)
        params_dict, copied = _align_reactive_params(model_state.to_pure_dict(), track_count=track_count)
        model_state.replace_by_pure_dict(params_dict)
        model = nnx.merge(graphdef, model_state)
        return nnx.state(model), copied

    params, copied = _copy_state(state.params, track_count=True)
    ema_params, _ = _copy_state(state.ema_params, track_count=False)

    if copied > 0:
        logging.info("Copied %d reactive expert parameter tensors from action expert.", copied)
    else:
        logging.warning("Reactive expert copy enabled, but no *_2 params were found.")
    return dataclasses.replace(state, params=params, ema_params=ema_params)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume and not getattr(config.model, "residual_policy_in_use", False):
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding
