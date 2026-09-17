from collections.abc import Sequence

import jax.numpy as jnp

_ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    mask = jnp.broadcast_to(mask, values.shape)
    count = jnp.sum(mask)
    total = jnp.sum(jnp.where(mask, values, 0.0))
    return jnp.where(count > 0, total / jnp.maximum(count, 1), jnp.nan)


def _masked_max(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    mask = jnp.broadcast_to(mask, values.shape)
    return jnp.where(jnp.any(mask), jnp.max(jnp.where(mask, values, 0.0)), jnp.nan)


def residual_prediction_metrics(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    is_online: jnp.ndarray,
    action_scale: Sequence[float],
) -> dict[str, jnp.ndarray]:
    """Summarize direct residual predictions for W&B without another model forward pass."""

    action_dim = len(action_scale)
    prediction = prediction[..., :action_dim]
    target = target[..., :action_dim]
    scale = jnp.asarray(action_scale, dtype=prediction.dtype)
    additive_prediction = prediction * scale
    additive_target = target * scale

    fallback_mask = jnp.all(jnp.isclose(target, 0.0, atol=1e-6), axis=-1)
    intervention_mask = jnp.logical_not(fallback_mask)
    online_mask = jnp.broadcast_to(is_online[..., None], fallback_mask.shape)
    offline_fallback_mask = jnp.logical_and(fallback_mask, jnp.logical_not(online_mask))
    online_fallback_mask = jnp.logical_and(fallback_mask, online_mask)

    error = prediction - target
    additive_error = additive_prediction - additive_target
    metrics = {
        "residual/all/pred_abs_mean_model": jnp.mean(jnp.abs(prediction)),
        "residual/all/target_abs_mean_model": jnp.mean(jnp.abs(target)),
        "residual/all/error_rmse_model": jnp.sqrt(jnp.mean(jnp.square(error))),
        "residual/fallback/fraction": jnp.mean(fallback_mask),
        "residual/fallback/pred_abs_mean_model": _masked_mean(jnp.abs(prediction), fallback_mask),
        "residual/fallback/pred_rmse_model": jnp.sqrt(_masked_mean(jnp.square(prediction), fallback_mask)),
        "residual/fallback/additive_abs_mean_normalized": _masked_mean(jnp.abs(additive_prediction), fallback_mask),
        "residual/fallback/additive_max_abs_normalized": _masked_max(jnp.abs(additive_prediction), fallback_mask),
        "residual/offline_fallback/fraction": jnp.mean(offline_fallback_mask),
        "residual/offline_fallback/additive_abs_mean_normalized": _masked_mean(
            jnp.abs(additive_prediction), offline_fallback_mask
        ),
        "residual/online_fallback/fraction": jnp.mean(online_fallback_mask),
        "residual/online_fallback/additive_abs_mean_normalized": _masked_mean(
            jnp.abs(additive_prediction), online_fallback_mask
        ),
        "residual/intervention/fraction": jnp.mean(intervention_mask),
        "residual/intervention/error_rmse_model": jnp.sqrt(_masked_mean(jnp.square(error), intervention_mask)),
        "residual/intervention/error_rmse_additive_normalized": jnp.sqrt(
            _masked_mean(jnp.square(additive_error), intervention_mask)
        ),
    }

    for dim in range(action_dim):
        name = _ACTION_NAMES[dim] if dim < len(_ACTION_NAMES) else f"dim_{dim}"
        metrics[f"residual/fallback/additive_mean_normalized/{name}"] = _masked_mean(
            additive_prediction[..., dim], fallback_mask
        )
        metrics[f"residual/fallback/additive_abs_mean_normalized/{name}"] = _masked_mean(
            jnp.abs(additive_prediction[..., dim]), fallback_mask
        )
        metrics[f"residual/intervention/pred_additive_mean_normalized/{name}"] = _masked_mean(
            additive_prediction[..., dim], intervention_mask
        )
        metrics[f"residual/intervention/target_additive_mean_normalized/{name}"] = _masked_mean(
            additive_target[..., dim], intervention_mask
        )

    return metrics
