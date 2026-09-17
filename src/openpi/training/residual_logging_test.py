import jax.numpy as jnp
import pytest

from openpi.training import residual_logging


def test_residual_prediction_metrics_separate_fallback_sources_and_interventions():
    prediction = jnp.full((2, 2, 7), 0.1)
    prediction = prediction.at[1, 0].set(0.2)
    prediction = prediction.at[1, 1].set(1.5)
    target = jnp.zeros_like(prediction)
    target = target.at[1, 1].set(1.0)

    metrics = residual_logging.residual_prediction_metrics(
        prediction,
        target,
        is_online=jnp.array([False, True]),
        action_scale=(1.0,) * 7,
    )

    assert metrics["residual/fallback/fraction"] == pytest.approx(0.75)
    assert metrics["residual/fallback/additive_abs_mean_normalized"] == pytest.approx(0.4 / 3)
    assert metrics["residual/offline_fallback/additive_abs_mean_normalized"] == pytest.approx(0.1)
    assert metrics["residual/online_fallback/additive_abs_mean_normalized"] == pytest.approx(0.2)
    assert metrics["residual/intervention/error_rmse_model"] == pytest.approx(0.5)
    assert metrics["residual/fallback/additive_mean_normalized/x"] == pytest.approx(0.4 / 3)


def test_residual_prediction_metrics_mark_missing_groups_as_nan():
    prediction = jnp.zeros((1, 2, 7))
    metrics = residual_logging.residual_prediction_metrics(
        prediction,
        jnp.zeros_like(prediction),
        is_online=jnp.array([False]),
        action_scale=(1.0,) * 7,
    )

    assert jnp.isnan(metrics["residual/online_fallback/additive_abs_mean_normalized"])
    assert jnp.isnan(metrics["residual/intervention/error_rmse_model"])
