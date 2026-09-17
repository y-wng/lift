from types import SimpleNamespace

from flax import nnx
import jax.numpy as jnp
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import weight_loaders as _weight_loaders
from scripts import train_online_dagger as _train


def _residual_train_config() -> _config.TrainConfig:
    model = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        cross_attention_config="dummy",
        action_horizon=4,
        residual_policy_in_use=True,
    )
    return _config.TrainConfig(
        name="test_residual",
        model=model,
        weight_loader=_weight_loaders.NoOpWeightLoader(),
    )


def test_residual_init_uses_base_then_residual_overlay():
    args = _config.OnlineDaggerTrainConfig(
        config_name="test_residual",
        base_init_checkpoint="/tmp/base/params",
        residual_init_checkpoint="/tmp/residual/params",
    )

    configured = _train._configure_init_weight_loader(_residual_train_config(), args)

    assert isinstance(configured.weight_loader, _weight_loaders.OverlayWeightLoader)
    assert configured.weight_loader.overlay_params_path == "/tmp/residual/params"
    assert isinstance(configured.weight_loader.base_loader, _weight_loaders.CheckpointWeightLoader)
    assert configured.weight_loader.base_loader.params_path == "/tmp/base/params"


def test_split_and_legacy_init_flags_cannot_be_mixed():
    args = _config.OnlineDaggerTrainConfig(
        config_name="test_residual",
        init_checkpoint="/tmp/legacy/params",
        base_init_checkpoint="/tmp/base/params",
    )

    with pytest.raises(ValueError, match="either --init-checkpoint"):
        _train._configure_init_weight_loader(_residual_train_config(), args)


def test_prefetch_iterator_preserves_order():
    iterator = _train._PrefetchIterator((index for index in range(5)), depth=2)
    try:
        assert [next(iterator) for _ in range(5)] == list(range(5))
    finally:
        iterator.close()


def test_residual_ema_updates_only_trainable_params():
    config = _residual_train_config()
    ema_params = nnx.State(
        {
            "base": {"weight": nnx.Param(jnp.array([1.0]))},
            "residual_policy": {"weight": nnx.Param(jnp.array([2.0]))},
        }
    )
    new_params = nnx.State(
        {
            "base": {"weight": nnx.Param(jnp.array([99.0]))},
            "residual_policy": {"weight": nnx.Param(jnp.array([4.0]))},
        }
    )
    state = SimpleNamespace(ema_decay=0.5, ema_params=ema_params)
    updated = _train._update_ema_params(config, state, new_params)

    assert updated is not None
    base_weight = getattr(updated["base"]["weight"], "value", updated["base"]["weight"])
    residual_weight = getattr(updated["residual_policy"]["weight"], "value", updated["residual_policy"]["weight"])
    assert jnp.asarray(base_weight).tolist() == [1.0]
    assert jnp.asarray(residual_weight).tolist() == [3.0]
