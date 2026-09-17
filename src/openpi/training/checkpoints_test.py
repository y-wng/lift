from flax import nnx
import jax
import optax

from openpi.models import pi0_config
from openpi.training import checkpoints
from openpi.training import utils as training_utils


def test_residual_checkpoint_filter_keeps_only_residual_params():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        cross_attention_config="dummy",
        action_horizon=4,
        use_state=False,
        residual_policy_in_use=True,
        residual_width=32,
        residual_mlp_dim=64,
        residual_num_layers=2,
        residual_num_heads=4,
        residual_dropout_rate=0.0,
        residual_wrench_hidden_dim=16,
    )
    model = nnx.eval_shape(config.create, jax.random.key(0))
    params = nnx.state(model)
    residual_filter = nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))
    tx = optax.identity()
    state = training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(residual_filter)),
        ema_decay=0.99,
        ema_params=params,
    )

    filtered = checkpoints._filter_checkpoint_params(state, residual_filter)
    params_paths = ["/".join(str(part) for part in path) for path in filtered.params.flat_state()]
    ema_paths = ["/".join(str(part) for part in path) for path in filtered.ema_params.flat_state()]

    assert params_paths
    assert all("residual_policy" in path for path in params_paths)
    assert any("residual_policy/core" in path for path in params_paths)
    assert any("residual_policy/wrench_gru" in path for path in params_paths)
    assert all("image_encoder" not in path for path in params_paths)
    assert ema_paths == params_paths
