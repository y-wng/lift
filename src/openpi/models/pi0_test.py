from flax import traverse_util
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import pi0 as _pi0
from openpi.models import residual_policy as _residual_policy
import openpi.models.gemma as _gemma
from openpi.models.gru import create_gru_network
import openpi.models.pi0_config as _pi0_config
from openpi.shared.attention_visualization import save_attention_mask
import openpi.shared.nnx_utils as nnx_utils


def _residual_config(**kwargs) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        cross_attention_config="dummy",
        action_horizon=4,
        action_dim=8,
        use_state=False,
        residual_policy_in_use=True,
        residual_width=32,
        residual_mlp_dim=64,
        residual_num_layers=2,
        residual_num_heads=4,
        residual_dropout_rate=0.0,
        residual_wrench_hidden_dim=16,
        **kwargs,
    )


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_pi0_residual_policy_freeze_filter():
    config = _residual_config()
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    freeze_filter = config.get_freeze_filter()
    frozen_state = nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()
    trainable_state = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(freeze_filter))).flat_state()

    assert all("residual_policy" not in path for path in frozen_state)
    assert any("PaliGemma" in path and "img" in path for path in frozen_state)
    assert any("residual_policy" in path and "wrench_gru" in path for path in trainable_state)
    assert any("residual_policy" in path and "core" in path for path in trainable_state)
    assert all("image_encoder" not in path for path in trainable_state)
    assert all("residual_policy" in path for path in trainable_state)


def test_residual_policy_parameter_budget():
    config = _pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        use_state=False,
        residual_policy_in_use=True,
    )
    image_feature_dim = _gemma.get_config(config.paligemma_variant).width
    image_token_count = (224 // 14) ** 2 * len(config.residual_image_keys)

    def create_residual_modules(key):
        rngs = nnx.Rngs(key)
        core = _residual_policy.create_residual_policy_core(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            image_token_count=image_token_count,
            image_feature_dim=image_feature_dim,
            width=config.residual_width,
            mlp_dim=config.residual_mlp_dim,
            num_layers=config.residual_num_layers,
            num_heads=config.residual_num_heads,
            dropout_rate=config.residual_dropout_rate,
            dtype=config.dtype,
        )
        core.lazy_init(rngs=rngs, method="init")
        wrench_gru = create_gru_network(
            hidden_size=config.residual_wrench_hidden_dim,
            output_size=config.residual_width,
            action_horizon=config.action_horizon,
        )
        wrench_gru.lazy_init(rngs=rngs, method="init")
        return nnx.Dict(core=core, wrench_gru=wrench_gru)

    abstract_modules = nnx.eval_shape(create_residual_modules, jax.random.key(0))
    parameter_count = sum(
        variable_state.value.size for variable_state in nnx.state(abstract_modules, nnx.Param).flat_state().values()
    )

    assert 28_000_000 <= parameter_count <= 32_000_000


def test_residual_state_update_only_touches_residual_subtree():
    config = _residual_config()
    model_a = config.create(jax.random.key(0))
    model_b = config.create(jax.random.key(1))
    residual_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*residual_policy.*"))
    base_filter = nnx.All(nnx.Param, nnx.Not(residual_filter))

    base_before = nnx.state(model_a, base_filter).flat_state()
    residual_state = nnx.state(model_a).filter(residual_filter)
    residual_state.replace_by_pure_dict(nnx.state(model_b).filter(residual_filter).to_pure_dict())
    base_after = nnx.state(model_a, base_filter).flat_state()

    assert base_before.keys() == base_after.keys()
    for key in base_before:
        assert jnp.array_equal(base_before[key].value, base_after[key].value)


def test_residual_policy_submodule_helper_uses_matching_paths():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        cross_attention_config="dummy",
        action_horizon=4,
        action_dim=8,
        use_state=False,
        residual_policy_in_use=True,
    )
    residual_model = nnx.Dict(
        residual_policy=_pi0.create_residual_policy_submodule(config, nnx.Rngs(jax.random.key(0)))
    )
    flat_state = nnx.state(residual_model).to_pure_dict()
    flat_state = traverse_util.flatten_dict(flat_state, sep="/")

    assert flat_state
    assert all(path.startswith("residual_policy/") for path in flat_state)


def test_residual_params_can_fill_abstract_model_residual_subtree():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        cross_attention_config="dummy",
        action_horizon=4,
        action_dim=8,
        use_state=False,
        residual_policy_in_use=True,
    )
    concrete_model = config.create(jax.random.key(0))
    residual_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*residual_policy.*"))
    residual_params = nnx.state(concrete_model).filter(residual_filter).to_pure_dict()

    abstract_model = nnx.eval_shape(config.create, jax.random.key(1))
    _, abstract_state = nnx.split(abstract_model)
    abstract_state.replace_by_pure_dict(residual_params)

    flat_state = traverse_util.flatten_dict(abstract_state.to_pure_dict(), sep="/")
    residual_missing = [
        path
        for path, value in flat_state.items()
        if path.startswith("residual_policy/") and isinstance(value, jax.ShapeDtypeStruct)
    ]

    assert not residual_missing


def test_residual_cross_attn_mask(tmp_path):
    image_mask = jnp.array([[True, True, False]])
    wrench_mask = jnp.array([True])
    mask = _residual_policy.make_residual_cross_attn_mask(image_mask, wrench_mask, action_horizon=4)

    assert mask.shape == (1, 1, 4, 7)
    assert mask[0, 0, 0].tolist() == [True, True, False, True, False, False, False]
    assert mask[0, 0, 3].tolist() == [True, True, False, True, True, True, True]
    save_attention_mask(mask[0, 0], file_path=tmp_path / "residual_cross_attn_mask.png")

    no_wrench_mask = _residual_policy.make_residual_cross_attn_mask(image_mask, jnp.array([False]), action_horizon=4)
    assert no_wrench_mask[0, 0, 3].tolist() == [True, True, False, False, False, False, False]


def test_force_history_mask_ablation_keeps_only_first_force_token():
    history_mask = _pi0.make_cross_attn_mask((1, 4, 4, 1), use_force_history=True)
    single_frame_mask = _pi0.make_cross_attn_mask((1, 4, 4, 1), use_force_history=False)

    assert history_mask[0, 0, 2].tolist() == [True, True, False, False]
    assert single_frame_mask[0, 0, 2].tolist() == [True, False, False, False]


def test_residual_wrench_gru_is_causal():
    config = _residual_config()
    model = config.create(jax.random.key(0))
    wrench = jnp.arange(24, dtype=jnp.float32).reshape(1, 4, 6) / 10
    future_changed = wrench.at[:, 3, :].add(1_000)

    memory = model.residual_policy.wrench_gru(wrench)
    changed_memory = model.residual_policy.wrench_gru(future_changed)

    assert jnp.array_equal(memory[:, :3], changed_memory[:, :3])
    assert not jnp.array_equal(memory[:, 3], changed_memory[:, 3])


def test_residual_action_is_causal_with_respect_to_wrench():
    config = _residual_config()
    model = config.create(jax.random.key(0))
    core = model.residual_policy.core
    core.action_out_proj["kernel"].value = jax.random.normal(
        jax.random.key(1), core.action_out_proj["kernel"].value.shape
    )
    image_feature_dim = core.condition_obs_proj["kernel"].value.shape[0]
    image_memory = jax.random.normal(jax.random.key(2), (1, 256, image_feature_dim))
    image_mask = jnp.ones((1, 256), dtype=jnp.bool_)
    wrench = jnp.arange(24, dtype=jnp.float32).reshape(1, 4, 6) / 10
    future_changed = wrench.at[:, 3, :].add(1_000)

    def residual_actions(wrench_sequence):
        memory = model.residual_policy.wrench_gru(wrench_sequence)
        return core(
            image_memory,
            image_mask,
            memory,
            jnp.array([True]),
            jax.random.key(5),
            train=False,
        )

    actions = residual_actions(wrench)
    changed_actions = residual_actions(future_changed)

    assert jnp.array_equal(actions[:, :3], changed_actions[:, :3])
    assert not jnp.array_equal(actions[:, 3], changed_actions[:, 3])


def test_residual_loss_directly_regresses_only_physical_dimensions():
    key = jax.random.key(11)
    config = _residual_config()
    model = config.create(key)
    actions = jnp.ones((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    actions = actions.at[..., -1].set(1_000)
    observation = config.fake_obs(1)

    assert jnp.any(model.residual_policy.core.action_out_proj["kernel"].value != 0)
    model.residual_policy.core.action_out_proj["kernel"].value = jnp.zeros_like(
        model.residual_policy.core.action_out_proj["kernel"].value
    )
    model.residual_policy.core.action_out_proj["bias"].value = jnp.zeros_like(
        model.residual_policy.core.action_out_proj["bias"].value
    )

    loss, prediction = model.compute_residual_loss_with_prediction(key, observation, actions)
    assert jnp.array_equal(prediction, jnp.zeros_like(prediction))
    assert jnp.allclose(loss, 1.0)


def test_residual_prediction_is_deterministic_and_masks_padding():
    config = _residual_config()
    model = config.create(jax.random.key(0))
    core = model.residual_policy.core
    image_feature_dim = core.condition_obs_proj["kernel"].value.shape[0]
    image_memory = jax.random.normal(jax.random.key(1), (1, 256, image_feature_dim))
    image_mask = jnp.ones((1, 256), dtype=jnp.bool_)
    wrench = jnp.zeros((1, config.action_horizon, 6), dtype=jnp.float32)
    wrench_mask = jnp.array([True])

    prediction_a = model._sample_residual_actions(
        jax.random.key(2), image_memory, image_mask, wrench, wrench_mask, num_steps=1
    )
    prediction_b = model._sample_residual_actions(
        jax.random.key(3), image_memory, image_mask, wrench, wrench_mask, num_steps=50
    )

    assert jnp.array_equal(prediction_a, prediction_b)
    assert jnp.array_equal(prediction_a[..., 7:], jnp.zeros_like(prediction_a[..., 7:]))
