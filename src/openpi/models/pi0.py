from collections.abc import Callable
import logging
from typing import Literal

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import residual_policy as _residual_policy
import openpi.models.gemma as _gemma
from openpi.models.gru import create_gru_network as gru
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.jax_debug as _jax_debug

logger = logging.getLogger("openpi")

PrefixAttentionSchedule = Literal["ones", "zeros", "linear", "exp"]
_SIGLIP_PATCH_SIZE = 14
_SIGLIP_IMAGE_TOKEN_COUNT = (_model.IMAGE_RESOLUTION[0] // _SIGLIP_PATCH_SIZE) * (
    _model.IMAGE_RESOLUTION[1] // _SIGLIP_PATCH_SIZE
)


def create_residual_policy_submodule(config: pi0_config.Pi0Config, rngs: nnx.Rngs) -> nnx.Dict:
    """Create the residual-policy subtree with the same paths used by Pi0."""
    paligemma_config = _gemma.get_config(config.paligemma_variant)
    residual_wrench_gru = gru(
        hidden_size=config.residual_wrench_hidden_dim,
        output_size=config.residual_width,
        action_horizon=config.action_horizon,
    )
    residual_wrench_gru.lazy_init(rngs=rngs, method="init")
    residual_core = _residual_policy.create_residual_policy_core(
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        image_token_count=_SIGLIP_IMAGE_TOKEN_COUNT * len(config.residual_image_keys),
        image_feature_dim=paligemma_config.width,
        width=config.residual_width,
        mlp_dim=config.residual_mlp_dim,
        num_layers=config.residual_num_layers,
        num_heads=config.residual_num_heads,
        dropout_rate=config.residual_dropout_rate,
        dtype=config.dtype,
    )
    residual_core.lazy_init(rngs=rngs, method="init")
    return nnx.Dict(wrench_gru=residual_wrench_gru, core=residual_core)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def make_reactive_attn_mask(input_mask, ar_mask, spec_modules):
    mask_ar = jnp.broadcast_to(ar_mask, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]

    total_len = input_mask.shape[-1]
    state_token_count, action_token_count = spec_modules
    if state_token_count > 0:
        attn_mask = attn_mask.at[
            :,
            -(state_token_count + action_token_count),
            -2 * (state_token_count + action_token_count) : -(state_token_count + action_token_count),
        ].set(False)
    attn_mask = attn_mask.at[:, :, -(state_token_count + action_token_count) :].set(False)
    attn_mask = attn_mask.at[
        :, 0 : -2 * (state_token_count + action_token_count), -2 * (state_token_count + action_token_count) :
    ].set(False)
    for i in range(action_token_count):
        mask_row = jnp.array(
            [
                bool(
                    k < total_len - 2 * (state_token_count + action_token_count)
                    or (k >= total_len - (state_token_count + action_token_count + i) and k < total_len - i)
                )
                for k in range(total_len)
            ],
            dtype=bool,
        )
        attn_mask = attn_mask.at[:, -i - 1].set(mask_row)

    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def make_cross_attn_mask(spec_modules, *, use_force_history: bool = True):
    batch_size, s1, s2, s3 = spec_modules
    attn_mask = jnp.zeros((batch_size, 1, s1, s2), dtype=bool)
    for i in range(s1):
        if s1 == s2 + 1:
            mask_row = jnp.array([k <= (i - s3 - 1) and (use_force_history or k == 0) for k in range(s2)], dtype=bool)
        else:
            mask_row = jnp.array([k <= (i - s3) and (use_force_history or k == 0) for k in range(s2)], dtype=bool)
        attn_mask = attn_mask.at[:, 0, i].set(mask_row)
    return attn_mask


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_state = config.use_state
        self.reactive_in_use = config.reactive_in_use
        self.residual_policy_in_use = config.residual_policy_in_use
        self.original_head = config.original_head
        self.use_force_history = config.use_force_history
        self.cross_attention_latency = config.cross_attention_latency
        self.residual_image_keys = config.residual_image_keys
        self.residual_action_dim = len(config.residual_action_scale) if config.residual_policy_in_use else 0
        self.residual_action_scale = (
            _residual_policy.expand_action_scale(config.residual_action_scale, config.action_dim)
            if config.residual_policy_in_use
            else (1.0,) * config.action_dim
        )

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        cross_config = _gemma.get_config(config.cross_attention_config)

        llm_configs = [paligemma_config, action_expert_config]
        if config.reactive_in_use:
            llm_configs.append(_gemma.get_config(config.reactive_action_expert_variant))

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=llm_configs,
                cross_config=cross_config,
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        if config.pi05:
            use_adarms = [False, True, True] if config.reactive_in_use else [False, True]
        else:
            use_adarms = [False, False, False] if config.reactive_in_use else [False, False]
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        model_parts = {"llm": llm, "img": img}
        if config.reactive_in_use:
            wrench_gru = gru(
                hidden_size=action_expert_config.width // 2,
                output_size=action_expert_config.width,
                action_horizon=config.action_horizon,
            )
            wrench_gru.lazy_init(rngs=rngs, method="init")
            model_parts["wrench_gru"] = wrench_gru
        self.PaliGemma = nnx.Dict(**model_parts)

        if config.residual_policy_in_use:
            self.residual_policy = create_residual_policy_submodule(config, rngs)

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]
        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05 and self.use_state:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        if not self.pi05 and self.use_state:
            ar_mask += [True] + ([False] * (self.action_horizon - 1))
        else:
            ar_mask += [False] * self.action_horizon
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _cross_context(self, observation: _model.Observation):
        if not self.cross_attention_latency:
            raise ValueError("cross_attention_latency must be set")
        batch_size = observation.state.shape[0]
        if (not self.pi05) and self.use_state:
            cross_special_modules = [
                batch_size,
                self.action_horizon + 1,
                self.action_horizon,
                self.cross_attention_latency,
            ]
            q_len = self.action_horizon + 1
        else:
            cross_special_modules = [batch_size, self.action_horizon, self.action_horizon, self.cross_attention_latency]
            q_len = self.action_horizon
        cross_attn_mask = make_cross_attn_mask(
            cross_special_modules,
            use_force_history=self.use_force_history,
        )
        cross_positions = [
            jnp.cumsum(jnp.ones((batch_size, q_len), dtype=jnp.int32), axis=1) - 1,
            jnp.cumsum(jnp.ones((batch_size, self.action_horizon), dtype=jnp.int32), axis=1) - 1,
        ]
        return cross_attn_mask, cross_positions

    def _get_wrench_memory(self, observation: _model.Observation):
        if observation.wrench is None:
            batch_size = observation.state.shape[0]
            wrench = jnp.zeros((batch_size, self.action_horizon, 6), dtype=observation.state.dtype)
        else:
            wrench = observation.wrench
        wrench_mask = self._get_wrench_mask(observation)
        if self.residual_policy_in_use:
            return self.residual_policy.wrench_gru(wrench) * wrench_mask[:, None, None]
        return self.PaliGemma.wrench_gru(wrench) * wrench_mask[:, None, None]

    def _get_wrench_mask(self, observation: _model.Observation):
        if observation.wrench is None:
            batch_size = observation.state.shape[0]
            return jnp.zeros((batch_size,), dtype=jnp.bool_)
        if observation.wrench_mask is None:
            return jnp.ones((observation.wrench.shape[0],), dtype=jnp.bool_)
        return observation.wrench_mask

    def _get_residual_wrench_memory(self, wrench, wrench_mask, batch_size: int):
        if wrench is None:
            wrench = jnp.zeros((batch_size, self.action_horizon, 6), dtype=jnp.float32)
            wrench_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
        else:
            wrench = jnp.asarray(wrench, dtype=jnp.float32)
            if wrench.ndim == 2:
                wrench = wrench[None, ...]
            if wrench_mask is None:
                wrench_mask = jnp.ones((wrench.shape[0],), dtype=jnp.bool_)
            else:
                wrench_mask = jnp.asarray(wrench_mask, dtype=jnp.bool_)
                if wrench_mask.ndim == 0:
                    wrench_mask = jnp.broadcast_to(wrench_mask, (wrench.shape[0],))
        return self.residual_policy.wrench_gru(wrench) * wrench_mask[:, None, None], wrench_mask

    def _get_residual_image_memory(
        self,
        observation: _model.Observation,
        prefix_tokens: jax.Array | None = None,
    ):
        missing_keys = set(self.residual_image_keys) - set(observation.images)
        if missing_keys:
            raise ValueError(f"Residual image keys are missing from the observation: {sorted(missing_keys)}")

        encoded_images = {}
        if prefix_tokens is None:
            for image_key in self.residual_image_keys:
                encoded_images[image_key] = self.PaliGemma.img(observation.images[image_key], train=False)[0]
        else:
            required_prefix_tokens = _SIGLIP_IMAGE_TOKEN_COUNT * len(observation.images)
            if prefix_tokens.shape[1] < required_prefix_tokens:
                raise ValueError(
                    f"Base prefix has {prefix_tokens.shape[1]} tokens, but at least "
                    f"{required_prefix_tokens} image tokens are required."
                )
            token_offset = 0
            for image_key in observation.images:
                next_offset = token_offset + _SIGLIP_IMAGE_TOKEN_COUNT
                if image_key in self.residual_image_keys:
                    encoded_images[image_key] = prefix_tokens[:, token_offset:next_offset]
                token_offset = next_offset

        residual_tokens = []
        image_masks = []
        for image_key in self.residual_image_keys:
            tokens = jax.lax.stop_gradient(encoded_images[image_key])
            if tokens.shape[1] != _SIGLIP_IMAGE_TOKEN_COUNT:
                raise ValueError(
                    f"Frozen SigLIP returned {tokens.shape[1]} tokens for {image_key!r}; "
                    f"expected {_SIGLIP_IMAGE_TOKEN_COUNT}."
                )
            mask = jnp.broadcast_to(observation.image_masks[image_key][:, None], tokens.shape[:2])
            residual_tokens.append(tokens * mask[:, :, None])
            image_masks.append(mask)
        return jnp.concatenate(residual_tokens, axis=1), jnp.concatenate(image_masks, axis=1)

    def _unnormalize_residual_actions(self, normalized_residual):
        scale = jnp.asarray(self.residual_action_scale, dtype=normalized_residual.dtype)
        return normalized_residual * scale

    def _mask_residual_padding(self, residual_actions):
        if self.residual_action_dim == self.action_dim:
            return residual_actions
        return residual_actions.at[..., self.residual_action_dim :].set(0)

    def sample_base_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: jax.Array | None = None,
    ) -> _model.Actions:
        """Sample only the frozen base-policy action chunk for residual target generation."""

        observation = _model.preprocess_observation(None, observation, train=False)
        base_actions, _, _ = self._sample_base_actions_with_prefix(
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
            debug=False,
        )
        return base_actions

    def _sample_base_actions_with_prefix(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: jax.Array | None = None,
        debug: bool = False,
    ):
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            memory=None,
            cross_positions=[jnp.zeros((batch_size, 1), dtype=jnp.int32)] * 2,
            cross_attn_mask=jnp.zeros((batch_size, 1, 1, 1), dtype=bool),
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            pref_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([pref_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                memory=None,
                cross_positions=[jnp.zeros((batch_size, 1), dtype=jnp.int32)] * 2,
                cross_attn_mask=jnp.zeros((batch_size, 1, 1, 1), dtype=bool),
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        base_actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        if debug:
            _jax_debug.inspect(
                "residual/base_actions",
                base_actions,
                prefix_tokens,
                prefix_mask,
                names=("base_actions", "prefix_tokens", "prefix_mask"),
            )
        return base_actions, prefix_tokens, prefix_mask

    def compute_residual_loss_with_prediction(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah ad"]]:
        """Compute the direct residual loss and return the prediction from the same forward pass."""

        if not self.residual_policy_in_use:
            raise ValueError("Residual loss with prediction is only available when residual_policy_in_use=True.")
        preprocess_rng, dropout_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=self.residual_image_keys,
            augment_wrist_rotation=True,
        )

        image_memory, image_mask = self._get_residual_image_memory(observation)
        residual_target = actions

        wrench_memory = self._get_wrench_memory(observation)
        wrench_mask = self._get_wrench_mask(observation)
        _jax_debug.inspect(
            "residual/targets",
            residual_target,
            residual_target[0],
            wrench_memory,
            wrench_mask,
            names=(
                "residual_target",
                "first_sample_residual_chunk",
                "wrench_memory",
                "wrench_mask",
            ),
            full_value_names=("first_sample_residual_chunk",),
        )
        residual_pred = self.residual_policy.core(
            image_memory,
            image_mask,
            wrench_memory,
            wrench_mask,
            dropout_rng,
            train=train,
        )
        residual_loss = jnp.mean(
            jnp.square(
                residual_pred[..., : self.residual_action_dim] - residual_target[..., : self.residual_action_dim]
            ),
            axis=-1,
        )
        _jax_debug.inspect(
            "residual/prediction",
            residual_pred,
            residual_loss,
            names=("residual_pred", "residual_loss"),
        )
        return residual_loss, residual_pred

    def _sample_actions_with_residual(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: jax.Array | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        base_rng, residual_rng = jax.random.split(rng)
        base_actions, prefix_tokens, prefix_mask = self._sample_base_actions_with_prefix(
            base_rng,
            observation,
            num_steps=num_steps,
            noise=noise,
        )
        del prefix_mask
        image_memory, image_mask = self._get_residual_image_memory(observation, prefix_tokens)
        residual_actions = self._sample_residual_actions(
            residual_rng,
            image_memory,
            image_mask,
            observation.wrench,
            observation.wrench_mask,
            num_steps=num_steps,
        )
        return base_actions + self._unnormalize_residual_actions(residual_actions)

    def _sample_residual_actions(
        self,
        rng: at.KeyArrayLike,
        image_memory: jax.Array,
        image_mask: jax.Array,
        wrench: jax.Array | None,
        wrench_mask: jax.Array | None,
        *,
        num_steps: int = 10,
    ) -> jax.Array:
        """Predict a deterministic residual chunk from cached image tokens and current wrench."""
        del num_steps
        batch_size = image_memory.shape[0]
        wrench_memory, wrench_mask = self._get_residual_wrench_memory(wrench, wrench_mask, batch_size)
        residual_actions = self.residual_policy.core(
            image_memory,
            image_mask,
            wrench_memory,
            wrench_mask,
            rng,
            train=False,
        )
        return self._mask_residual_padding(residual_actions)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.residual_policy_in_use:
            residual_loss, _ = self.compute_residual_loss_with_prediction(rng, observation, actions, train=train)
            return residual_loss

        if not self.reactive_in_use:
            preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
            batch_shape = actions.shape[:-2]
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
            u_t = noise - actions
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
                memory=None,
                cross_positions=[jnp.zeros((observation.state.shape[0], 1), dtype=jnp.int32)] * 2,
                cross_attn_mask=jnp.zeros((observation.state.shape[0], 1, 1, 1), dtype=bool),
            )
            del prefix_out
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)

        preprocess_rng, noise_rng, time_rng, reac_noise_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        wrench_memory = self._get_wrench_memory(observation)
        cross_attn_mask, cross_positions = self._cross_context(observation)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        reac_noise = jax.random.normal(reac_noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        reac_x_t = time_expanded * reac_noise + (1 - time_expanded) * actions
        reac_u_t = reac_noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        reacfix_tokens, reacfix_mask, reacfix_ar_mask, reac_adarms_cond = self.embed_suffix(observation, reac_x_t, time)

        input_mask = jnp.concatenate([prefix_mask, suffix_mask, reacfix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask, reacfix_ar_mask], axis=0)
        special_modules = [1, self.action_horizon] if ((not self.pi05) and self.use_state) else [0, self.action_horizon]
        attn_mask = make_reactive_attn_mask(input_mask, ar_mask, special_modules)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        positions = positions.at[:, -(special_modules[0] + special_modules[1]) :].set(
            positions[:, -2 * (special_modules[0] + special_modules[1]) : -(special_modules[0] + special_modules[1])]
        )
        (_, suffix_out, reacfix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens, reacfix_tokens],
            mask=attn_mask,
            adarms_cond=[None, adarms_cond, reac_adarms_cond],
            positions=positions,
            memory=wrench_memory,
            cross_attn_mask=cross_attn_mask,
            cross_positions=cross_positions,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        reac_v_t = self.action_out_proj(reacfix_out[:, -self.action_horizon :])
        if self.original_head:
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)
        return jnp.mean(jnp.square(v_t - u_t), axis=-1) + jnp.mean(jnp.square(reac_v_t - reac_u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: jax.Array | None = None,
    ) -> _model.Actions:
        if self.residual_policy_in_use:
            return self._sample_actions_with_residual(rng, observation, num_steps=num_steps, noise=noise)

        if not self.reactive_in_use:
            observation = _model.preprocess_observation(None, observation, train=False)
            dt = -1.0 / num_steps
            batch_size = observation.state.shape[0]
            if noise is None:
                noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
                memory=None,
                cross_positions=[jnp.zeros((batch_size, 1), dtype=jnp.int32)] * 2,
                cross_attn_mask=jnp.zeros((batch_size, 1, 1, 1), dtype=bool),
            )

            def step(carry):
                x_t, time = carry
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t, jnp.broadcast_to(time, batch_size)
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                    memory=None,
                    cross_positions=[jnp.zeros((batch_size, 1), dtype=jnp.int32)] * 2,
                    cross_attn_mask=jnp.zeros((batch_size, 1, 1, 1), dtype=bool),
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return x_t + dt * v_t, time + dt

            def cond(carry):
                return carry[1] >= -dt / 2

            x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
            return x_0

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        rng, reac_rng = jax.random.split(rng, 2)
        base_noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        reac_noise = jax.random.normal(reac_rng, (batch_size, self.action_horizon, self.action_dim))
        return self.reactive_get_k_actions(
            observation,
            noise=base_noise,
            reac_noise=reac_noise,
            prefix_tokens=None,
            prefix_mask=None,
            prefix_ar_mask=None,
            dt=dt,
            kv_cache=None,
        )

    def reactive_replan(self, observation: _model.Observation, num_steps: int = 10):
        if not self.reactive_in_use:
            raise ValueError("reactive_replan requires reactive_in_use=True")
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        cross_attn_mask, cross_positions = self._cross_context(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            memory=None,
            cross_attn_mask=cross_attn_mask,
            cross_positions=cross_positions,
        )
        return self.action_horizon, prefix_tokens, prefix_mask, prefix_ar_mask, dt, kv_cache, self.action_dim

    def reactive_get_k_actions(
        self,
        observation: _model.Observation,
        noise,
        reac_noise,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        dt,
        kv_cache,
    ):
        if not self.reactive_in_use:
            raise ValueError("reactive_get_k_actions requires reactive_in_use=True")
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        wrench_memory = self._get_wrench_memory(observation)
        cross_attn_mask, cross_positions = self._cross_context(observation)

        if prefix_tokens is None:
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None, None],
                mask=prefix_attn_mask,
                positions=positions,
                memory=None,
                cross_attn_mask=cross_attn_mask,
                cross_positions=cross_positions,
            )

        def step(carry):
            x_t, reac_x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            reacfix_tokens, reacfix_mask, reacfix_ar_mask, reac_adarms_cond = self.embed_suffix(
                observation, reac_x_t, jnp.broadcast_to(time, batch_size)
            )

            input_mask = jnp.concatenate([prefix_mask, suffix_mask, reacfix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask, reacfix_ar_mask], axis=0)
            special_modules = (
                [1, self.action_horizon] if ((not self.pi05) and self.use_state) else [0, self.action_horizon]
            )
            full_attn_mask = make_reactive_attn_mask(input_mask, ar_mask, special_modules)[
                :, -suffix_tokens.shape[1] - reacfix_tokens.shape[1] :, :
            ]
            positions = jnp.concatenate(
                [
                    jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(reacfix_mask, axis=-1) - 1,
                    jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(reacfix_mask, axis=-1) - 1,
                ],
                axis=1,
            )

            (_, suffix_out, reacfix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens, reacfix_tokens],
                mask=full_attn_mask,
                adarms_cond=[None, adarms_cond, reac_adarms_cond],
                positions=positions,
                kv_cache=kv_cache,
                memory=wrench_memory,
                cross_attn_mask=cross_attn_mask,
                cross_positions=cross_positions,
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            reac_v_t = self.action_out_proj(reacfix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, reac_x_t + dt * reac_v_t, time + dt

        def cond(carry):
            return carry[2] >= -dt / 2

        x_0, reac_x_0, _ = jax.lax.while_loop(cond, step, (noise, reac_noise, 1.0))
        return x_0 if self.original_head else reac_x_0

    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_actions: jax.Array,
        inference_delay: int,
        prefix_attention_horizon: int,
        max_guidance_weight: float,
        *,
        num_steps: int = 10,
        prefix_attention_schedule: PrefixAttentionSchedule = "exp",
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        prefix_actions = prefix_actions[None, ...]
        prefix_actions = jnp.concatenate(
            [prefix_actions, jnp.zeros((batch_size, self.action_horizon, self.action_dim - prefix_actions.shape[-1]))],
            axis=2,
        )
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

        def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
            start = jnp.minimum(start, end)
            if schedule == "ones":
                w = jnp.ones(total)
            elif schedule == "zeros":
                w = (jnp.arange(total) < start).astype(jnp.float32)
            elif schedule in {"linear", "exp"}:
                w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
                if schedule == "exp":
                    w = w * jnp.expm1(w) / (jnp.e - 1)
            else:
                raise ValueError(f"Invalid schedule: {schedule}")
            return jnp.where(jnp.arange(total) >= end, 0, w)

        if not self.reactive_in_use:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
                memory=None,
                cross_positions=[jnp.zeros((batch_size, 1), dtype=jnp.int32)] * 2,
                cross_attn_mask=jnp.zeros((batch_size, 1, 1, 1), dtype=bool),
            )

            def pinv_corrected_velocity(v_t_fn: Callable[[jax.Array, float], jax.Array], x_t, t, p_actions):
                @jax.vmap
                def _single(x_ti: jax.Array, yi: jax.Array) -> jax.Array:
                    def denoiser(x):
                        v = v_t_fn(x, t)
                        return x - v * t, v

                    x_0, vjp_fun, v_t = jax.vjp(denoiser, x_ti, has_aux=True)
                    weights = get_prefix_weights(
                        inference_delay,
                        prefix_attention_horizon + inference_delay,
                        p_actions.shape[1],
                        prefix_attention_schedule,
                    )
                    error = (yi - x_0) * weights[:, None]
                    _ = vjp_fun(error)[0]
                    inv_r2 = (t**2 + (1 - t) ** 2) / (t**2)
                    c = jnp.nan_to_num(t / (1 - t), posinf=max_guidance_weight)
                    guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                    return v_t - guidance_weight * error

                return _single(x_t, p_actions)

            def v_t_step(x_t: jax.Array, time: jax.Array):
                x_t = x_t[None, ...]
                time = time[None, ...]
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                pref_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([pref_attn_mask, suffix_attn_mask], axis=-1)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                    memory=None,
                    cross_positions=[jnp.zeros((batch_size, 1), dtype=jnp.int32)] * 2,
                    cross_attn_mask=jnp.zeros((batch_size, 1, 1, 1), dtype=bool),
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return v_t[0, ...]

            def rtc_step(carry):
                x_t, time = carry
                guided_vt = pinv_corrected_velocity(v_t_step, x_t, time, prefix_actions)
                return x_t + dt * guided_vt, time + dt

            def cond(carry):
                return carry[1] >= -dt / 2

            x_0, _ = jax.lax.while_loop(cond, rtc_step, (noise, 1.0))
            return x_0

        # RTC with reactive mode: both base and reactive streams are denoised jointly.
        # RTC guidance is applied exclusively to the output head (controlled by original_head),
        # while the other stream is denoised without guidance in the same forward pass.
        rng, reac_rng = jax.random.split(rng, 2)
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        reac_noise = jax.random.normal(reac_rng, (batch_size, self.action_horizon, self.action_dim))

        wrench_memory = self._get_wrench_memory(observation)
        cross_attn_mask, cross_positions = self._cross_context(observation)
        special_modules = [1, self.action_horizon] if ((not self.pi05) and self.use_state) else [0, self.action_horizon]

        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            memory=None,
            cross_attn_mask=cross_attn_mask,
            cross_positions=cross_positions,
        )

        def v_t_step_both(x_t_single: jax.Array, reac_x_t_single: jax.Array, time_scalar: jax.Array):
            """Single LLM forward pass returning velocities for both base and reactive streams."""
            x_t_b = x_t_single[None, ...]
            reac_x_t_b = reac_x_t_single[None, ...]
            time_b = jnp.broadcast_to(time_scalar, (1,))

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t_b, time_b)
            reacfix_tokens, reacfix_mask, reacfix_ar_mask, reac_adarms_cond = self.embed_suffix(
                observation, reac_x_t_b, time_b
            )

            input_mask = jnp.concatenate([prefix_mask, suffix_mask, reacfix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask, reacfix_ar_mask], axis=0)
            full_attn_mask = make_reactive_attn_mask(input_mask, ar_mask, special_modules)[
                :, -suffix_tokens.shape[1] - reacfix_tokens.shape[1] :, :
            ]
            # Both experts share the same position encoding (matching reactive_get_k_actions)
            pos_both = jnp.concatenate(
                [
                    jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(reacfix_mask, axis=-1) - 1,
                    jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(reacfix_mask, axis=-1) - 1,
                ],
                axis=1,
            )
            (_, suffix_out, reacfix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens, reacfix_tokens],
                mask=full_attn_mask,
                adarms_cond=[None, adarms_cond, reac_adarms_cond],
                positions=pos_both,
                kv_cache=kv_cache,
                memory=wrench_memory,
                cross_attn_mask=cross_attn_mask,
                cross_positions=cross_positions,
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])[0]
            reac_v_t = self.action_out_proj(reacfix_out[:, -self.action_horizon :])[0]
            return v_t, reac_v_t

        def pinv_corrected_velocity_reactive(x_t, reac_x_t, t, p_actions):
            @jax.vmap
            def _single(x_ti: jax.Array, reac_x_ti: jax.Array, yi: jax.Array):
                weights = get_prefix_weights(
                    inference_delay,
                    prefix_attention_horizon + inference_delay,
                    p_actions.shape[1],
                    prefix_attention_schedule,
                )
                inv_r2 = (t**2 + (1 - t) ** 2) / (t**2)
                c = jnp.nan_to_num(t / (1 - t), posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)

                if self.original_head:
                    # Guide the base stream; reactive stream runs unguided in the same pass
                    def denoiser(x):
                        v, reac_v = v_t_step_both(x, reac_x_ti, t)
                        return x - v * t, (v, reac_v)

                    x_0, vjp_fun, (v_t, reac_v_t) = jax.vjp(denoiser, x_ti, has_aux=True)
                    error = (yi - x_0) * weights[:, None]
                    _ = vjp_fun(error)[0]
                    return v_t - guidance_weight * error, reac_v_t

                # Guide the reactive stream; base stream runs unguided in the same pass
                def denoiser(reac_x):
                    v, reac_v = v_t_step_both(x_ti, reac_x, t)
                    return reac_x - reac_v * t, (v, reac_v)

                reac_x_0, vjp_fun, (v_t, reac_v_t) = jax.vjp(denoiser, reac_x_ti, has_aux=True)
                error = (yi - reac_x_0) * weights[:, None]
                _ = vjp_fun(error)[0]
                return v_t, reac_v_t - guidance_weight * error

            return _single(x_t, reac_x_t, p_actions)

        def rtc_step_reactive(carry):
            x_t, reac_x_t, time = carry
            guided_v_t, guided_reac_v_t = pinv_corrected_velocity_reactive(x_t, reac_x_t, time, prefix_actions)
            return x_t + dt * guided_v_t, reac_x_t + dt * guided_reac_v_t, time + dt

        def cond_reactive(carry):
            return carry[2] >= -dt / 2

        x_0, reac_x_0, _ = jax.lax.while_loop(cond_reactive, rtc_step_reactive, (noise, reac_noise, 1.0))
        return x_0 if self.original_head else reac_x_0
