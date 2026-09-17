import os
from pathlib import Path

import flax.linen as nn
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np

from openpi.shared.attention_visualization import save_attention_mask
import openpi.shared.jax_debug as _jax_debug

_MASK_IMAGE_PATH = Path(
    os.environ.get("OPENPI_RESIDUAL_CROSS_ATTN_MASK_PATH", "artifacts/residual_cross_attn_mask.png")
)


def _save_residual_cross_attn_mask(mask: np.ndarray) -> None:
    save_attention_mask(mask, _MASK_IMAGE_PATH)


def make_residual_cross_attn_mask(image_mask, wrench_mask, action_horizon: int):
    """Allow all valid image tokens and only causal wrench tokens."""

    batch_size = image_mask.shape[0]
    image_attn = jnp.broadcast_to(
        image_mask[:, None, :],
        (batch_size, action_horizon, image_mask.shape[1]),
    )

    causal_wrench_attn = jnp.tril(jnp.ones((action_horizon, action_horizon), dtype=jnp.bool_))
    causal_wrench_attn = jnp.broadcast_to(
        causal_wrench_attn[None, :, :],
        (batch_size, action_horizon, action_horizon),
    )
    causal_wrench_attn = jnp.logical_and(causal_wrench_attn, wrench_mask[:, None, None])
    cross_attn_mask = jnp.concatenate([image_attn, causal_wrench_attn], axis=-1)[:, None, :, :]
    if _jax_debug.enabled():
        jax.debug.callback(_save_residual_cross_attn_mask, cross_attn_mask[0, 0])
    return cross_attn_mask


def expand_action_scale(scale: tuple[float, ...], action_dim: int) -> tuple[float, ...]:
    """Pad a task-space residual scale to the model action dimension."""

    if not scale:
        raise ValueError("residual_action_scale must not be empty")
    if len(scale) > action_dim:
        raise ValueError(f"residual_action_scale has {len(scale)} values, but action_dim is {action_dim}")
    if any(value <= 0 for value in scale):
        raise ValueError("residual_action_scale values must be positive")
    return (*scale, *((1.0,) * (action_dim - len(scale))))


def _dropout(x, rng, rate: float, *, train: bool):
    if not train or rate == 0.0:
        return x
    keep_probability = 1.0 - rate
    keep = jax.random.bernoulli(rng, keep_probability, x.shape)
    return jnp.where(keep, x / keep_probability, 0).astype(x.dtype)


class _ResidualDecoderBlock(nn.Module):
    width: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float
    dtype: str

    @nn.compact
    def __call__(self, action_tokens, context, self_attn_mask, cross_attn_mask, dropout_rngs, *, train: bool):
        dtype = jnp.dtype(self.dtype)
        dense_init = nn.initializers.normal(stddev=0.02)

        hidden = nn.LayerNorm(dtype=dtype, name="self_attention_norm")(action_tokens)
        hidden = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.width,
            out_features=self.width,
            dtype=dtype,
            kernel_init=dense_init,
            dropout_rate=0.0,
            name="self_attention",
        )(hidden, hidden, mask=self_attn_mask, deterministic=True)
        action_tokens = action_tokens + _dropout(hidden, dropout_rngs[0], self.dropout_rate, train=train)

        query = nn.LayerNorm(dtype=dtype, name="cross_attention_query_norm")(action_tokens)
        memory = nn.LayerNorm(dtype=dtype, name="cross_attention_memory_norm")(context)
        hidden = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.width,
            out_features=self.width,
            dtype=dtype,
            kernel_init=dense_init,
            dropout_rate=0.0,
            name="cross_attention",
        )(query, memory, mask=cross_attn_mask, deterministic=True)
        action_tokens = action_tokens + _dropout(hidden, dropout_rngs[1], self.dropout_rate, train=train)

        hidden = nn.LayerNorm(dtype=dtype, name="mlp_norm")(action_tokens)
        hidden = nn.Dense(self.mlp_dim, dtype=dtype, kernel_init=dense_init, name="mlp_in")(hidden)
        hidden = nn.gelu(hidden)
        hidden = nn.Dense(self.width, dtype=dtype, kernel_init=dense_init, name="mlp_out")(hidden)
        return action_tokens + _dropout(hidden, dropout_rngs[2], self.dropout_rate, train=train)


class _LinenResidualPolicyCore(nn.Module):
    action_dim: int
    action_horizon: int
    image_token_count: int
    image_feature_dim: int
    width: int
    mlp_dim: int
    num_layers: int
    num_heads: int
    dropout_rate: float
    dtype: str

    @nn.compact
    def __call__(
        self,
        image_memory,
        image_mask,
        wrench_memory,
        wrench_mask,
        dropout_rng,
        *,
        train: bool = False,
    ):
        if image_memory.shape[1] != self.image_token_count:
            raise ValueError(
                f"Frozen SigLIP returned {image_memory.shape[1]} residual image tokens; "
                f"expected {self.image_token_count}."
            )

        dtype = jnp.dtype(self.dtype)
        dense_init = nn.initializers.normal(stddev=0.02)

        action_queries = self.param(
            "action_query_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.action_horizon, self.width),
        )
        action_tokens = jnp.broadcast_to(
            action_queries.astype(dtype),
            (image_memory.shape[0], self.action_horizon, self.width),
        )

        image_memory = nn.Dense(
            self.width,
            dtype=dtype,
            kernel_init=dense_init,
            name="condition_obs_proj",
        )(image_memory)
        condition_position = self.param(
            "condition_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.image_token_count + self.action_horizon, self.width),
        )
        image_context = image_memory + condition_position[:, : self.image_token_count].astype(dtype)
        image_context = nn.Dense(
            self.mlp_dim,
            dtype=dtype,
            kernel_init=dense_init,
            name="condition_mlp_in",
        )(image_context)
        image_context = image_context * jnp.tanh(jax.nn.softplus(image_context))
        image_context = nn.Dense(
            self.width,
            dtype=dtype,
            kernel_init=dense_init,
            name="condition_mlp_out",
        )(image_context)

        wrench_start = self.image_token_count
        wrench_memory = wrench_memory.astype(dtype) + condition_position[
            :, wrench_start : wrench_start + self.action_horizon
        ].astype(dtype)
        context = jnp.concatenate([image_context, wrench_memory], axis=1)

        self_attn_mask = jnp.tril(jnp.ones((self.action_horizon, self.action_horizon), dtype=jnp.bool_))
        self_attn_mask = jnp.broadcast_to(
            self_attn_mask[None, None, :, :],
            (image_memory.shape[0], 1, self.action_horizon, self.action_horizon),
        )
        cross_attn_mask = make_residual_cross_attn_mask(image_mask, wrench_mask, self.action_horizon)
        dropout_rngs = jax.random.split(dropout_rng, self.num_layers * 3)

        for layer in range(self.num_layers):
            action_tokens = _ResidualDecoderBlock(
                width=self.width,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name=f"decoder_block_{layer}",
            )(
                action_tokens,
                context,
                self_attn_mask,
                cross_attn_mask,
                dropout_rngs[layer * 3 : (layer + 1) * 3],
                train=train,
            )

        action_tokens = nn.LayerNorm(dtype=dtype, name="output_norm")(action_tokens)
        return nn.Dense(
            self.action_dim,
            dtype=dtype,
            kernel_init=nn.initializers.normal(stddev=0.001),
            bias_init=nn.initializers.zeros,
            name="action_out_proj",
        )(action_tokens)

    def init(self):
        self(
            jnp.zeros((1, self.image_token_count, self.image_feature_dim), dtype=jnp.dtype(self.dtype)),
            jnp.ones((1, self.image_token_count), dtype=jnp.bool_),
            jnp.zeros((1, self.action_horizon, self.width), dtype=jnp.dtype(self.dtype)),
            jnp.ones((1,), dtype=jnp.bool_),
            jax.random.key(0),
            train=False,
        )


def create_residual_policy_core(
    *,
    action_dim: int,
    action_horizon: int,
    image_token_count: int,
    image_feature_dim: int,
    width: int,
    mlp_dim: int,
    num_layers: int,
    num_heads: int,
    dropout_rate: float,
    dtype: str,
):
    return nnx_bridge.ToNNX(
        _LinenResidualPolicyCore(
            action_dim=action_dim,
            action_horizon=action_horizon,
            image_token_count=image_token_count,
            image_feature_dim=image_feature_dim,
            width=width,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dtype=dtype,
        )
    )
