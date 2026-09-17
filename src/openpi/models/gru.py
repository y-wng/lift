from flax import linen as nn
import flax.nnx.bridge as nnx_bridge
import jax.numpy as jnp


class _LinenGRUCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x):
        h_prev = carry
        r = nn.sigmoid(nn.Dense(self.hidden_size, name="r_dense")(jnp.concatenate([x, h_prev], axis=-1)))
        z = nn.sigmoid(nn.Dense(self.hidden_size, name="z_dense")(jnp.concatenate([x, h_prev], axis=-1)))
        h_candidate = nn.tanh(nn.Dense(self.hidden_size, name="h_dense")(jnp.concatenate([x, r * h_prev], axis=-1)))
        h = (1 - z) * h_prev + z * h_candidate
        return h, h


class _LinenGRUNetwork(nn.Module):
    hidden_size: int
    output_size: int
    action_horizon: int

    @nn.compact
    def __call__(self, inputs, initial_state=None):
        batch_size, sequence_length, _ = inputs.shape
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.hidden_size))

        # Wrap GRUCell with flax.linen.scan so parameters are created once and shared
        # across the time dimension. We scan over axis=1 (seq_len) of inputs.
        scanned_gru = nn.scan(
            _LinenGRUCell,
            variable_broadcast={"params": True},
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
            length=sequence_length,
        )(hidden_size=self.hidden_size)

        _, hidden_states = scanned_gru(initial_state, inputs)
        outputs = nn.Dense(self.output_size)(hidden_states)
        return jnp.array(outputs, dtype=jnp.bfloat16)

    def init(self):
        self(jnp.zeros((1, self.action_horizon, 6)))


def create_gru_network(hidden_size: int, output_size: int, action_horizon: int):
    linen_mod = _LinenGRUNetwork(hidden_size=hidden_size, output_size=output_size, action_horizon=action_horizon)
    return nnx_bridge.ToNNX(linen_mod)
