import numpy as np
import pytest

from openpi.training.residual_target_cache import ResidualTargetBuilder


def _identity_transform(item: dict) -> dict:
    return item


class _FakeRollout:
    def __init__(self, base_value: float):
        self.base_value = base_value
        self.calls = 0

    def __call__(self, model_state, observation):
        del model_state
        self.calls += 1
        batch_size = observation.state.shape[0]
        return np.full((batch_size, 2, 3), self.base_value, dtype=np.float32)


def _item(index: int, control_flag: tuple[float, float] = (-1.0, -1.0)) -> dict:
    return {
        "image": {"left_wrist_0_rgb": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"left_wrist_0_rgb": np.asarray(1, dtype=np.bool_)},
        "state": np.asarray([index, index + 1], dtype=np.float32),
        "actions": np.full((2, 3), index + 1, dtype=np.float32),
        "control_flag": np.asarray(control_flag, dtype=np.float32)[:, None],
    }


def _builder(rollout: _FakeRollout) -> ResidualTargetBuilder:
    return ResidualTargetBuilder(
        _identity_transform,
        rollout,
        batch_size=2,
        action_horizon=2,
        action_dim=3,
        action_scale=(0.5, 1.0, 2.0),
    )


def test_residual_target_builder_masks_and_normalizes_interventions():
    rollout = _FakeRollout(base_value=0.25)
    builder = _builder(rollout)
    items = [
        _item(0, (-1.0, 0.0)),
        _item(1, (0.0, 0.0)),
        _item(2, (0.0, -1.0)),
    ]

    targets = builder.build_online(None, items.__getitem__, range(3))

    np.testing.assert_allclose(targets[0, 0], [1.5, 0.75, 0.375])
    np.testing.assert_allclose(targets[0, 1], 0.0)
    np.testing.assert_allclose(targets[1], 0.0)
    np.testing.assert_allclose(targets[2, 0], 0.0)
    np.testing.assert_allclose(targets[2, 1], [5.5, 2.75, 1.375])
    assert rollout.calls == 2


def test_residual_target_builder_skips_rollout_without_interventions():
    rollout = _FakeRollout(base_value=999.0)
    builder = _builder(rollout)
    items = [_item(0, (0.0, 0.0)), _item(1, (0.0, 0.0))]

    targets = builder.build_online(None, items.__getitem__, range(2))

    np.testing.assert_array_equal(targets, 0.0)
    assert rollout.calls == 0


def test_residual_target_builder_requires_control_flag():
    rollout = _FakeRollout(base_value=0.0)
    builder = _builder(rollout)
    item = _item(0)
    del item["control_flag"]

    with pytest.raises(ValueError, match="control_flag"):
        builder.build_online(None, lambda _: item, range(1))
