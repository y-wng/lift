import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import base_policy

from openpi.models import model
from openpi.policies import single_iphone_flexiv_policy


class CountingPolicy(base_policy.BasePolicy):
    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return {"actions": np.arange(28).reshape(4, 7)}

    def reset(self):
        self.calls = 0


def test_single_arm_policy_input_output():
    example = single_iphone_flexiv_policy.make_single_iphone_flexiv_example()
    inputs = single_iphone_flexiv_policy.SingleiPhoneFlexivInputs(model_type=model.ModelType.PI05)(example)
    assert inputs["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert inputs["image_mask"]["left_wrist_0_rgb"]
    assert not inputs["image_mask"]["base_0_rgb"]
    np.testing.assert_array_equal(inputs["wrench"], example["wrench"])
    outputs = single_iphone_flexiv_policy.SingleiPhoneFlexivOutputs()({"actions": np.zeros((10, 32))})
    assert outputs["actions"].shape == (10, 7)


def test_action_chunk_broker_refresh_and_reset():
    policy = CountingPolicy()
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=2)
    first = broker.infer({})
    second = broker.infer({})
    assert policy.calls == 1
    np.testing.assert_array_equal(first["actions"], np.arange(7))
    np.testing.assert_array_equal(second["actions"], np.arange(7, 14))
    broker.infer({})
    assert policy.calls == 2
    broker.reset()
    assert policy.calls == 0
    np.testing.assert_array_equal(broker.infer({})["actions"], np.arange(7))
