"""Single-arm pose adaptation for real-time action-chunk continuation."""

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.shared import pose_utils

ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")
STATE_KEY = "observation/state"


def rebase_prefix_observation(
    previous_actions: np.ndarray, previous_state: np.ndarray | None, observation: dict
) -> dict:
    """Recover absolute poses before applying the current observation's transforms.

    States contain position and axis-angle rotation. Previous policy outputs
    contain position and xyz Euler rotation relative to the previous state.
    """
    state = np.asarray(observation[STATE_KEY])
    state = state[0] if state.ndim > 1 else state
    base_state = state if previous_state is None else np.asarray(previous_state)
    base_state = base_state[0] if base_state.ndim > 1 else base_state
    actions = np.asarray(previous_actions)
    if state.shape != (len(ACTION_NAMES),) or base_state.shape != state.shape:
        raise ValueError("Flexiv RTC requires single-arm 7D states.")
    if actions.ndim != 2 or actions.shape[-1] != len(ACTION_NAMES):
        raise ValueError("Flexiv RTC requires single-arm 7D action chunks.")
    relative = np.zeros(actions.shape[:-1] + (4, 4), dtype=actions.dtype)
    relative[..., :3, :3] = Rotation.from_euler("xyz", actions[..., 3:6]).as_matrix()
    relative[..., :3, 3] = actions[..., :3]
    relative[..., 3, 3] = 1.0
    base = np.broadcast_to(pose_utils.pose_to_mat(base_state[:6]), relative.shape)
    absolute = pose_utils.convert_pose_mat_rep(relative, base, pose_rep="relative", backward=True)
    result = dict(observation)
    result["actions"] = np.concatenate([pose_utils.mat_to_pose(absolute), actions[..., 6:7]], axis=-1)
    result[STATE_KEY] = state
    return result
