"""Shared conventions for human-intervention recordings."""

import numpy as np

CONTROL_FLAG_FEATURE_NAME = "control_flag"
CONTROL_FLAG_FEATURE_SPEC = {"dtype": "float32", "shape": (1,), "names": [CONTROL_FLAG_FEATURE_NAME]}
DEFAULT_INTERVENTION_VALUE = -1.0


def intervention_mask(values, intervention_value: float = DEFAULT_INTERVENTION_VALUE) -> np.ndarray:
    """Select intervention timesteps without changing the recording's labels."""
    if not np.isfinite(intervention_value):
        raise ValueError("intervention_value must be finite")
    return np.isclose(np.asarray(values, dtype=np.float32), intervention_value)


def is_intervention_chunk(values, intervention_value: float = DEFAULT_INTERVENTION_VALUE) -> bool:
    mask = intervention_mask(values, intervention_value)
    return bool(mask.size and np.all(mask))
