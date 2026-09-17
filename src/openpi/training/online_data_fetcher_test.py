import numpy as np
import pytest

from openpi.training.online_data_fetcher import OnlineDataFetcher


def test_residual_fetcher_includes_and_requires_control_flag():
    fetcher = OnlineDataFetcher(
        datacloud_endpoint="",
        identifier="",
        features={"state": {"dtype": "float32", "shape": (7,), "names": ["state"]}},
        require_control_flag=True,
    )

    assert "control_flag" in fetcher.features
    with pytest.raises(ValueError, match="control_flag"):
        fetcher._validate_control_flags([{"state": np.zeros((2, 7), dtype=np.float32)}])


def test_non_residual_fetcher_keeps_legacy_optional_control_flag_behavior():
    fetcher = OnlineDataFetcher(datacloud_endpoint="", identifier="")

    fetcher._validate_control_flags([{"state": np.zeros((2, 7), dtype=np.float32)}])
