import json
import sys

import pytest

from scripts import revalidate_lift
from scripts.revalidate_lift import _check_checkpoint


def test_public_task_versions():
    assert {task: dataset[0] for task, dataset in revalidate_lift.DATASETS.items()} == {
        "book": "book_insertion_v3_100",
        "hanoi": "hanoi_v1_100",
        "towel": "towelv3_100",
    }


def test_book_validation_does_not_require_other_tasks(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "chosen checkpoint" / "params"
    checkpoint.mkdir(parents=True)
    checked_datasets = []

    def check_dataset(dataset_root, *, online, action_horizon, expected_fps, intervention_value):
        checked_datasets.append(dataset_root)
        assert not online
        assert action_horizon == 10
        return {"path": str(dataset_root)}

    monkeypatch.setattr(revalidate_lift, "_check_dataset", check_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "revalidate_lift.py",
            "--tasks",
            "book",
            "--lerobot-root",
            str(tmp_path / "datasets"),
            "--checkpoint-root",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--asset-root",
            str(tmp_path),
        ],
    )
    revalidate_lift.main()
    result = json.loads(capsys.readouterr().out)
    assert checked_datasets == [tmp_path / "datasets/book_insertion_v3_100"]
    assert result["online"] == {}
    assert result["checkpoint"]["path"] == str(checkpoint)
    assert list(result["norm_stats_candidates"]) == ["book"]


def test_missing_requested_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        _check_checkpoint(tmp_path / "missing/params")
