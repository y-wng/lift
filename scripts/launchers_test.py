import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent


@pytest.fixture
def launcher_env(tmp_path):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENPI_", "REMOTE_", "SSH_"))
        and key not in {"CHECKPOINT_DIR", "SOURCE_DIR", "OUTPUT_PATH", "CONFIG_PATH"}
    }
    checkpoint = tmp_path / "checkpoint with spaces" / "params"
    checkpoint.mkdir(parents=True)
    environment.update(
        {
            "OPENPI_PYTHON": sys.executable,
            "OPENPI_DRY_RUN": "1",
            "OPENPI_INIT_CHECKPOINT": str(checkpoint),
            "OPENPI_BASE_INIT_CHECKPOINT": str(checkpoint),
        }
    )
    return environment


def run_launcher(name, environment, *arguments):
    return subprocess.run(
        ["bash", str(SCRIPTS_DIR / name), *arguments],
        env=environment,
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )


def command_args(result):
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Would run:")
    return shlex.split(result.stdout.removeprefix("Would run:"))


def test_converter_requires_explicit_input_and_output(launcher_env):
    result = run_launcher("nedf2_to_lerobot_incremental_flexiv_tdk.sh", launcher_env)
    assert result.returncode != 0
    assert "SOURCE_DIR" in result.stderr


def test_converter_uses_public_towel_preset(launcher_env, tmp_path):
    launcher_env.update({"SOURCE_DIR": str(tmp_path / "raw data"), "OUTPUT_PATH": str(tmp_path / "output")})
    arguments = command_args(run_launcher("nedf2_to_lerobot_incremental_flexiv_tdk.sh", launcher_env))
    assert str(SCRIPTS_DIR.parent / "preprocess_data/configs/towelv3_online.yaml") in arguments
    assert arguments[arguments.index("--source-dir") + 1] == launcher_env["SOURCE_DIR"]
    assert not Path(launcher_env["OUTPUT_PATH"]).exists()


def test_converter_rejects_same_source_and_output(launcher_env, tmp_path):
    launcher_env.update({"SOURCE_DIR": str(tmp_path), "OUTPUT_PATH": str(tmp_path / "nested" / "..")})
    result = run_launcher("nedf2_to_lerobot_incremental_flexiv_tdk.sh", launcher_env)
    assert result.returncode != 0
    assert "different directories" in result.stderr


@pytest.mark.parametrize("name", ["nedf2_to_lerobot_incremental_flexiv_tdk.sh"])
def test_utility_help_needs_no_private_configuration(name, launcher_env):
    result = run_launcher(name, launcher_env, "--help")
    assert result.returncode == 0
    assert "Usage:" in result.stdout


@pytest.mark.parametrize("suffix", ["", "_reactive", "_residual"])
def test_training_does_not_overwrite_by_default(suffix, launcher_env):
    arguments = command_args(run_launcher(f"train_online_dagger_lerobot{suffix}.sh", launcher_env))
    assert "--overwrite" not in arguments
    assert "--resume" not in arguments
    assert arguments[arguments.index("--config-name") + 1] == f"pi05_iPhoneSingle_book_insertion_v3_100{suffix}"


@pytest.mark.parametrize(("ratio", "fraction"), [("0to1", "1.0"), ("1to1", "0.5"), ("1to2", "0.6666666667")])
def test_ratio_wrapper_preserves_sampling_setting(ratio, fraction, launcher_env):
    arguments = command_args(run_launcher(f"train_online_dagger_lerobot_reactive_ratio_{ratio}.sh", launcher_env))
    assert arguments[arguments.index("--min-online-ratio") + 1] == fraction
    assert arguments[arguments.index("--max-online-ratio") + 1] == fraction


def test_force_ablation_still_reaches_python(launcher_env):
    arguments = command_args(run_launcher("train_online_dagger_lerobot_reactive_no_force_history.sh", launcher_env))
    assert "--disable-force-history" in arguments


def test_general_entrypoint_defaults_to_help(launcher_env):
    arguments = command_args(run_launcher("train_online_dagger.sh", launcher_env))
    assert arguments[-1] == "--help"


@pytest.mark.parametrize("suffix", ["", "_reactive", "_residual"])
def test_presets_preserve_gpu_visibility_and_forward_arguments(suffix, launcher_env):
    launcher_env["CUDA_VISIBLE_DEVICES"] = "5"
    arguments = command_args(run_launcher(f"train_online_dagger_lerobot{suffix}.sh", launcher_env, "--batch-size", "7"))
    assert "CUDA_VISIBLE_DEVICES=5" in arguments
    assert arguments[-2:] == ["--batch-size", "7"]
    launcher_env["OPENPI_CUDA_VISIBLE_DEVICES"] = "3"
    arguments = command_args(run_launcher(f"train_online_dagger_lerobot{suffix}.sh", launcher_env))
    assert "CUDA_VISIBLE_DEVICES=3" in arguments


def test_ratio_ablation_is_strict_and_uses_selected_task_name(launcher_env):
    launcher_env["OPENPI_CONFIG_NAME"] = "custom_task"
    arguments = command_args(run_launcher("train_online_dagger_lerobot_reactive_ratio_1to2.sh", launcher_env))
    assert "--no-allow-offline-warm-start" in arguments
    assert arguments[arguments.index("--exp-name") + 1].startswith("custom_task_ratio_1to2_")
    assert arguments[arguments.index("--task-description") + 1] == ""
