"""Roll out the frozen base policy on a small offline sample and plot action residuals."""

import argparse
import dataclasses
import logging
from pathlib import Path

import jax
import matplotlib as mpl

mpl.use("Agg")
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import matplotlib.pyplot as plt
import numpy as np
from train_online_dagger import init_train_state

import openpi.models.model as _model
from openpi.policies.flexiv_transforms import ACTION_NAMES
import openpi.training.config as _config
from openpi.training.residual_target_cache import BaseActionRollout
import openpi.training.sharding as _sharding
import openpi.transforms as _transforms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_iPhoneSingle_towelv3_100_residual")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset-repo", default="flexiv/towelv3_100")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/offline_rollout_test"),
    )
    return parser.parse_args()


def _stack_items(items: list[dict]) -> dict:
    return jax.tree.map(
        lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
        *items,
    )


def _make_transforms(config: _config.TrainConfig, data_config: _config.DataConfig):
    norm_stats = data_config.norm_stats or {}
    return _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def _physical_actions(actions: np.ndarray, stats, *, residual: bool = False) -> np.ndarray:
    actions = np.asarray(actions)[..., : len(ACTION_NAMES)]
    if stats is None:
        return actions
    if stats.q01 is not None and stats.q99 is not None:
        scale = np.asarray(stats.q99)[..., : actions.shape[-1]] - np.asarray(stats.q01)[..., : actions.shape[-1]]
        if residual:
            return actions * scale / 2.0
        return (actions + 1.0) * scale / 2.0 + np.asarray(stats.q01)[..., : actions.shape[-1]]
    scale = np.asarray(stats.std)[..., : actions.shape[-1]]
    if residual:
        return actions * scale
    return actions * scale + np.asarray(stats.mean)[..., : actions.shape[-1]]


def _plot_sample(
    output_dir: Path,
    sample_index: int,
    ground_truth: np.ndarray,
    base: np.ndarray,
    residual: np.ndarray,
) -> None:
    fig, axes = plt.subplots(len(ACTION_NAMES), 1, figsize=(10, 13), sharex=True)
    steps = np.arange(ground_truth.shape[0])
    for channel, (axis, name) in enumerate(zip(axes, ACTION_NAMES, strict=True)):
        axis.plot(steps, ground_truth[:, channel], label="GT", color="#1769aa", linewidth=1.8)
        axis.plot(steps, base[:, channel], label="Base", color="#d95f02", linewidth=1.5)
        axis.plot(steps, residual[:, channel], label="Residual target", color="#1b9e77", linewidth=1.5)
        axis.set_ylabel(name)
        axis.grid(visible=True, alpha=0.25)
        if channel == 0:
            axis.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Action chunk step")
    fig.suptitle(f"Offline sample {sample_index}: GT / base / residual target")
    fig.tight_layout()
    fig.savefig(output_dir / f"sample_{sample_index:04d}.png", dpi=130)
    plt.close(fig)


def _plot_overview(output_dir: Path, ground_truth: np.ndarray, base: np.ndarray, residual: np.ndarray) -> None:
    num_samples, action_horizon, _ = ground_truth.shape
    x = np.arange(num_samples * action_horizon)
    fig, axes = plt.subplots(len(ACTION_NAMES), 1, figsize=(14, 13), sharex=True)
    for channel, (axis, name) in enumerate(zip(axes, ACTION_NAMES, strict=True)):
        axis.plot(x, ground_truth[..., channel].reshape(-1), label="GT", color="#1769aa", linewidth=0.7)
        axis.plot(x, base[..., channel].reshape(-1), label="Base", color="#d95f02", linewidth=0.7)
        axis.plot(x, residual[..., channel].reshape(-1), label="Residual target", color="#1b9e77", linewidth=0.7)
        for boundary in range(action_horizon, len(x), action_horizon):
            axis.axvline(boundary, color="#999999", linewidth=0.25, alpha=0.35)
        axis.set_ylabel(name)
        axis.grid(visible=True, alpha=0.2)
        if channel == 0:
            axis.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Flattened sample / action-chunk step")
    fig.suptitle(f"First {num_samples} offline samples: GT / base / residual target")
    fig.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=140)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0 or args.num_steps <= 0:
        raise ValueError("num-samples, batch-size, and num-steps must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        fsdp_devices=1,
        weight_loader=_config.weight_loaders.CheckpointWeightLoader(params_path=args.checkpoint),
    )
    mesh = _sharding.make_mesh(1)
    rng = jax.random.key(config.seed)
    train_state, _ = init_train_state(config, rng, mesh, resume=False)
    jax.block_until_ready(train_state.params)
    logging.info("Loaded base model checkpoint: %s", args.checkpoint)

    data_config = config.data.create(config.assets_dirs, config.model)
    transform = _make_transforms(config, data_config)
    metadata = LeRobotDatasetMetadata(args.dataset_repo)
    action_horizon = config.model.action_horizon
    dataset = LeRobotDataset(
        args.dataset_repo,
        delta_timestamps={"actions": [t / metadata.fps for t in range(action_horizon)]},
    )
    num_samples = min(args.num_samples, len(dataset))
    items = []
    for index in range(num_samples):
        item = dataset[index]
        if data_config.prompt_from_task:
            item = _transforms.PromptFromLeRobotTask(metadata.tasks)(item)
        transformed = transform(dict(item))
        if transformed.get("wrench") is None:
            transformed["wrench"] = np.zeros((action_horizon, 6), dtype=np.float32)
            transformed["wrench_mask"] = np.asarray(0, dtype=np.bool_)
        else:
            transformed["wrench_mask"] = np.asarray(1, dtype=np.bool_)
        items.append(transformed)

    rollout = BaseActionRollout(
        train_state.model_def,
        mesh=mesh,
        input_sharding=None,
        num_steps=args.num_steps,
        seed=config.seed,
    )
    stats = data_config.norm_stats.get("actions") if data_config.norm_stats else None
    ground_truth_chunks = []
    base_chunks = []
    for start in range(0, num_samples, args.batch_size):
        stop = min(start + args.batch_size, num_samples)
        batch_items = items[start:stop]
        actual_size = len(batch_items)
        if actual_size < args.batch_size:
            batch_items = batch_items + [batch_items[-1]] * (args.batch_size - actual_size)
        batch = _stack_items(batch_items)
        base_actions = rollout(train_state.params, _model.Observation.from_dict(batch))
        ground_truth = np.asarray(batch["actions"], dtype=np.float32)
        ground_truth_chunks.append(ground_truth[:actual_size])
        base_chunks.append(base_actions[:actual_size])
        logging.info("Rolled out samples %d-%d/%d", start, stop - 1, num_samples)

    ground_truth = np.concatenate(ground_truth_chunks, axis=0)
    base = np.concatenate(base_chunks, axis=0)
    residual = ground_truth - base
    np.savez_compressed(
        output_dir / "actions_normalized.npz",
        sample_indices=np.arange(num_samples),
        ground_truth=ground_truth,
        base=base,
        residual_target=residual,
    )

    ground_truth_physical = _physical_actions(ground_truth, stats)
    base_physical = _physical_actions(base, stats)
    residual_physical = _physical_actions(residual, stats, residual=True)
    np.savez_compressed(
        output_dir / "actions_physical.npz",
        sample_indices=np.arange(num_samples),
        ground_truth=ground_truth_physical,
        base=base_physical,
        residual_target=residual_physical,
    )
    _plot_overview(output_dir, ground_truth_physical, base_physical, residual_physical)
    for index in range(num_samples):
        _plot_sample(
            output_dir,
            index,
            ground_truth_physical[index],
            base_physical[index],
            residual_physical[index],
        )

    mae = np.mean(np.abs(ground_truth_physical - base_physical), axis=(0, 1))
    summary_lines = ["action,mean_absolute_gt_minus_base"]
    summary_lines.extend(f"{name},{value:.8g}" for name, value in zip(ACTION_NAMES, mae, strict=True))
    (output_dir / "error_summary.csv").write_text("\n".join(summary_lines) + "\n")
    logging.info("Saved rollout arrays, overview, 100 sample plots, and error summary to %s", output_dir)


if __name__ == "__main__":
    main()
