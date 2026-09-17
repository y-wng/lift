"""Evaluate a residual-only checkpoint against offline and online LeRobot data.

The residual checkpoint is overlaid on the frozen base policy.  The
saved physical arrays are in the task-space representation produced by the
single-iPhone transform: x/y/z in metres, rx/ry/rz in radians, and gripper in
the dataset's native unit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from flax import nnx
import jax
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np

import openpi.models.model as _model
import openpi.models.residual_policy as _residual_policy
from openpi.policies.flexiv_transforms import ACTION_NAMES
import openpi.training.config as _config
from openpi.training.initialization import init_train_state
import openpi.training.sharding as _sharding
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def evaluation_action_scale(model_config) -> np.ndarray:
    """Validate the evaluator schema and use the model's configured scale."""
    if not model_config.residual_policy_in_use:
        raise ValueError("Residual evaluation requires a residual-policy config.")
    if len(model_config.residual_action_scale) != len(ACTION_NAMES):
        raise ValueError("This evaluator supports only the single-arm 7D action schema.")
    return np.asarray(model_config.residual_action_scale, dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_iPhoneSingle_towelv3_100_residual")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--offline-dataset", required=True)
    parser.add_argument("--online-dataset", required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--repeat-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--spread",
        action="store_true",
        help="Select samples evenly across the full dataset instead of the first consecutive frames.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _stack(items: list[dict]) -> dict:
    return jax.tree.map(
        lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
        *items,
    )


def _physical_delta(normalized: np.ndarray, stats) -> np.ndarray:
    if stats is None:
        return normalized
    if stats.q01 is not None and stats.q99 is not None:
        scale = (
            np.asarray(stats.q99, dtype=np.float32)[: normalized.shape[-1]]
            - np.asarray(stats.q01, dtype=np.float32)[: normalized.shape[-1]]
        )
        return normalized * scale / 2.0
    return normalized * np.asarray(stats.std, dtype=np.float32)[: normalized.shape[-1]]


def _physical_absolute(normalized: np.ndarray, stats) -> np.ndarray:
    if stats is None:
        return normalized
    if stats.q01 is not None and stats.q99 is not None:
        q01 = np.asarray(stats.q01, dtype=np.float32)[: normalized.shape[-1]]
        q99 = np.asarray(stats.q99, dtype=np.float32)[: normalized.shape[-1]]
        return (normalized + 1.0) * (q99 - q01) / 2.0 + q01
    mean = np.asarray(stats.mean, dtype=np.float32)[: normalized.shape[-1]]
    std = np.asarray(stats.std, dtype=np.float32)[: normalized.shape[-1]]
    return normalized * std + mean


def _metrics(pred: np.ndarray, target: np.ndarray, *, source: str) -> dict:
    error = pred - target
    flat_pred = pred.reshape(-1, pred.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    result = {
        "source": source,
        "num_samples": int(pred.shape[0]),
        "num_action_steps": int(pred.shape[0] * pred.shape[1]),
        "mae": np.mean(np.abs(error), axis=(0, 1)).tolist(),
        "rmse": np.sqrt(np.mean(np.square(error), axis=(0, 1))).tolist(),
        "bias": np.mean(error, axis=(0, 1)).tolist(),
        "target_mae_from_zero": np.mean(np.abs(target), axis=(0, 1)).tolist(),
        "target_rms": np.sqrt(np.mean(np.square(target), axis=(0, 1))).tolist(),
        "pred_abs_mean": np.mean(np.abs(pred), axis=(0, 1)).tolist(),
        "sign_agreement": np.mean(np.sign(pred) == np.sign(target), axis=(0, 1)).tolist(),
        "action_names": list(ACTION_NAMES),
    }
    corr = []
    for dim in range(flat_pred.shape[-1]):
        x, y = flat_pred[:, dim], flat_target[:, dim]
        corr.append(float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-8 and np.std(y) > 1e-8 else 0.0)
    result["pearson"] = corr
    result["mae_over_zero_baseline"] = (
        np.asarray(result["mae"]) / (np.asarray(result["target_mae_from_zero"]) + 1e-8)
    ).tolist()
    return result


def _trajectory_metrics(pred: np.ndarray, target: np.ndarray, sample_indices: np.ndarray) -> dict:
    """Metrics on executed first action of consecutive frames, split by episodes."""
    pred_first, target_first = pred[:, 0], target[:, 0]
    deltas_pred, deltas_target = [], []
    second_pred, second_target = [], []
    for i in range(1, len(sample_indices)):
        if sample_indices[i, 0] != sample_indices[i - 1, 0] or sample_indices[i, 1] != sample_indices[i - 1, 1] + 1:
            continue
        deltas_pred.append(pred_first[i] - pred_first[i - 1])
        deltas_target.append(target_first[i] - target_first[i - 1])
        if (
            i >= 2
            and sample_indices[i - 1, 0] == sample_indices[i - 2, 0]
            and sample_indices[i - 1, 1] == sample_indices[i - 2, 1] + 1
        ):
            second_pred.append(pred_first[i] - 2 * pred_first[i - 1] + pred_first[i - 2])
            second_target.append(target_first[i] - 2 * target_first[i - 1] + target_first[i - 2])
    if not deltas_pred:
        return {"consecutive_pairs": 0}
    dp, dt = np.asarray(deltas_pred), np.asarray(deltas_target)
    result = {
        "consecutive_pairs": len(dp),
        "first_action_diff_abs_mean": np.mean(np.abs(dp), axis=0).tolist(),
        "first_action_diff_rms": np.sqrt(np.mean(np.square(dp), axis=0)).tolist(),
        "target_first_action_diff_abs_mean": np.mean(np.abs(dt), axis=0).tolist(),
        "target_first_action_diff_rms": np.sqrt(np.mean(np.square(dt), axis=0)).tolist(),
        "action_names": list(ACTION_NAMES),
    }
    if second_pred:
        sp, st = np.asarray(second_pred), np.asarray(second_target)
        result["first_action_second_diff_rms"] = np.sqrt(np.mean(np.square(sp), axis=0)).tolist()
        result["target_first_action_second_diff_rms"] = np.sqrt(np.mean(np.square(st), axis=0)).tolist()
    return result


def main() -> None:
    args = _parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0 or args.num_steps <= 0 or args.repeat_samples < 0:
        raise ValueError("num-samples, batch-size, num-steps must be positive; repeat-samples must be non-negative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _config.get_config(args.config_name)
    task_action_scale = evaluation_action_scale(config.model)
    config = dataclasses.replace(
        config,
        fsdp_devices=1,
        weight_loader=_weight_loaders.OverlayWeightLoader(
            base_loader=_weight_loaders.CheckpointWeightLoader(args.base_checkpoint),
            overlay_params_path=args.residual_checkpoint,
        ),
    )
    mesh = _sharding.make_mesh(1)
    train_state, _ = init_train_state(config, jax.random.key(config.seed), mesh, resume=False)
    jax.block_until_ready(train_state.params)
    logging.info("Loaded base=%s residual=%s", args.base_checkpoint, args.residual_checkpoint)

    data_config = config.data.create(config.assets_dirs, config.model)
    transform = _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    stats = data_config.norm_stats.get("actions") if data_config.norm_stats else None
    action_horizon = config.model.action_horizon
    action_scale = np.asarray(
        _residual_policy.expand_action_scale(tuple(task_action_scale), config.model.action_dim), dtype=np.float32
    )
    model_def = train_state.model_def

    def infer(params, rng, observation):
        m = nnx.merge(model_def, params)
        observation = _model.preprocess_observation(None, observation, train=False)
        base_rng, residual_rng = jax.random.split(rng)
        base_actions, prefix_tokens, _ = m._sample_base_actions_with_prefix(  # noqa: SLF001
            base_rng, observation, num_steps=args.num_steps
        )
        image_memory, image_mask = m._get_residual_image_memory(observation, prefix_tokens)  # noqa: SLF001
        residual_normalized = m._sample_residual_actions(  # noqa: SLF001
            residual_rng,
            image_memory,
            image_mask,
            observation.wrench,
            observation.wrench_mask,
            num_steps=args.num_steps,
        )
        return base_actions, residual_normalized, residual_normalized * jax.numpy.asarray(action_scale)

    infer_jit = jax.jit(infer)

    def evaluate_dataset(path: str, source: str) -> dict:
        metadata = LeRobotDatasetMetadata(path)
        delta_timestamps = {"actions": [t / metadata.fps for t in range(action_horizon)]}
        if "left_wrench" in metadata.features:
            delta_timestamps["left_wrench"] = [t / metadata.fps for t in range(action_horizon)]
        dataset = LeRobotDataset(path, delta_timestamps=delta_timestamps)
        count = min(args.num_samples, len(dataset))
        selected_indices = (
            np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist() if args.spread else list(range(count))
        )
        items, sample_indices = [], []
        for index in selected_indices:
            item = dataset[index]
            if data_config.prompt_from_task:
                item = _transforms.PromptFromLeRobotTask(metadata.tasks)(item)
            transformed = transform(dict(item))
            if transformed.get("wrench") is None:
                transformed["wrench"] = np.zeros((action_horizon, 6), dtype=np.float32)
                transformed["wrench_mask"] = np.zeros((), dtype=np.bool_)
            else:
                transformed["wrench_mask"] = np.ones((), dtype=np.bool_)
            items.append(transformed)
            sample_indices.append((int(item["episode_index"]), int(item["frame_index"])))

        base_chunks, residual_chunks, residual_scaled_chunks, gt_chunks = [], [], [], []
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            actual = stop - start
            batch_items = items[start:stop]
            if actual < args.batch_size:
                batch_items += [batch_items[-1]] * (args.batch_size - actual)
            batch = _stack(batch_items)
            outputs = infer_jit(
                train_state.params,
                jax.random.fold_in(jax.random.key(args.seed), start),
                _model.Observation.from_dict(batch),
            )
            base_chunks.append(np.asarray(outputs[0])[:actual, :, : len(ACTION_NAMES)])
            residual_chunks.append(np.asarray(outputs[1])[:actual, :, : len(ACTION_NAMES)])
            residual_scaled_chunks.append(np.asarray(outputs[2])[:actual, :, : len(ACTION_NAMES)])
            gt_chunks.append(np.asarray(batch["actions"])[:actual, :, : len(ACTION_NAMES)])
            logging.info("%s samples %d-%d/%d", source, start, stop - 1, count)

        base_norm = np.concatenate(base_chunks)
        residual_norm = np.concatenate(residual_chunks)
        residual_scaled_norm = np.concatenate(residual_scaled_chunks)
        gt_norm = np.concatenate(gt_chunks)
        target_scaled_norm = gt_norm - base_norm
        target_residual_norm = target_scaled_norm / task_action_scale
        base_physical = _physical_absolute(base_norm, stats)
        gt_physical = _physical_absolute(gt_norm, stats)
        pred_physical = _physical_delta(residual_scaled_norm, stats)
        target_physical = _physical_delta(target_scaled_norm, stats)
        indices = np.asarray(sample_indices, dtype=np.int64)
        arrays = {
            "sample_indices": indices,
            "ground_truth_normalized": gt_norm,
            "base_normalized": base_norm,
            "residual_pred_normalized": residual_norm,
            "residual_pred_scaled_normalized": residual_scaled_norm,
            "residual_target_normalized": target_residual_norm,
            "residual_target_scaled_normalized": target_scaled_norm,
            "ground_truth_physical": gt_physical,
            "base_physical": base_physical,
            "residual_pred_physical": pred_physical,
            "residual_target_physical": target_physical,
        }
        np.savez_compressed(args.output_dir / f"{source}_arrays.npz", **arrays)
        result = {
            "source": source,
            "dataset": path,
            "metrics_physical": _metrics(pred_physical, target_physical, source=source),
            "trajectory_physical": _trajectory_metrics(pred_physical, target_physical, indices),
            "target_physical_min": target_physical.min(axis=(0, 1)).tolist(),
            "target_physical_max": target_physical.max(axis=(0, 1)).tolist(),
            "pred_physical_min": pred_physical.min(axis=(0, 1)).tolist(),
            "pred_physical_max": pred_physical.max(axis=(0, 1)).tolist(),
            "normalized_metrics": _metrics(residual_scaled_norm, target_scaled_norm, source=source),
        }
        if args.repeat_samples:
            repeat_count = min(args.repeat_samples, count)
            repeat_items = items[:repeat_count]
            repeat_batch = _stack(repeat_items)
            repeat_outputs = []
            for repeat in range(3):
                out = infer_jit(
                    train_state.params,
                    jax.random.fold_in(jax.random.key(args.seed + 100_000), repeat),
                    _model.Observation.from_dict(repeat_batch),
                )
                repeat_outputs.append(np.asarray(out[1])[:repeat_count, :, : len(ACTION_NAMES)])
            repeat_pred = np.stack(repeat_outputs, axis=0)
            repeat_physical = _physical_delta(repeat_pred * task_action_scale, stats)
            result["repeat_sampling_physical_std_mean"] = np.mean(np.std(repeat_physical, axis=0), axis=(0, 1)).tolist()
            result["repeat_sampling_physical_std_max"] = np.max(np.std(repeat_physical, axis=0), axis=(0, 1)).tolist()
        return result

    results = [
        evaluate_dataset(args.offline_dataset, "offline"),
        evaluate_dataset(args.online_dataset, "online"),
    ]
    (args.output_dir / "summary.json").write_text(
        json.dumps({"checkpoint": args.residual_checkpoint, "results": results}, indent=2)
    )
    with (args.output_dir / "metrics.csv").open("w") as f:
        f.write("source,action,mae,rmse,bias,target_mae,pearson,sign_agreement\n")
        for result in results:
            metrics = result["metrics_physical"]
            for i, action in enumerate(ACTION_NAMES):
                f.write(
                    ",".join(
                        map(
                            str,
                            [
                                result["source"],
                                action,
                                metrics["mae"][i],
                                metrics["rmse"][i],
                                metrics["bias"][i],
                                metrics["target_mae_from_zero"][i],
                                metrics["pearson"][i],
                                metrics["sign_agreement"][i],
                            ],
                        )
                    )
                    + "\n"
                )
    logging.info("Wrote evaluation results to %s", args.output_dir)


if __name__ == "__main__":
    main()
