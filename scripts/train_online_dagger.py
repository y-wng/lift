"""
Online DAgger training script for OpenPi.

This script implements online DAgger training with adaptive sampling between
offline and online data, fetching new episodes from a data cloud service.

"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import functools
import logging
import platform

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
from openpi.shared.episode_schema import CONTROL_FLAG_FEATURE_NAME
from openpi.shared.episode_schema import CONTROL_FLAG_FEATURE_SPEC
import openpi.shared.jax_debug as _jax_debug
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.initialization import init_train_state
from openpi.training.initialization import maybe_copy_reactive_expert_params
from openpi.training.online_data_fetcher import OnlineDataFetcher
import openpi.training.residual_logging as _residual_logging
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


class _PrefetchIterator:
    """Keep data preparation ahead of the device train step.

    The iterator owns a single producer thread so the underlying dataset iterator is
    never advanced concurrently. A small future queue overlaps image decoding and
    transforms with the previous JAX step without changing batch shapes.
    """

    def __init__(self, iterable, depth: int):
        if depth < 0:
            raise ValueError(f"prefetch depth must be non-negative, got {depth}")
        self._iterator = iter(iterable)
        self._depth = depth
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-data-prefetch") if depth else None
        self._futures = deque()
        self._closed = False
        for _ in range(depth):
            self._submit()

    def _submit(self) -> None:
        if self._executor is not None and not self._closed:
            self._futures.append(self._executor.submit(next, self._iterator))

    def __iter__(self):
        return self

    def __next__(self):
        if not self._futures:
            return next(self._iterator)
        future = self._futures.popleft()
        try:
            value = future.result()
        except StopIteration:
            self.close()
            raise
        self._submit()
        return value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    online_config: _config.OnlineDaggerTrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        combined_config = {
            **dataclasses.asdict(config),
            "online_dagger": dataclasses.asdict(online_config),
        }
        wandb.init(
            name=f"{config.exp_name}_dagger",
            config=combined_config,
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _update_ema_params(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    new_params: nnx.State,
) -> nnx.State | None:
    if state.ema_decay is None:
        return None
    if state.ema_params is None:
        raise ValueError("ema_params must be initialized when ema_decay is set")

    def update_state(old_state: nnx.State, current_state: nnx.State) -> nnx.State:
        current_by_path = current_state.flat_state()
        return old_state.map(
            lambda path, old: old.replace(
                value=state.ema_decay * old.value + (1 - state.ema_decay) * current_by_path[path].value
            )
        )

    if not getattr(config.model, "residual_policy_in_use", False):
        return jax.tree.map(
            lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
            state.ema_params,
            new_params,
        )

    # Frozen base parameters are identical in old/new state. Keep those leaves
    # unchanged and run the EMA arithmetic only over the residual parameter tree.
    trainable_ema = state.ema_params.filter(config.trainable_filter)
    frozen_ema = state.ema_params.filter(nnx.Not(config.trainable_filter))
    trainable_params = new_params.filter(config.trainable_filter)
    updated_trainable_ema = update_state(trainable_ema, trainable_params)
    return nnx.State.merge(frozen_ema, updated_trainable_ema)


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(new_state, ema_params=_update_ema_params(config, state, new_params))

    if getattr(config.model, "residual_policy_in_use", False):
        # Residual training only updates the small residual module. Avoid walking
        # the frozen base model just to compute a diagnostic norm every step.
        kernel_params = new_params.filter(config.trainable_filter)
    else:
        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    _jax_debug.inspect(
        "train/metrics",
        info["loss"],
        info["grad_norm"],
        info["param_norm"],
        names=("loss", "grad_norm", "param_norm"),
    )
    return new_state, info


@at.typecheck
def train_step_with_per_sample_loss(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    is_online: at.Bool[at.Array, " b"],
) -> tuple[training_utils.TrainState, dict[str, at.Array], at.Array]:
    """Train step that also returns per-sample losses for adaptive sampling."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if getattr(config.model, "residual_policy_in_use", False):
            chunked_loss, residual_prediction = model.compute_residual_loss_with_prediction(
                rng, observation, actions, train=True
            )
            residual_metrics = _residual_logging.residual_prediction_metrics(
                residual_prediction,
                actions,
                is_online,
                config.model.residual_action_scale,
            )
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            residual_metrics = {}
        return jnp.mean(chunked_loss), (jnp.mean(chunked_loss, axis=-1), residual_metrics)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, (per_sample_loss, residual_metrics)), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(new_state, ema_params=_update_ema_params(config, state, new_params))

    if getattr(config.model, "residual_policy_in_use", False):
        kernel_params = new_params.filter(config.trainable_filter)
    else:
        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    if getattr(config.model, "residual_policy_in_use", False):
        info.update(
            residual_core_grad_norm=optax.global_norm(grads.filter(nnx_utils.PathRegex(".*residual_policy/core.*"))),
            residual_output_grad_norm=optax.global_norm(
                grads.filter(nnx_utils.PathRegex(".*residual_policy/core/action_out_proj.*"))
            ),
            residual_wrench_grad_norm=optax.global_norm(
                grads.filter(nnx_utils.PathRegex(".*residual_policy/wrench_gru.*"))
            ),
        )
        info.update(residual_metrics)
    _jax_debug.inspect(
        "train/metrics",
        *info.values(),
        names=tuple(info),
    )
    info["batch_online_samples"] = jnp.sum(is_online)
    info["batch_total_samples"] = jnp.asarray(is_online.size)
    return new_state, info, per_sample_loss


def _configure_init_weight_loader(
    config: _config.TrainConfig, args: _config.OnlineDaggerTrainConfig
) -> _config.TrainConfig:
    """Apply checkpoint initialization options to a training config.

    Residual checkpoints intentionally contain only the trainable residual tree.
    When both paths are supplied, initialize the frozen base tree first and overlay
    the residual tree on top. The legacy single-checkpoint flags remain supported
    for existing launch scripts.
    """

    is_residual = getattr(config.model, "residual_policy_in_use", False)
    has_split_paths = args.base_init_checkpoint is not None or args.residual_init_checkpoint is not None

    if not is_residual:
        if has_split_paths or args.init_checkpoint_is_residual:
            raise ValueError("--base-init-checkpoint/--residual-init-checkpoint are only valid for residual policies.")
        if args.init_checkpoint is None:
            return config
        logging.info("Using complete init checkpoint directly: %s", args.init_checkpoint)
        return dataclasses.replace(
            config,
            weight_loader=_weight_loaders.CheckpointWeightLoader(params_path=args.init_checkpoint),
        )

    if args.init_checkpoint is not None and has_split_paths:
        raise ValueError("Use either --init-checkpoint or the split residual checkpoint flags, not both.")

    base_path = args.base_init_checkpoint
    residual_path = args.residual_init_checkpoint
    if args.init_checkpoint is not None:
        if args.init_checkpoint_is_residual:
            residual_path = args.init_checkpoint
        else:
            # Legacy residual runs passed a complete reactive checkpoint here.
            base_path = args.init_checkpoint

    base_loader = config.weight_loader
    if base_path is not None:
        base_loader = _weight_loaders.CheckpointWeightLoader(params_path=base_path)
        logging.info("Using residual base-policy checkpoint: %s", base_path)

    if residual_path is not None:
        logging.info("Overlaying residual-policy checkpoint: %s", residual_path)
        base_loader = _weight_loaders.OverlayWeightLoader(
            base_loader=base_loader,
            overlay_params_path=residual_path,
        )
    elif args.init_checkpoint_is_residual:
        raise AssertionError("Residual init path should have been set from --init-checkpoint.")

    return dataclasses.replace(config, weight_loader=base_loader)


def main(args: _config.OnlineDaggerTrainConfig):
    init_logging()
    logging.info(f"Running Online DAgger on: {platform.node()}")

    # Get base training config
    config = _config.get_config(args.config_name)
    if args.exp_name:
        config = dataclasses.replace(config, exp_name=args.exp_name)
    if args.checkpoint_base_dir:
        config = dataclasses.replace(config, checkpoint_base_dir=args.checkpoint_base_dir)
    if args.overwrite:
        config = dataclasses.replace(config, overwrite=True)
    if args.resume:
        config = dataclasses.replace(config, resume=True)
    config = _configure_init_weight_loader(config, args)
    if args.prefetch_batches < 0:
        raise ValueError("--prefetch-batches must be non-negative")
    if args.sampler_update_interval <= 0:
        raise ValueError("--sampler-update-interval must be positive")
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if _jax_debug.enabled():
        logging.warning(
            "JAX debug callbacks are enabled; host-side summaries can serialize each train step "
            "and lower GPU utilization."
        )
    if args.save_interval is not None:
        config = dataclasses.replace(config, save_interval=args.save_interval)
    if args.log_interval is not None:
        config = dataclasses.replace(config, log_interval=args.log_interval)
    if args.num_train_steps is not None:
        config = dataclasses.replace(config, num_train_steps=args.num_train_steps)
    if args.online_use_wrench:
        config = dataclasses.replace(config, online_use_wrench=True)
    if args.disable_force_history:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, use_force_history=False),
        )

    # Use online dagger config directly from args
    online_config = args
    if args.fsdp_devices is not None:
        config = dataclasses.replace(config, fsdp_devices=args.fsdp_devices)
    if not args.allow_offline_warm_start and not (args.local_lerobot_data_root or args.datacloud_endpoint):
        raise ValueError("This ablation requires an online data source; offline warm starts are disabled.")

    # Initialize data fetcher
    data_fetcher = OnlineDataFetcher(
        datacloud_endpoint=online_config.datacloud_endpoint,
        identifier=online_config.identifier,
        query_filter=online_config.query_filter,
        local_lerobot_data_root=online_config.local_lerobot_data_root,
        robot_type=online_config.robot_type,
        fps=online_config.fps,
        task_description=online_config.task_description,
        features=online_config.features,
        use_absolute_action=online_config.use_absolute_action,
        action_type=online_config.action_type,
        temporal_downsample_ratio=online_config.temporal_downsample_ratio,
        use_dino=online_config.use_dino,
        episode_clip_head_seconds=online_config.episode_clip_head_seconds,
        episode_clip_tail_seconds=online_config.episode_clip_tail_seconds,
        gripper_width_bias=online_config.gripper_width_bias,
        gripper_width_scale=online_config.gripper_width_scale,
        # Residual targets are masked by human-intervention control flags. Keep
        # the field in every online episode and fail at fetch time if it is absent.
        require_control_flag=online_config.online_intervention_only or config.model.residual_policy_in_use,
    )
    if online_config.online_intervention_only:
        data_fetcher.features.setdefault(CONTROL_FLAG_FEATURE_NAME, dict(CONTROL_FLAG_FEATURE_SPEC))
        logging.info("Online intervention filter enabled: control_flag == %s", online_config.intervention_value)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, online_config, resuming=resuming, enabled=config.wandb_enabled)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    train_state = maybe_copy_reactive_expert_params(
        train_state,
        enabled=(
            config.copy_reactive_from_action_expert
            and not resuming
            and not getattr(config.model, "residual_policy_in_use", False)
        ),
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # Create hybrid data loader
    data_loader = _data_loader.create_hybrid_data_loader(
        config,
        online_repo_id=online_config.online_repo_id,
        sharding=data_sharding,
        window_size=online_config.window_size,
        boost_factor=online_config.boost_factor,
        min_online_ratio=online_config.min_online_ratio,
        max_online_ratio=online_config.max_online_ratio,
        initial_online_weight=online_config.initial_online_weight,
        robot_type=online_config.robot_type,
        fps=online_config.fps,
        features=online_config.features,
        online_intervention_only=online_config.online_intervention_only,
        intervention_value=online_config.intervention_value,
        allow_offline_warm_start=online_config.allow_offline_warm_start,
        residual_model_def=train_state.model_def if getattr(config.model, "residual_policy_in_use", False) else None,
        residual_model_state=train_state.params if getattr(config.model, "residual_policy_in_use", False) else None,
        residual_mesh=mesh if getattr(config.model, "residual_policy_in_use", False) else None,
        residual_target_num_steps=online_config.residual_target_num_steps,
        residual_target_batch_size=online_config.residual_target_batch_size,
    )
    data_config = data_loader.data_config()
    _jax_debug.configure_action_normalization(
        data_config.norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
    initial_episodes = data_fetcher.fetch_new_episodes()
    if initial_episodes:
        data_loader.dataset.append_episodes(
            initial_episodes,
            task=online_config.task_description,
            residual_model_state=train_state.params if config.model.residual_policy_in_use else None,
        )
    if not online_config.allow_offline_warm_start and data_loader.dataset.online_len == 0:
        raise ValueError(
            "No eligible online samples were found; verify intervention labels and completed episode markers."
        )
    data_iter = _PrefetchIterator(data_loader, args.prefetch_batches)
    batch_with_flags = next(data_iter)

    # Unpack batch (observation, actions, is_online)
    if len(batch_with_flags) == 3:
        observation, actions, is_online = batch_with_flags
    else:
        observation, actions = batch_with_flags
        is_online = jnp.zeros(actions.shape[0], dtype=jnp.bool_)

    batch = (observation, actions)
    logging.info(f"Initialized hybrid data loader:\n{training_utils.array_tree_to_info(batch)}")

    if resuming:
        train_state = _checkpoints.restore_state(
            checkpoint_manager,
            train_state,
            data_loader,
            params_filter=config.trainable_filter if getattr(config.model, "residual_policy_in_use", False) else None,
        )

    ptrain_step = jax.jit(
        functools.partial(train_step_with_per_sample_loss, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, data_sharding),
        # Per-sample losses follow the batch sharding; replicating them would add
        # a cross-device collective to every step just for adaptive sampling.
        out_shardings=(train_state_sharding, replicated_sharding, data_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    hybrid_dataset = data_loader.dataset
    last_batch_online_samples = 0
    last_batch_offline_samples = 0
    last_batch_online_ratio = 0.0
    log_online_samples = 0
    log_online_batches = 0
    total_sampled_online_samples = 0
    total_sampled_online_batches = 0

    for step in pbar:
        # Fetch new episodes periodically
        if step > 0 and step % online_config.fetch_interval == 0:
            new_episodes = data_fetcher.fetch_new_episodes()
            if new_episodes:
                hybrid_dataset.append_episodes(
                    new_episodes,
                    task=online_config.task_description,
                    residual_model_state=(
                        train_state.params if getattr(config.model, "residual_policy_in_use", False) else None
                    ),
                )
                logging.info(f"Added {len(new_episodes)} new episodes at step {step}")

        with sharding.set_mesh(mesh):
            train_state, info, per_sample_loss = ptrain_step(train_rng, train_state, batch, is_online)
        infos.append(info)

        # Update adaptive sampler weights periodically. Converting per-sample loss
        # and source flags to NumPy synchronizes the device, so doing it every step
        # can leave the accelerator idle while the host updates the sampler.
        if is_online is not None and (
            step % online_config.sampler_update_interval == 0 or step == config.num_train_steps - 1
        ):
            is_online_np = np.array(jax.device_get(is_online))
            per_sample_loss_np = np.asarray(jax.device_get(per_sample_loss))

            # Update sampler with average losses
            n_online = int(np.sum(is_online_np))
            n_offline = int(len(is_online_np) - n_online)
            if n_online > 0:
                hybrid_dataset.adaptive_sampler.add_loss(
                    float(np.mean(per_sample_loss_np[is_online_np])), is_online=True
                )
            if n_offline > 0:
                hybrid_dataset.adaptive_sampler.add_loss(
                    float(np.mean(per_sample_loss_np[~is_online_np])), is_online=False
                )

            hybrid_dataset.adaptive_sampler.update_weights()

        if step % config.log_interval == 0 or step == config.num_train_steps - 1:
            stacked_infos = common_utils.stack_forest(infos)
            # Source-specific residual metrics are NaN when a log window contains
            # no matching tokens. Ignore those empty batches during aggregation.
            reduced_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_infos))

            online_counts = np.asarray(jax.device_get(stacked_infos["batch_online_samples"]))
            batch_counts = np.asarray(jax.device_get(stacked_infos["batch_total_samples"]))
            last_batch_online_samples = int(online_counts[-1])
            last_batch_offline_samples = int(batch_counts[-1] - online_counts[-1])
            last_batch_online_ratio = float(online_counts[-1] / batch_counts[-1])
            log_online_samples = int(online_counts.sum())
            log_online_batches = int(np.count_nonzero(online_counts))
            total_sampled_online_samples += log_online_samples
            total_sampled_online_batches += log_online_batches
            # Add online dagger specific metrics
            sampling_stats = hybrid_dataset.get_sampling_stats()
            reduced_info.update(
                {
                    "online_weight": sampling_stats["online_weight"],
                    "offline_weight": sampling_stats["offline_weight"],
                    "online_loss_mean": sampling_stats["online_loss_mean"],
                    "offline_loss_mean": sampling_stats["offline_loss_mean"],
                    "online_episodes": hybrid_dataset.online_episodes_count,
                    "offline_episodes": hybrid_dataset.offline_episodes_count,
                    "online_samples": hybrid_dataset.online_len,
                    "offline_samples": hybrid_dataset.offline_len,
                    "fetched_episodes": data_fetcher.fetched_count,
                    "last_batch_online_samples": last_batch_online_samples,
                    "last_batch_offline_samples": last_batch_offline_samples,
                    "last_batch_online_ratio": last_batch_online_ratio,
                    "log_online_samples": log_online_samples,
                    "log_online_batches": log_online_batches,
                    "total_sampled_online_samples": total_sampled_online_samples,
                    "total_sampled_online_batches": total_sampled_online_batches,
                }
            )

            info_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in reduced_info.items()
            )
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
            log_online_samples = 0
            log_online_batches = 0

        # Get next batch
        batch_with_flags = next(data_iter)
        if len(batch_with_flags) == 3:
            observation, actions, is_online = batch_with_flags
        else:
            observation, actions = batch_with_flags
            is_online = jnp.zeros(actions.shape[0], dtype=jnp.bool_)
        batch = (observation, actions)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            try:
                _checkpoints.save_state(
                    checkpoint_manager,
                    train_state,
                    data_loader,
                    step,
                    params_filter=(
                        config.trainable_filter if getattr(config.model, "residual_policy_in_use", False) else None
                    ),
                )
                checkpoint_manager.wait_until_finished()
            except Exception as e:
                logging.error(f"Failed to save checkpoint at step {step}: {e}")
            done_file = config.checkpoint_dir / str(step) / ".done"
            done_file.write_text("done\n")

    data_iter.close()

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    args = tyro.cli(_config.OnlineDaggerTrainConfig)
    main(args)
