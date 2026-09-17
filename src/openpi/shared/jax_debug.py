"""Opt-in host-side inspection hooks for values inside jitted JAX code.

Set ``OPENPI_JAX_DEBUG=1`` before starting the process to enable the hooks.
Set ``OPENPI_JAX_DEBUG_BREAK=1`` to enter the connected ``debugpy`` debugger
from the callback in single-device runs (or ``pdb`` when no debugpy client is
attached). Interactive breakpoints are disabled for multi-device runs because
pausing one device blocks the others at collective operations. The callback
keeps a ``debug_values`` dictionary in its local scope, so values can be
inspected by name when the breakpoint is hit. ``OPENPI_JAX_DEBUG_TAGS`` can be
a comma-separated list such as ``residual/targets,train/metrics``.
"""

from collections.abc import Sequence
import logging
import os
from typing import Any

import jax
import numpy as np

_LOGGER = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


_ENABLED = _env_flag("OPENPI_JAX_DEBUG")
_BREAK = _env_flag("OPENPI_JAX_DEBUG_BREAK")
_TAG_FILTER = frozenset(filter(None, (tag.strip() for tag in os.environ.get("OPENPI_JAX_DEBUG_TAGS", "").split(","))))


class _ActionNormalization:
    stats: tuple[np.ndarray, np.ndarray] | None = None
    uses_quantiles: bool = False


_action_normalization = _ActionNormalization()


def enabled() -> bool:
    """Return whether debug callbacks are enabled for this process."""

    return _ENABLED


def configure_action_normalization(norm_stats: Any, *, use_quantiles: bool) -> None:
    """Configure physical-action values exposed by residual debug callbacks."""

    action_stats = None if norm_stats is None else norm_stats.get("actions")
    if action_stats is None:
        _action_normalization.stats = None
        return

    if use_quantiles:
        if action_stats.q01 is None or action_stats.q99 is None:
            raise ValueError("Action quantiles are required for quantile-normalized debug values")
        _action_normalization.stats = (np.asarray(action_stats.q01), np.asarray(action_stats.q99))
    else:
        _action_normalization.stats = (np.asarray(action_stats.mean), np.asarray(action_stats.std))
    _action_normalization.uses_quantiles = use_quantiles


def _add_physical_action_values(debug_values: dict[str, np.ndarray]) -> None:
    if _action_normalization.stats is None:
        return
    if not {"base_actions", "ground_truth_actions", "residual_target"}.issubset(debug_values):
        return

    lower, upper = _action_normalization.stats
    action_dim = lower.shape[-1]
    base = debug_values["base_actions"][..., :action_dim]
    ground_truth = debug_values["ground_truth_actions"][..., :action_dim]
    residual = debug_values["residual_target"][..., :action_dim]
    if _action_normalization.uses_quantiles:
        scale = upper - lower + 1e-6
        debug_values["base_actions_physical"] = (base + 1.0) / 2.0 * scale + lower
        debug_values["ground_truth_actions_physical"] = (ground_truth + 1.0) / 2.0 * scale + lower
        debug_values["residual_target_physical"] = residual * scale / 2.0
    else:
        scale = upper + 1e-6
        debug_values["base_actions_physical"] = base * scale + lower
        debug_values["ground_truth_actions_physical"] = ground_truth * scale + lower
        debug_values["residual_target_physical"] = residual * scale


def inspect(
    tag: str,
    *values: Any,
    names: Sequence[str] | None = None,
    full_value_names: Sequence[str] | None = None,
) -> None:
    """Schedule a host callback that prints or interactively inspects values.

    This function intentionally has no returned value and therefore cannot
    perturb the numerical computation.  It is safe to call from code traced by
    ``jax.jit``, ``jax.grad`` and control-flow primitives.  Names listed in
    ``full_value_names`` are printed in full in addition to the usual summary.
    """

    if not _ENABLED or (_TAG_FILTER and tag not in _TAG_FILTER):
        return

    value_names = tuple(names or (f"value_{i}" for i in range(len(values))))
    if len(value_names) != len(values):
        raise ValueError(f"Expected {len(values)} names for debug tag {tag!r}, got {len(value_names)}")
    full_value_names = frozenset(full_value_names or ())
    device_count = jax.device_count()
    break_enabled = _BREAK and device_count == 1
    if _BREAK and not break_enabled:
        _LOGGER.warning(
            "JAX DEBUG [%s] interactive breakpoint disabled because %d JAX devices are active; "
            "pausing one device would block multi-device collectives",
            tag,
            device_count,
        )

    def _callback(*runtime_values: Any) -> None:
        # Keep this dictionary deliberately local: it is convenient to inspect
        # by name after setting a breakpoint in this callback.
        debug_values = {name: np.asarray(value) for name, value in zip(value_names, runtime_values, strict=True)}
        _add_physical_action_values(debug_values)
        summaries = []
        for name, value in debug_values.items():
            is_numeric = value.dtype.kind in "biufc" or value.dtype.name == "bfloat16"
            finite = bool(np.all(np.isfinite(value))) if is_numeric else True
            if value.size and is_numeric:
                stats_value = value.astype(np.float32, copy=False) if value.dtype.name == "bfloat16" else value
                summaries.append(
                    f"{name}: shape={value.shape} dtype={value.dtype} finite={finite} "
                    f"min={np.nanmin(stats_value):.5g} max={np.nanmax(stats_value):.5g} mean={np.nanmean(stats_value):.5g}"
                )
            else:
                summaries.append(f"{name}: shape={value.shape} dtype={value.dtype} finite={finite}")
        _LOGGER.warning("JAX DEBUG [%s] %s", tag, " | ".join(summaries))
        for name, value in debug_values.items():
            if name not in full_value_names:
                continue
            printable_value = value.astype(np.float32, copy=False) if value.dtype.name == "bfloat16" else value
            _LOGGER.warning(
                "JAX DEBUG [%s] %s full value:\n%s",
                tag,
                name,
                np.array2string(printable_value, threshold=np.inf, max_line_width=200),
            )
        if break_enabled:
            try:
                import debugpy
            except ImportError:
                breakpoint()
            else:
                if debugpy.is_client_connected():
                    debugpy.breakpoint()
                else:
                    breakpoint()

    # Ordered debug effects are not supported by multi-device jitted
    # computations, which is the normal training setup for this repository.
    jax.debug.callback(_callback, *values)
