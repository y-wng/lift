"""Optional, layout-aware trajectory diagnostics for offline training."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from openpi import transforms


class TrainingTrajectoryVisualizer:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_interval: int,
        norm_stats: dict | None = None,
        use_quantiles: bool = False,
        action_names: Sequence[str] | None = None,
    ):
        self.output_dir = Path(output_dir) / "trajectory_visualizations"
        self.save_interval = save_interval
        self.action_names = tuple(action_names) if action_names else None
        self.action_dim = len(self.action_names) if self.action_names else None
        if self.action_dim is None and norm_stats and "actions" in norm_stats:
            self.action_dim = len(norm_stats["actions"].mean)
        self.unnormalizer = transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles)

    def should_save(self, step: int, *, final_step: bool = False) -> bool:
        return self.save_interval > 0 and (final_step or (step > 0 and step % self.save_interval == 0))

    def save(self, step: int, actions, predicted_actions) -> dict[str, float]:
        """Plot configured action channels; never infer a second arm from padding."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        truth = self.unnormalizer({"actions": np.asarray(actions)})["actions"]
        prediction = self.unnormalizer({"actions": np.asarray(predicted_actions)})["actions"]
        if truth.shape != prediction.shape or truth.ndim != 3:
            raise ValueError("Trajectory diagnostics require matching [batch, horizon, action_dim] arrays.")
        action_dim = self.action_dim or truth.shape[-1]
        if action_dim > truth.shape[-1]:
            raise ValueError("Configured action labels exceed the model action dimension.")
        truth = truth[..., :action_dim]
        prediction = prediction[..., :action_dim]
        names = self.action_names or tuple(f"action_{index}" for index in range(action_dim))
        figure = Figure(figsize=(10, max(3, 2 * action_dim)))
        FigureCanvasAgg(figure)
        axes = figure.subplots(action_dim, 1, squeeze=False)
        for index, name in enumerate(names):
            axis = axes[index, 0]
            axis.plot(truth[0, :, index], label="Target")
            axis.plot(prediction[0, :, index], label="Prediction")
            axis.set_ylabel(name)
            axis.legend()
        axes[-1, 0].set_xlabel("Action step")
        figure.tight_layout()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.output_dir / f"step_{step:08d}.png")
        figure.clear()
        error = prediction - truth
        return {"trajectory/mse": float(np.mean(error**2)), "trajectory/mae": float(np.mean(np.abs(error)))}
