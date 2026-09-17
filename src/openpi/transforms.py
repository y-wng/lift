from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools
from scipy.spatial.transform import Rotation

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize
import openpi.shared.pose_utils as dsl

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


def residual_policy_norm_stats(norm_stats: Mapping[str, NormStats] | None) -> dict[str, NormStats] | None:
    """Keep base state/action stats while leaving residual image and wrench inputs raw."""

    if norm_stats is None:
        return None
    return {
        key: value
        for key, value in norm_stats.items()
        if not any(term in key.lower() for term in ("image", "img", "wrench"))
    }


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item.get(k, None), self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        if x is None:
            return x
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        if x is None:
            return x
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=self.strict,
        )

    def _unnormalize(self, x, stats: NormStats):
        if x is None:
            return x
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        if x is None:
            return x
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class IPhoneZeroStateAndRealRelativeActions(DataTransformFn):
    """Repacks absolute state and actions into zero state and relative actions space."""

    bimanual: bool = False
    gravity_anchored_relative_actions: bool = False
    gravity_axis: str = "y"

    def array_to_pose_and_gripper(self, x):
        # convert single arm array to pose and gripper
        assert x.shape[-1] == 7, f"iPhone single arm array must have 7 dimensions, but got {x.shape[-1]}"
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        pos, axis_angle, grip = x[:, :3], x[:, 3:6], x[:, 6:7]
        pose = np.concatenate([pos, axis_angle], axis=1)
        pose_mat = dsl.pose_to_mat(pose)  # pos, axis_angle -> mat
        return pose_mat, grip

    def gravity_anchored_state_pose_mat(self, pose_mat):
        pose_mat = pose_mat.copy()
        rot = pose_mat[:, :3, :3]
        if self.gravity_axis == "x":
            rot = Rotation.from_matrix(rot).as_euler("yzx")
            rot[:, :2] = 0
            rot = Rotation.from_euler("yzx", rot).as_matrix()
        elif self.gravity_axis == "y":
            rot = Rotation.from_matrix(rot).as_euler("xzy")
            rot[:, :2] = 0
            rot = Rotation.from_euler("xzy", rot).as_matrix()
        elif self.gravity_axis == "z":
            rot = Rotation.from_matrix(rot).as_euler("xyz")
            rot[:, :2] = 0
            rot = Rotation.from_euler("xyz", rot).as_matrix()
        else:
            raise ValueError(f"Invalid gravity axis: {self.gravity_axis}")
        pose_mat[:, :3, :3] = rot
        return pose_mat

    def __call__(self, data: DataDict) -> DataDict:
        state = data["state"]
        if "actions" in data:
            state_left_pose_mat, state_left_grip = self.array_to_pose_and_gripper(state[..., :7])
            if self.gravity_anchored_relative_actions:
                state_left_pose_mat = self.gravity_anchored_state_pose_mat(state_left_pose_mat)
            action_left_pose_mat, action_left_grip = self.array_to_pose_and_gripper(data["actions"][:, :7])
            action_left_relative_pose_mat = dsl.convert_pose_mat_rep(
                action_left_pose_mat, state_left_pose_mat[0], "relative"
            )
            action_left_relative_rpy = Rotation.from_matrix(action_left_relative_pose_mat[:, :3, :3]).as_euler(
                "xyz", degrees=False
            )
            action_left_relative_pos = action_left_relative_pose_mat[:, :3, 3]
            action_left = np.concatenate([action_left_relative_pos, action_left_relative_rpy, action_left_grip], axis=1)
            if self.bimanual:
                state_right_pose_mat, state_right_grip = self.array_to_pose_and_gripper(state[..., 7:14])
                if self.gravity_anchored_relative_actions:
                    state_right_pose_mat = self.gravity_anchored_state_pose_mat(state_right_pose_mat)
                action_right_pose_mat, action_right_grip = self.array_to_pose_and_gripper(data["actions"][:, 7:14])
                action_right_relative_pose_mat = dsl.convert_pose_mat_rep(
                    action_right_pose_mat, state_right_pose_mat[0], "relative"
                )
                action_right_relative_rpy = Rotation.from_matrix(action_right_relative_pose_mat[:, :3, :3]).as_euler(
                    "xyz", degrees=False
                )
                action_right_relative_pos = action_right_relative_pose_mat[:, :3, 3]
                action_right = np.concatenate(
                    [action_right_relative_pos, action_right_relative_rpy, action_right_grip], axis=1
                )
                data["actions"] = np.concatenate([action_left, action_right], axis=1)
            else:
                data["actions"] = action_left
        data["state"] = np.zeros_like(state)
        return data


@dataclasses.dataclass(frozen=True)
class IPhoneIdentityActions(DataTransformFn):
    """Repacks"""

    bimanual: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        return data


@dataclasses.dataclass(frozen=True)
class UmiZeroStateAndRelativeActions(DataTransformFn):
    """Repacks absolute state and actions into zero state and relative actions space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.mask is None:
            return data
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        state = data["state"]
        if "actions" in data:
            actions = data["actions"]
            actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
            data["actions"] = actions
        data["state"] = np.zeros_like(state)
        return data


@dataclasses.dataclass(frozen=True)
class UmiZeroStateAndRealRelativeActions(DataTransformFn):
    """Repacks absolute state and actions into zero state and relative actions space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def array_to_pose_and_gripper(self, x):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        left_pos, left_axis_angle, left_grip, right_pos, right_axis_angle, right_grip = (
            x[:, :3],
            x[:, 3:6],
            x[:, 6:7],
            x[:, 7:10],
            x[:, 10:13],
            x[:, 13:14],
        )
        left_pose = np.concatenate([left_pos, left_axis_angle], axis=1)
        right_pose = np.concatenate([right_pos, right_axis_angle], axis=1)
        left_pose_mat = dsl.pose_to_mat(left_pose)  # pos, axis_angle -> mat
        right_pose_mat = dsl.pose_to_mat(right_pose)
        return left_pose_mat, left_grip, right_pose_mat, right_grip

    def __call__(self, data: DataDict) -> DataDict:
        if self.mask is None:
            return data
        state = data["state"]
        if "actions" in data:
            state_left_pose_mat, state_left_grip, state_right_pose_mat, state_right_grip = (
                self.array_to_pose_and_gripper(state)
            )
            action_left_pose_mat, action_left_grip, action_right_pose_mat, action_right_grip = (
                self.array_to_pose_and_gripper(data["actions"])
            )
            # convert to relative poses based on the state
            action_left_relative_pose_mat = dsl.convert_pose_mat_rep(
                action_left_pose_mat, state_left_pose_mat[0], "relative"
            )
            action_right_relative_pose_mat = dsl.convert_pose_mat_rep(
                action_right_pose_mat, state_right_pose_mat[0], "relative"
            )
            # convert rotation to rpy
            action_left_relative_rpy = Rotation.from_matrix(action_left_relative_pose_mat[:, :3, :3]).as_euler(
                "xyz", degrees=False
            )
            action_right_relative_rpy = Rotation.from_matrix(action_right_relative_pose_mat[:, :3, :3]).as_euler(
                "xyz", degrees=False
            )
            action_left_relative_pos = action_left_relative_pose_mat[:, :3, 3]
            action_right_relative_pos = action_right_relative_pose_mat[:, :3, 3]
            # merge actions
            action_left = np.concatenate([action_left_relative_pos, action_left_relative_rpy, action_left_grip], axis=1)
            action_right = np.concatenate(
                [action_right_relative_pos, action_right_relative_rpy, action_right_grip], axis=1
            )
            data["actions"] = np.concatenate([action_left, action_right], axis=1)
        data["state"] = np.zeros_like(state)
        return data


@dataclasses.dataclass(frozen=True)
class UmiIdentityActions(DataTransformFn):
    """Repacks"""

    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        return data


@dataclasses.dataclass(frozen=True)
class UmiDeltaStateAndActions(DataTransformFn):
    """Repacks absolute state and actions into delta state and actions space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.mask is None:
            return data
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        state, prev_state = data["state"], data["prev_state"]
        if "actions" not in data:
            data["state"][:dims] = state - np.where(mask, prev_state[:dims], 0)
            return data
        actions = data["actions"]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions
        data["state"][:dims] = state - np.where(mask, prev_state[:dims], 0)
        return data


@dataclasses.dataclass(frozen=True)
class UmiAbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data
        state, actions, prev_state = data["state"], data["actions"], data["prev_state"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        state[:dims] += np.where(mask, prev_state[:dims], 0)
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
