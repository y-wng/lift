import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_single_iphone_flexiv_example() -> dict:
    """Create a random single-arm input example with wrist image and wrench."""
    return {
        "observation/state": np.random.rand(7),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "actions": np.random.rand(7),
        "prompt": "do something",
        "wrench": np.random.rand(6),
        "wrench_mask": np.False_,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SingleiPhoneFlexivInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": np.zeros_like(left_wrist_image),
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": np.zeros_like(left_wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "prev_state" in data:
            inputs["prev_state"] = data["prev_state"]

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "wrench" in data:
            inputs["wrench"] = data["wrench"]

        if "wrench_mask" in data:
            inputs["wrench_mask"] = data["wrench_mask"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SingleiPhoneFlexivOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :7])
        return {"actions": actions}
