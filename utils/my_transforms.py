"""
Custom data transforms for medical imaging processing.
"""

import torch
import numpy as np
import monai
from monai.transforms.transform import MapTransform


class JointCrop(MapTransform):
    """Crops all inputs using the bounding box computed from a significant key."""

    def __init__(self, keys, significant_key, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.significant_key = significant_key

    def __call__(self, data):
        data = dict(data)
        transform = monai.transforms.CropForeground(allow_smaller=False)
        bounding_box = transform.compute_bounding_box(img=data[self.significant_key])

        for key in self.keys:
            data[key] = transform.crop_pad(data[key], *bounding_box)

        return data


class AddBackgroundGTChannel(MapTransform):
    """Adds a background ground truth (GT) channel."""

    def __init__(self, keys, allow_missing_keys=False, device="cuda"):
        super().__init__(keys, allow_missing_keys)
        self.device = device

    def __call__(self, data):
        data = dict(data)
        for key in self.keys:
            data[key] = self._add_gt(data[key])
        return data

    @staticmethod
    def _add_gt(gt: torch.Tensor):
        _, h, w, d = gt.shape
        background = torch.zeros((1, h, w, d), dtype=gt.dtype)
        background[torch.sum(gt, dim=0, keepdim=True) == 0] = 1
        return torch.cat((background, gt), dim=0)


class Normalize0to1(MapTransform):
    """Normalizes input data to the range [0, 1]."""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        data = dict(data)
        for key in self.keys:
            data[key] = self._normalize(data[key])
        return data

    @staticmethod
    def _normalize(tensor: torch.Tensor):
        return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor) + 1e-8)


class RenameKey(MapTransform):
    """Renames a key in the dataset dictionary."""

    def __init__(self, old_name, new_name, allow_missing_keys=False):
        super().__init__(keys=[old_name], allow_missing_keys=allow_missing_keys)
        self.old_name = old_name
        self.new_name = new_name

    def __call__(self, data):
        data = dict(data)
        if self.old_name in data:
            data[self.new_name] = data.pop(self.old_name)
        return data


class RemoveKeys(MapTransform):
    """Removes specified keys from the dataset dictionary."""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data):
        data = dict(data)
        for key in self.keys:
            data.pop(key, None)  # Avoids KeyError if key is missing
        return data


class RemoveChannel(MapTransform):
    """Removes specified channels from an image tensor."""

    def __init__(self, keys, channels_to_remove, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.channels_to_remove = set(channels_to_remove) if isinstance(channels_to_remove, list) else {
            channels_to_remove}

    def __call__(self, data):
        data = dict(data)
        for key in self.keys:
            n_channels = data[key].shape[0]
            remaining_channels = [i for i in range(n_channels) if i not in self.channels_to_remove]
            data[key] = data[key][remaining_channels]
            if data[key].ndim != len(data[key].shape):
                data[key] = data[key][None, ...]
        return data


class RandomSagittalFlip(MapTransform):
    """Randomly flips an image along the sagittal axis."""

    def __init__(self, keys, allow_missing_keys=False, axis=1):
        super().__init__(keys, allow_missing_keys)
        self.axis = axis

    def __call__(self, data):
        data = dict(data)
        if torch.rand(1).item() > 0.5:  # 50% probability
            for key in self.keys:
                data[key] = torch.flip(data[key], [self.axis])
        return data


class ConcatItemsdClone(MapTransform):
    """A duplicate of `monai.transforms.ConcatItemsd` to be used multiple times in a pipeline."""

    def __init__(self, keys, dim, name, allow_missing_keys=False):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.concat = monai.transforms.ConcatItemsd(keys=keys, dim=dim, name=name)

    def __call__(self, data):
        return self.concat(data)
