import monai
import numpy as np
import torch


class Sensitivity:
    def __init__(self, include_background=False):
        self.include_background = include_background

    def __call__(self, y_pred, y):
        if type(y_pred) is np.ndarray:
            y_pred = torch.from_numpy(y_pred)
        if type(y) is np.ndarray:
            y = torch.from_numpy(y)

        confusion_matrix = monai.metrics.get_confusion_matrix(y_pred=y_pred, y=y,
                                                              include_background=self.include_background)
        value = monai.metrics.compute_confusion_matrix_metric("sensitivity", confusion_matrix)
        return value


class Specificity:
    def __init__(self, include_background=False):
        self.include_background = include_background

    def __call__(self, y_pred, y):
        if type(y_pred) is np.ndarray:
            y_pred = torch.from_numpy(y_pred)
        if type(y) is np.ndarray:
            y = torch.from_numpy(y)

        confusion_matrix = monai.metrics.get_confusion_matrix(y_pred=y_pred, y=y,
                                                              include_background=self.include_background)
        value = monai.metrics.compute_confusion_matrix_metric("specificity", confusion_matrix)
        return value


class Accuracy:
    def __init__(self, include_background=False):
        self.include_background = include_background

    def __call__(self, y_pred, y):

        if type(y_pred) is np.ndarray:
            y_pred = torch.from_numpy(y_pred)
        if type(y) is np.ndarray:
            y = torch.from_numpy(y)

        confusion_matrix = monai.metrics.get_confusion_matrix(y_pred=y_pred, y=y,
                                                              include_background=self.include_background)
        value = monai.metrics.compute_confusion_matrix_metric("accuracy", confusion_matrix)
        return value


class MaskedDiceMetric(monai.metrics.DiceMetric):
    """
    A custom Dice metric that supports optional masking and right brain removal.

    This metric extends the standard MONAI DiceMetric by:
    1. Applying a mask to predictions and ground truth
    2. Optionally removing the right side of the brain from calculations
    """

    def __init__(self, remove_right_brain=False, include_background=True, **kwargs):
        """
        Initialize the MaskedDiceMetric.

        Args:
            remove_right_brain (bool, optional): Whether to remove the right side of the brain. Defaults to False.
            include_background (bool, optional): Whether to include background in Dice calculation. Defaults to True.
            **kwargs: Additional arguments passed to the parent DiceMetric constructor.
        """
        super().__init__(include_background=include_background, **kwargs)
        self.remove_right_brain = remove_right_brain

    def __call__(self, y_pred, y, mask):
        """
        Calculate the masked Dice metric.

        Args:
            y_pred (torch.Tensor): Predicted segmentation tensor
            y (torch.Tensor): Ground truth segmentation tensor
            mask (torch.Tensor): Mask tensor to apply to predictions and ground truth

        Returns:
            The Dice metric after applying mask and optional right brain removal
        """
        # Ensure correct tensor dimensions
        mask = self._ensure_mask_dimensions(mask)
        y = self._ensure_tensor_dimensions(y)
        y_pred = self._ensure_tensor_dimensions(y_pred)

        # Ensure tensors are on the same device
        device = y_pred.device
        mask = mask.to(device)
        y = y.to(device)

        # Remove right brain if specified
        if self.remove_right_brain:
            mask = self._remove_right_brain(mask)

        # Apply mask
        y_pred = y_pred * mask
        y = y * mask

        metric = super().__call__(y_pred=y_pred, y=y)

        if torch.isnan(metric).any():  # in case of empty ground truth on the left hemisphere, extremely rare
            if torch.allclose(y, y_pred):
                metric = torch.ones_like(metric)
            else:
                metric = torch.zeros_like(metric)

        return metric

    def _ensure_mask_dimensions(self, mask):
        """
        Ensure mask tensor has correct dimensions.

        Args:
            mask (torch.Tensor): Input mask tensor

        Returns:
            torch.Tensor: Mask tensor with correct dimensions
        """
        if len(mask.shape) == 4 and mask.shape[0] == 1:
            return mask[None, ...]
        return mask

    def _ensure_tensor_dimensions(self, tensor):
        """
        Ensure segmentation tensor has correct dimensions.

        Args:
            tensor (torch.Tensor): Input segmentation tensor

        Returns:
            torch.Tensor: Segmentation tensor with correct dimensions
        """
        if len(tensor.shape) == 4 and tensor.shape[0] == 1:
            return tensor[None, ...]
        return tensor

    def _remove_right_brain(self, mask):
        """
        Remove the right side of the brain from the mask.

        Args:
            mask (torch.Tensor): Input mask tensor

        Returns:
            torch.Tensor: Mask with right brain side zeroed out
        """
        # Validate mask dimensions
        if not self.remove_right_brain or len(mask.shape) != 5:
            return mask

        if mask.shape[0] != 1:
            raise ValueError(
                f"Cannot perform right side removal for non-single batch size tensor. "
                f"Got mask size: {mask.shape} with batch size {mask.shape[0]}"
            )

        # Convert to torch tensor if needed
        torch_mask = torch.from_numpy(mask).to(mask.device) if not isinstance(mask,
                                                                              torch.Tensor) else mask.clone().detach()

        # Find midpoint along sagittal plane
        x_coords = torch.nonzero(torch_mask, as_tuple=True)[2]
        mid_sagittal = (x_coords.max() - x_coords.min()) // 2

        # Zero out right side of the mask
        mask[:, :, :mid_sagittal + 1, ...] = 0

        return mask
