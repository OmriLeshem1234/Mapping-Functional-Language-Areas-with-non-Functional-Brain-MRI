import nibabel as nib
import numpy as np
from pathlib2 import Path


def read_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads a NIfTI file and returns the image data and affine matrix."""
    nifti = nib.load(str(path))
    return nifti.get_fdata(), nifti.affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path):
    """Saves a NIfTI file with the given data and affine matrix."""
    path = Path(path)

    if ".nii" not in path.name:
        raise ValueError("Invalid file extension. Allowed: .nii or .nii.gz")

    if path.parent.exists() is False:
        path.parent.mkdir(parents=True, exist_ok=True)

    nib.save(nib.Nifti1Image(data, affine), str(path))
