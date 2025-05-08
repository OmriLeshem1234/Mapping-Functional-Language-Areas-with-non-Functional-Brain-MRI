import numpy as np
import torch
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from utils.nifti_io import read_nifti
from utils.my_metrics import MaskedDiceMetric
import os
from time import sleep

# --------------------------------------------------------------------------------
# Mapping of Broca and Wernicke regions from the SENSAAS Atlas
# The Broca and Wernicke atlases were derived from the SENSAAS Atlas
# (Labache et al., 2019) by merging specific cortical regions associated
# with language processing.
# Reference: Labache, Loic, et al. "A SENtence Supramodal Areas AtlaS (SENSAAS) based on multiple
# task-induced activation mapping and graph analysis of intrinsic connectivity in 144 healthy
# right-handers." Brain Structure and Function 224.2 (2019): 859-882.
# --------------------------------------------------------------------------------
# Broca Atlas: Constructed by merging the following regions from the SENSAAS Atlas:
# - F301, F3t, INSa3, f2_2
# Wernicke Atlas: Constructed by merging the following regions from the SENSAAS Atlas:
# - STS3, STS4, STS2, T2_4, SMG7, AG2
# --------------------------------------------------------------------------------
# The dictionary keys represent the regional abbreviations, and the values
# correspond to their respective gray levels in the original SENSAAS.nii atlas file.
# --------------------------------------------------------------------------------
wernicke_abbreviation = {"STS3": 16, "STS4": 17, "STS2": 15, "T2_4": 19, "SMG7": 7, "AG2": 8}
broca_abbreviation = {"F301": 4, "F3t": 3, "INSa3": 12, "f2_2": 2}
# --------------------------------------------------------------------------------


# Define root paths
project_dir = Path(os.path.dirname(os.path.dirname(__file__))).parent
atlas_path = project_dir.joinpath("data").joinpath("sensaas_atlas")
data_path = project_dir.joinpath("data").joinpath("data_nii")

# Single dictionary for all paths
paths = {
    "wernicke": atlas_path.joinpath("wernicke_atlas.nii.gz"),
    "broca": atlas_path.joinpath("broca_atlas.nii.gz"),
    "t1": atlas_path.joinpath("T1_FSL_atlas_2mm_RAS_MNI.nii.gz"),
}


def register_images(moving_image, fixed_image, output_image, transform_file, dof=6, print_messages=True):
    """
    Registers a moving image (T1) to a fixed image (B0) using a specified degrees of freedom (DOF).
    """
    moving_image, fixed_image, output_image, transform_file = map(str, [moving_image, fixed_image, output_image,
                                                                        transform_file])

    if dof not in [6, 9, 12]:
        raise ValueError(f"Invalid DOF input. Expected one of [6, 9, 12], but got {dof}")

    moving_img = sitk.ReadImage(moving_image)
    fixed_img = sitk.ReadImage(fixed_image)

    moving_img = sitk.Cast(moving_img, sitk.sitkFloat32)
    fixed_img = sitk.Cast(fixed_img, sitk.sitkFloat32)

    transform_type = {6: sitk.Euler3DTransform(), 9: sitk.Similarity3DTransform(), 12: sitk.AffineTransform(3)}[dof]

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, transform_type, sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=70)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    registration.SetInterpolator(sitk.sitkBSpline)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed_img, moving_img)

    if print_messages:
        print(f'Final metric value: {registration.GetMetricValue()}')
        print(f'Optimizer stopping condition: {registration.GetOptimizerStopConditionDescription()}')

    registered_img = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkBSpline, 0.0,
                                   moving_img.GetPixelID())
    sitk.WriteImage(registered_img, output_image)
    sitk.WriteTransform(final_transform, transform_file)


def warp_image(moving_image, reference_image, transform_file, output_image, binary=False):
    """
    Warps an input image based on a given transformation.
    """
    moving_image, reference_image, transform_file, output_image = map(str,
                                                                      [moving_image, reference_image, transform_file,
                                                                       output_image])

    transform = sitk.ReadTransform(transform_file)
    moving_img = sitk.Cast(sitk.ReadImage(moving_image), sitk.sitkFloat32)
    ref_img = sitk.Cast(sitk.ReadImage(reference_image), sitk.sitkFloat32)

    interpolation_method = sitk.sitkNearestNeighbor if binary else sitk.sitkBSpline

    warped_img = sitk.Resample(moving_img, ref_img, transform, interpolation_method, 0.0, moving_img.GetPixelID())
    sitk.WriteImage(warped_img, output_image)


def compute_atlas_registration_dice_for_one_case(case_idx, region, delete_garbage=True,
                                                 remove_right_brain=True):
    """
    Computes the Dice score for one case and a given atlas.
    """
    case_path = data_path.joinpath(f"case_{case_idx}")
    atlas_file = paths[region]
    registered_atlas = case_path.joinpath(f"atlas_prediction_{region}.nii.gz")

    # Step 1: Register T1 Atlas to Case
    register_images(
        moving_image=paths["t1"],
        fixed_image=case_path.joinpath("t1.nii.gz"),
        output_image=case_path.joinpath("t1_atlas_2_case.nii.gz"),
        transform_file=case_path.joinpath("t1_atlas_2_case_transform.txt"),
        dof=12,
        print_messages=False
    )

    # Step 2: Warp Atlas to Subject Space
    warp_image(
        moving_image=atlas_file,
        reference_image=case_path.joinpath("t1.nii.gz"),
        transform_file=case_path.joinpath("t1_atlas_2_case_transform.txt"),
        output_image=registered_atlas,
        binary=True
    )

    # Step 3: Load Data
    ground_truth, _ = read_nifti(case_path.joinpath(f"{region}.nii.gz"))
    predicted, _ = read_nifti(registered_atlas)
    mask, _ = read_nifti(case_path.joinpath("brain_mask.nii.gz"))

    # Step 4: Clean up temporary files if needed
    if delete_garbage:
        registered_atlas.unlink(missing_ok=True)
        case_path.joinpath("t1_atlas_2_case_transform.txt").unlink(missing_ok=True)
        case_path.joinpath("t1_atlas_2_case.nii.gz").unlink(missing_ok=True)

    # Step 5: Compute Dice Metric
    metric = MaskedDiceMetric(include_background=True, remove_right_brain=remove_right_brain)
    return metric(
        y=torch.from_numpy(ground_truth)[None, ...],
        y_pred=torch.from_numpy(predicted)[None, ...],
        mask=torch.from_numpy(mask)[None, ...]
    )


def compute_atlas_registration_dice_all_dataset(region, delete_garbage=True, remove_right_brain=True):
    dice_scores = []
    for case_idx in tqdm(range(30), desc=f"Processing {region} atlas"):
        dice_scores.append(
            compute_atlas_registration_dice_for_one_case(case_idx, region, delete_garbage, remove_right_brain))

    mean_dice = torch.mean(torch.stack(dice_scores)).item()
    std_dice = torch.std(torch.stack(dice_scores)).item()
    return mean_dice, std_dice


def compute_atlas_registration_dice_all_dataset_with_repetitions(region, delete_garbage=True,
                                                                 remove_right_brain=True, n_repetitions=3):
    mean_values, std_values = [], []

    for _ in range(n_repetitions):
        mean_dice, std_dice = compute_atlas_registration_dice_all_dataset(region, delete_garbage,
                                                                          remove_right_brain)
        mean_values.append(mean_dice)
        std_values.append(std_dice)

    sleep(1)
    print(
        f"{region.capitalize()} atlas registration dice results: {np.mean(mean_values):.4f} ± {np.mean(std_values):.4f}\n\n")


if __name__ == '__main__':
    # Compute Dice for both Broca and Wernicke
    for region in ["broca", "wernicke"]:
        compute_atlas_registration_dice_all_dataset_with_repetitions(region)
