import os
from pathlib import Path
import numpy as np
from utils.nifti_io import read_nifti, save_nifti


def check_output_files_exist(num_cases=30):
    """
    Check if all output combined prediction files exist for all cases.

    Args:
        num_cases: Number of cases to check

    Returns:
        bool: True if all output files exist for all cases, False otherwise
    """
    project_dir = Path(os.path.dirname(os.path.dirname(__file__)))
    base_output_dir = project_dir / "outputs" / "agynet"

    # Define the required output files
    required_output_files = [
        "predictions_probability.nii.gz",
        "predictions.nii.gz"
    ]

    all_exist = True

    for case_idx in range(num_cases):
        case_dir = base_output_dir / f"case_{case_idx}"

        # Skip if case directory doesn't exist
        if not case_dir.exists():
            all_exist = False
            continue

        # Check if all output files exist
        for filename in required_output_files:
            file_path = case_dir / filename
            if not file_path.exists():
                all_exist = False
                return all_exist

    return all_exist


def check_all_required_files_exist(num_cases=30):
    """
    Check if all required prediction files exist for all cases.

    Args:
        num_cases: Number of cases to check

    Returns:
        bool: True if all required files exist for all cases, False otherwise
    """
    project_dir = Path(os.path.dirname(os.path.dirname(__file__)))
    base_output_dir = project_dir.joinpath("outputs/agynet")

    # Define the required files
    required_files = [
        "predictions_probability_broca.nii.gz",
        "predictions_broca.nii.gz",
        "predictions_probability_wernicke.nii.gz",
        "predictions_wernicke.nii.gz"
    ]

    all_complete = True

    for case_idx in range(num_cases):
        case_dir = base_output_dir / f"case_{case_idx}"

        # First check if case directory exists
        if not case_dir.exists():
            all_complete = False
            print(f"Case {case_idx} is missing: directory does not exist")
            continue

        # Check if all required files exist
        for filename in required_files:
            file_path = case_dir / filename
            if not file_path.exists():
                all_complete = False
                print(f"Case {case_idx} is missing: {filename}")
                break

    return all_complete


def concatenate_agynet_predictions(num_cases=30):
    """
    Concatenate Broca and Wernicke prediction files for all cases.

    Args:
        num_cases: Number of cases to process
    """
    if check_output_files_exist():
        return

    if not check_all_required_files_exist(num_cases):
        raise FileNotFoundError("Some required files are missing. Please check the output directory.")

    project_dir = Path(os.path.dirname(os.path.dirname(__file__)))
    base_output_dir = project_dir / "outputs" / "agynet"

    for case_idx in range(num_cases):
        case_dir = base_output_dir / f"case_{case_idx}"

        # Check if case directory exists
        if not case_dir.exists():
            print(f"Warning: {case_dir} does not exist. Skipping.")
            continue

        # Read Broca predictions
        broca_pred_prob_path = case_dir / "predictions_probability_broca.nii.gz"
        broca_pred_path = case_dir / "predictions_broca.nii.gz"

        # Read Wernicke predictions
        wernicke_pred_prob_path = case_dir / "predictions_probability_wernicke.nii.gz"
        wernicke_pred_path = case_dir / "predictions_wernicke.nii.gz"

        try:
            # Load data and affine
            broca_prob_data, broca_affine = read_nifti(broca_pred_prob_path)
            broca_pred_data, _ = read_nifti(broca_pred_path)

            wernicke_prob_data, wernicke_affine = read_nifti(wernicke_pred_prob_path)
            wernicke_pred_data, _ = read_nifti(wernicke_pred_path)

            # Concatenate along axis 3 (which should be the class dimension)
            combined_prob = np.concatenate([broca_prob_data, wernicke_prob_data], axis=3)
            combined_pred = np.concatenate([broca_pred_data, wernicke_pred_data], axis=3)

            # Save combined predictions
            save_nifti(combined_prob, broca_affine, case_dir / "predictions_probability.nii.gz")
            save_nifti(combined_pred, broca_affine, case_dir / "predictions.nii.gz")


        except Exception as e:
            print(f"Error processing case_{case_idx}: {e}")


if __name__ == "__main__":
    concatenate_agynet_predictions()
