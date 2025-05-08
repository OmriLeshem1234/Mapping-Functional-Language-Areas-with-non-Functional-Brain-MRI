from scipy.ndimage import distance_transform_edt as distance_transform
from utils.my_metrics import Sensitivity
from utils.nifti_io import read_nifti
import numpy as np
from pathlib import Path
import json


def create_batch(case_idx: int, model: str):
    """
    Creates a batch dictionary containing Broca, Wernicke, ground truth (GT), and prediction arrays.

    Args:
        idx (int): Case ID.
        model (str): Model name to locate prediction outputs.

    Returns:
        dict: A batch dictionary with NumPy arrays for:
              - 'broca': Broca region segmentation
              - 'wernicke': Wernicke region segmentation
              - 'gt': Ground truth segmentation (concatenation of Broca & Wernicke)
              - 'prediction': Model-generated segmentation

    Raises:
        ValueError: If the model is not 'agynet' or 'swin_unetr'.
    """

    # Validate model name
    model = model.lower()
    valid_models = {"agynet", "swin_unetr"}
    if model not in valid_models:
        raise ValueError(f"Invalid model name '{model}'. Must be one of {valid_models}.")

    # Base directory
    base_dir = Path(__file__).parents[2]

    # Define paths
    case_path = base_dir / "data" / "data_nii" / f"case_{case_idx}"
    pred_path = base_dir / "outputs" / model / f"case_{case_idx}" / "predictions.nii.gz"

    # Read NIfTI files
    broca, _ = read_nifti(case_path / "broca.nii.gz")
    wernicke, _ = read_nifti(case_path / "wernicke.nii.gz")
    prediction, _ = read_nifti(pred_path)

    # Create GT as a concatenation of Broca & Wernicke (adds new axis for channel dimension)
    gt = np.concatenate((broca[..., None], wernicke[..., None]), axis=-1).astype(np.uint8)

    return {"broca": broca, "wernicke": wernicke, "gt": gt, "prediction": prediction}


def compute_tolerance_sensitivity_single_case(case_idx, model, max_tolerance=5):
    """
    Computes sensitivity within a specified tolerance for segmentation predictions.

    Args:
        batch (dict): Contains "y_pred_seg" (predicted segmentation) and "gt" (ground truth).
        include_background (bool): Whether to include background class in computation.
        max_tolerance (int): Maximum tolerance range for sensitivity calculation.

    Returns:
        dict: Sensitivity values at different tolerance levels.
    """

    # Convert tensors to numpy arrays and squeeze unnecessary dimensions
    batch = create_batch(case_idx=case_idx, model=model)
    gt = batch["gt"]
    prediction = batch["prediction"]

    gt = np.transpose(gt, (3, 0, 1, 2))
    prediction = np.transpose(prediction, (3, 0, 1, 2))

    # Compute distance maps for Broca and Wernicke regions
    distance_map = np.stack([
        distance_transform(1 - prediction[0]),  # Broca
        distance_transform(1 - prediction[1])  # Wernicke
    ], axis=0)

    # Initialize sensitivity metric
    sensitivity = Sensitivity(include_background=True)
    sensitivity_values = {}

    for i in range(1, max_tolerance + 1):
        y_tolerance = (distance_map <= i).astype(np.float32)

        metric = sensitivity(y_pred=y_tolerance[None, ...], y=gt.astype(np.float32)[None, ...]).numpy()
        sensitivity_values[f"sensitivity_tolerance_{i}"] = metric

    return sensitivity_values


def compute_tolerance_sensitivity_single_model(model, num_cases=30):
    """
    Computes the mean sensitivity values for Broca and Wernicke across multiple cases and saves as a JSON file.

    Args:
        model (str): The model name to evaluate.
        num_cases (int): Number of cases to process (default is 30).

    Returns:
        None: Saves results as a JSON file.
    """

    # Define the base directory for saving results
    base_dir = Path(__file__).parents[2] / "outputs" / "tests_analysis" / f"sensitivity_margins_{model}"
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
    save_path = base_dir / "tolerance_sensitivity.json"

    # Compute sensitivity results for each case
    results_list = [compute_tolerance_sensitivity_single_case(case_idx=case_idx, model=model) for case_idx in
                    range(num_cases)]

    # Initialize accumulation dictionaries
    broca_sensitivity, wernicke_sensitivity = {}, {}

    # Accumulate values across cases
    for result in results_list:
        for key, value in result.items():
            broca_sensitivity.setdefault(key, []).append(float(value[0][0]))  # Convert np.float32 -> float
            wernicke_sensitivity.setdefault(key, []).append(float(value[0][1]))  # Convert np.float32 -> float

    # Compute mean values for each sensitivity tolerance
    broca_sensitivity = {f"broca_sensitivity_margin_{k.split('_')[-1]}": float(np.mean(v)) for k, v in
                         broca_sensitivity.items()}
    wernicke_sensitivity = {f"wernicke_sensitivity_margin_{k.split('_')[-1]}": float(np.mean(v)) for k, v in
                            wernicke_sensitivity.items()}

    # Merge results into a single dictionary
    result_dict = {**broca_sensitivity, **wernicke_sensitivity}

    # Ensure directory exists before saving
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    save_path.write_text(json.dumps(result_dict, indent=4))

    print(f"Results saved to: {save_path}")


def compute_tolerance_sensitivity():
    compute_tolerance_sensitivity_single_model(model="agynet")
    compute_tolerance_sensitivity_single_model(model="swin_unetr")


if __name__ == "__main__":
    compute_tolerance_sensitivity()
