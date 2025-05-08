import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from utils.nifti_io import read_nifti


def create_batch(case_idx: int, model: str):
    """
    Loads Broca, Wernicke, ground truth (GT), brain mask, and prediction arrays for a given case.

    Args:
        case_idx (int): Case ID.
        model (str): Model name (either 'agynet' or 'swin_unetr').

    Returns:
        dict: Batch containing:
              - 'broca': Broca region segmentation
              - 'wernicke': Wernicke region segmentation
              - 'gt': Ground truth segmentation (concatenation of Broca & Wernicke)
              - 'mask': Brain mask
              - 'prediction': Model-generated segmentation

    Raises:
        ValueError: If the model is invalid.
        FileNotFoundError: If any required file is missing.
    """

    model = model.lower()
    valid_models = {"agynet", "swin_unetr"}
    if model not in valid_models:
        raise ValueError(f"Invalid model name '{model}'. Must be one of {valid_models}.")

    # Define paths
    base_dir = Path(__file__).parents[2]
    case_path = base_dir / "data" / "data_nii" / f"case_{case_idx}"
    pred_path = base_dir / "outputs" / model / f"case_{case_idx}" / "predictions_probability.nii.gz"

    # Load NIfTI files
    try:
        broca, _ = read_nifti(case_path / "broca.nii.gz")
        wernicke, _ = read_nifti(case_path / "wernicke.nii.gz")
        mask, _ = read_nifti(case_path / "brain_mask.nii.gz")
        prediction, _ = read_nifti(pred_path)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file for case {case_idx}: {e}")

    # Create GT as a concatenation of Broca & Wernicke
    gt = np.stack((broca, wernicke), axis=-1).astype(np.uint8)

    return {"broca": broca, "wernicke": wernicke, "gt": gt, "mask": mask, "prediction": prediction}


def flatten_with_mask(array, mask):
    """Flattens an array using a binary mask."""
    return array[mask == 1]


def flatten_and_concat(array_list, mask_list):
    """Flattens and concatenates multiple masked arrays into a single array."""
    return np.concatenate([flatten_with_mask(arr, mask) for arr, mask in zip(array_list, mask_list)])


def compute_roc_auc(seg_class, model, num_cases=30):
    """
    Computes ROC curve and AUC score for the given segmentation class and model.

    Args:
        seg_class (str): 'broca' or 'wernicke'.
        model (str): Model name ('swin_unetr' or 'agynet').
        num_cases (int): Number of cases to process.

    Returns:
        tuple: (fpr, tpr, auc)
    """

    gt_list, pred_list, mask_list = [], [], []

    for case_idx in tqdm(range(num_cases), desc=f"Processing {seg_class} - {model}"):
        # Load batch data
        batch = create_batch(case_idx, model)

        # Extract GT, Predictions, and Mask
        gt_list.append(batch["gt"][..., 0] if seg_class == "broca" else batch["gt"][..., 1])
        pred_list.append(batch["prediction"][..., 0] if seg_class == "broca" else batch["prediction"][..., 1])
        mask_list.append(batch["mask"])

    # Flatten and concatenate data
    gt_flat = flatten_and_concat(gt_list, mask_list)
    pred_flat = flatten_and_concat(pred_list, mask_list)

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_true=gt_flat, y_score=pred_flat)
    auc = roc_auc_score(y_true=gt_flat, y_score=pred_flat)

    return fpr, tpr, auc


def create_roc_curve():
    """
    Computes and saves ROC curves for Broca and Wernicke segmentations for Swin-UNETR and AGYnet.
    """

    save_dir = Path(__file__).parents[2] / "outputs" / "tests_analysis" / "ROC_curves"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "ROC.png"

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    plot_config = {
        "s": 14, "title_size": 16, "linewidth": 2, "legend_size": 10,  # Reduced legend size
        "tick_size": 12, "x_labelpad": 10, "y_labelpad": 10,
        "wspace": 0.3
    }

    plt.rcParams.update({
        'xtick.labelsize': plot_config["tick_size"],
        'ytick.labelsize': plot_config["tick_size"]
    })

    # Define models and colors
    models = ["swin_unetr", "agynet"]
    colors = ["blue", "red"]

    # Plot ROC Curves for Broca
    for model, color in zip(models, colors):
        fpr, tpr, auc = compute_roc_auc(seg_class="broca", model=model)
        if fpr is not None:
            ax[0].plot(fpr, tpr, color=color, label=f'{model.upper()} (AUC = {auc:.2f})',
                       linewidth=plot_config["linewidth"])

    ax[0].set_xlabel('1 - Specificity', fontsize=plot_config["s"], labelpad=plot_config["x_labelpad"])
    ax[0].set_ylabel('Sensitivity', fontsize=plot_config["s"], labelpad=plot_config["y_labelpad"])
    ax[0].set_ylim([0.5, 1.00])
    ax[0].set_title('(a) Broca', fontsize=plot_config["title_size"], y=1.02)
    ax[0].legend(fontsize=plot_config["legend_size"], loc='lower right',
                 frameon=False)  # Smaller legend in bottom right

    # Plot ROC Curves for Wernicke
    for model, color in zip(models, colors):
        fpr, tpr, auc = compute_roc_auc(seg_class="wernicke", model=model)
        if fpr is not None:
            ax[1].plot(fpr, tpr, color=color, label=f'{model.upper()} (AUC = {auc:.2f})',
                       linewidth=plot_config["linewidth"])

    ax[1].set_xlabel('1 - Specificity', fontsize=plot_config["s"], labelpad=plot_config["x_labelpad"])
    ax[1].set_ylabel('Sensitivity', fontsize=plot_config["s"], labelpad=plot_config["y_labelpad"])
    ax[1].set_ylim([0.5, 1.00])
    ax[1].set_title('(b) Wernicke', fontsize=plot_config["title_size"], y=1.02)
    ax[1].legend(fontsize=plot_config["legend_size"], loc='lower right',
                 frameon=False)  # Smaller legend in bottom right

    # Adjust layout and save figure
    plt.subplots_adjust(wspace=plot_config["wspace"])

    for axis in ax:
        for spine in axis.spines.values():
            spine.set_linewidth(1.5)  # Adjust bounding box thickness

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"ROC curve saved at: {save_path}")


if __name__ == "__main__":
    create_roc_curve()
