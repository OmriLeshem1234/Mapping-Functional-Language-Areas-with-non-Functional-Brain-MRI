from utils.merge_output_nii import concatenate_agynet_predictions, check_all_required_files_exist
from testing.code.compute_roc_curves import create_roc_curve
from testing.code.sensitivity_margins_analysis import compute_tolerance_sensitivity
from testing.code.atlas_based_segmentation_comparison import \
    compute_atlas_registration_dice_all_dataset_with_repetitions
from time import sleep


def post_tester_analysis():
    if not check_all_required_files_exist():
        return

    print("\n\nAll required inference output files are present. Continuing with further analysis. Please wait...")

    # Some initialization
    concatenate_agynet_predictions()

    # Compute atlas-based left-hemisphere Dice index for both Broca and Wernicke
    print("\n\nComputing atlas-based left-hemisphere Dice index for both Broca and Wernicke...\n")
    sleep(0.5)
    for region in ["broca", "wernicke"]:
        compute_atlas_registration_dice_all_dataset_with_repetitions(region)

    print("\n\nComputing ROC curves...")
    sleep(0.3)
    create_roc_curve()
    print("finished computing ROC curves.")

    print("\n\nComputing margins sensitivity analysis...")
    sleep(0.3)
    compute_tolerance_sensitivity()
    print("finished computing margins sensitivity analysis.")


if __name__ == "__main__":
    post_tester_analysis()
