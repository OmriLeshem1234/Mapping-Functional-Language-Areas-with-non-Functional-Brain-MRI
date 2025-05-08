import json
from pathlib2 import Path

# Hardcoded Configuration
base_dir = Path(__file__).parent.joinpath("data").joinpath("data_nii")

baseDirs = {
    "t1": base_dir,
    "dec_rgb": base_dir,
    "brain_mask": base_dir,
    "gm": base_dir,
    "broca": base_dir,
    "wernicke": base_dir,
}

# Dataset split percentages
trainPerc = 70
valPerc = 10
testPerc = 20


def divide_data(datalist):
    """Splits data into training, validation, and test sets."""
    if (trainPerc + valPerc + testPerc) != 100:
        raise ValueError("Percentages of train/val/test must sum to 100!")

    train_num = round(len(datalist) * (trainPerc / 100))
    val_num = round(len(datalist) * (valPerc / 100))

    return {
        "training": datalist[:train_num],
        "validation": datalist[train_num: train_num + val_num],
        "test": datalist[train_num + val_num:]
    }


def create_dict(allData):
    """Creates a structured dictionary for JSON output."""
    final_dict = {"training": [], "validation": [], "test": []}

    for mod in final_dict:
        for patient in allData[mod]:
            paths = {modality: str(baseDirs[modality].joinpath(patient, f"{modality}.nii.gz")) for modality in baseDirs}
            final_dict[mod].append(paths)

    return final_dict


def process_fold(fold):
    """Processes a single fold and creates its corresponding JSON dataset file."""
    datalist_path = Path(__file__).parent.joinpath("data").joinpath("case_lists").joinpath(f"fold_{fold}_data_list.txt")
    save_path = Path(__file__).parent.joinpath("data").joinpath("json_datasets")
    datasetName = f"functional_language_areas_mapping_fold_{fold}.json"

    # Ensure save directory exists
    save_path.mkdir(parents=True, exist_ok=True)
    outPath = save_path.joinpath(datasetName)

    # Load list of patients
    datalist = datalist_path.read_text().splitlines()

    # Divide data and create JSON structure
    allData = divide_data(datalist)
    jsonDict = create_dict(allData)

    # Save JSON file
    outPath.write_text(json.dumps(jsonDict, indent=4))
    print(f"Output JSON file written to: {outPath}")


def main_json_datasets():
    """Generates JSON dataset files for all 5 folds."""
    for fold in range(5):
        process_fold(fold)


if __name__ == "__main__":
    main_json_datasets()
