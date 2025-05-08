import os
import torch.utils.data
import numpy as np
from utils.load_metrics import LoadMetrics
from utils.my_utils import *
from pathlib import Path
from tqdm import tqdm
import json
from utils.nifti_io import save_nifti
from utils.my_metrics import MaskedDiceMetric
from time import sleep


class Tester:
    def __init__(self,
                 args: dict,
                 model,
                 dataloader: torch.utils.data.DataLoader,
                 device: str = 'cuda'):
        sleep(0.5)  # for cleaner prints
        self.args = args
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.save_prediction = True
        self.save_analysis = True
        self.root_path = Path(os.path.dirname(os.path.dirname(__file__))).parent
        self.pred_output_dir = self.root_path.joinpath("outputs")
        self.final_activation = getattr(self.args, 'final_activation', False)
        self.masked_dice = MaskedDiceMetric(include_background=False,
                                            reduction="mean",
                                            get_not_nans=False,
                                            remove_right_brain=True
                                            )
        self.amp_dtype_str = getattr(self.args, 'amp_dtype', "torch.float32")
        self.amp_dtype = getattr(torch, self.amp_dtype_str, None)
        self.batch_path = dict()
        self.curr_case_idx = None
        self.curr_affine_matrix = None
        self.experiment_name = getattr(self.args, 'experiment_name', None)
        self.model_name = "agynet" if "agynet" in self.experiment_name else "swin_unetr"
        self.region_name = ""
        if self.model_name == "agynet":
            self.region_name = "_broca" if "broca" in self.experiment_name else "_wernicke"

    def load_metrics(self):
        self.metrics_obj = LoadMetrics(self.args)
        self.metrics_obj.create_metrics()

        # Creating a dictionary to keep metrics values
        self.metrics_values = {k: [] for k in self.metrics_obj.metrics}
        self.metrics_values["LeftHemisphereMaskedDice"] = []

    def compute_metrics(self, y, y_pred):
        y, y_pred = y.detach().cpu(), y_pred.detach().cpu()
        for metricName, metricFunc in self.metrics_obj.metrics.items():
            self.metrics_values[metricName].append(metricFunc(y_pred=y_pred, y=y).numpy())

    def compute_mean_metrics(self):
        for metricName, metricFunc in self.metrics_obj.metrics.items():
            self.metrics_values[metricName] = np.mean(self.metrics_values[metricName], axis=0)

        if self.metrics_values["LeftHemisphereMaskedDice"]:
            tensors = [tensor.detach().cpu().numpy() for tensor in self.metrics_values["LeftHemisphereMaskedDice"]]
            processed_tensors = []
            for tensor in tensors:
                if tensor.size == 1:
                    processed_tensors.append(np.array([[tensor.item()]]))
                elif tensor.ndim == 1:
                    processed_tensors.append(tensor.reshape(1, -1))
                else:
                    processed_tensors.append(tensor)
            mean_value = np.mean(np.vstack(processed_tensors), axis=0)
            self.metrics_values["LeftHemisphereMaskedDice"] = mean_value.reshape(1, -1)

    def get_case_idx(self, _dict):
        """
        Extract case index from batch path for saving predictions in the correct directory.

        This method traverses up the directory tree from the given path until it finds
        a directory with a name matching "case_X" format, then extracts X as the case index.

        Args:
            _dict (dict): Dictionary containing filepaths associated with the current batch.
                         Only the first path in the dictionary is used.

        Returns:
            None: Sets self.curr_case_idx with the extracted case index if found.
                  Returns early if batch_size is not 1.

        Note:
            Requires self.root_path to be set as the stopping point for directory traversal.
            Only works when batch_size=1 as larger batches may contain multiple cases.
        """

        path = None
        for key in _dict:
            path = Path(_dict[key])
            break

        while path != self.root_path:
            if "case_" in path.name:
                self.curr_case_idx = int(path.name.removeprefix("case_"))
                return
            path = path.parent

    def get_affine_matrix(self, batch):
        """
        Extract the affine transformation matrix from the input image.

        The affine matrix is used when saving NIfTI files to preserve the
        correct spatial orientation and voxel-to-world coordinate mapping.

        Args:
            batch (dict): Dictionary containing model inputs with an 'img' key
                         that has an affine attribute.

        Returns:
            None: Sets self.curr_affine_matrix with the extracted affine matrix.
                  Returns early if batch_size is not 1.

        Note:
            Requires batch_size=1, as multiple images may have different affine matrices.
            The affine matrix is extracted from the first image in the batch.
        """

        if batch["img"].shape[0] != 1:
            raise ValueError("Batch size must be 1 to extract affine matrix.")

        self.curr_affine_matrix = np.squeeze(batch["img"].affine.cpu().detach().numpy())

    def write_output(self, data: dict):
        OUTPUT_DIR = self.root_path.joinpath("outputs").joinpath("tests_analysis")

        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_filename = f"{self.args.project_name}{self.args.experiment_number}_" \
                          f"{self.args.experiment_name}_test.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        print(f"Writing output to {output_filepath}")
        # Load output file
        if os.path.exists(output_filepath):
            # pylint: disable=C0103
            with open(output_filepath, 'r', encoding='utf-8') as f:
                all_output_data = json.load(f)
        else:
            all_output_data = {}

        # Add new data and write to file
        all_output_data.update(data)

        # correction
        all_output_data = {key: str(val) for key, val in all_output_data.items()}

        # pylint: disable=C0103
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)

    def create_analysis(self):
        self.write_output(data=self.metrics_values)

    @staticmethod
    def probability2onehot(y_pred):
        """
        Convert the model's output to one-hot encoding.
        """

        y_pred_seg = torch.permute(
            torch.nn.functional.one_hot(
                torch.argmax(y_pred, dim=1),
                num_classes=y_pred.shape[1]
            ),
            dims=(0, 4, 1, 2, 3)
        ).to(torch.int)

        return y_pred_seg

    def eval(self, batch):
        # Change model to eval mode
        self.model.eval()

        for key in batch:
            batch[key] = batch[key].to(self.device)

        # Calculating segmentation prediction from model & one-hot vectoring
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
            with torch.no_grad():
                batch["y_pred"] = self.model(batch["img"])

                if self.final_activation:
                    batch["y_pred"] = torch.nn.functional.softmax(batch["y_pred"], dim=1)

                batch["y_pred_seg"] = self.probability2onehot(batch["y_pred"])

        # Compute metrics
        for i in range(batch["gt"].shape[0]):
            self.compute_metrics(y=batch["gt"][i].unsqueeze(0), y_pred=batch["y_pred_seg"][i].unsqueeze(0))
            self.metrics_values["LeftHemisphereMaskedDice"].append(
                self.masked_dice(y=batch["gt"][i], y_pred=batch["y_pred_seg"][i],
                                 mask=batch["brain_mask"][i]))

        # Save prediction if necessary
        if self.save_prediction:
            self.save_prediction_results_new(batch)

    def save_prediction_results_new(self, batch):
        """
        Save model predictions as NIfTI files.

        Saves:
        - Raw model prediction (probabilities)
        - Segmentation prediction (one-hot encoded)
        """
        # Extract data from batch with correct key names
        y_pred = batch["y_pred"].detach().cpu().numpy()[0]
        y_pred = np.squeeze(np.transpose(y_pred, (1, 2, 3, 0)))

        y_pred_seg = batch["y_pred_seg"].detach().cpu().numpy()[0]
        y_pred_seg = np.squeeze(np.transpose(y_pred_seg, (1, 2, 3, 0)))

        # eliminate background class
        y_pred = y_pred[..., 1:]
        y_pred_seg = y_pred_seg[..., 1:]

        # Get case info
        self.get_case_idx(self.batch_path)
        self.get_affine_matrix(batch)

        # Create directory and save files
        path = Path(self.pred_output_dir)
        path = path.joinpath(self.model_name)
        if not path.exists():
            path.mkdir(exist_ok=True, parents=True)

        # Save y_pred and y_pred_seg
        save_nifti(y_pred, self.curr_affine_matrix,
                   path.joinpath(f"case_{self.curr_case_idx}").joinpath(
                       f"predictions_probability{self.region_name}.nii.gz"))
        save_nifti(y_pred_seg, self.curr_affine_matrix,
                   path.joinpath(f"case_{self.curr_case_idx}").joinpath(
                       f"predictions{self.region_name}.nii.gz"))

    def test_loop(self):

        # Load metrics
        self.load_metrics()

        loader = tqdm(self.dataloader)

        for batch_idx, batch in enumerate(loader):
            self.batch_num = batch_idx + 1
            loader.set_description(f'Batch {self.batch_num} in process')

            self.batch_path = self.dataloader.dataset.data[batch_idx]

            # Evaluate data
            self.eval(batch=batch)

        self.compute_mean_metrics()

        if self.save_analysis:
            self.create_analysis()
