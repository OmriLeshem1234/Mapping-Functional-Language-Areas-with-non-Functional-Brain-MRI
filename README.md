# Mapping-Functional-Language-Areas-with-non-Functional-Brain-MRI

Official implementation of the paper:  
**Mapping Functional Language Areas with Non-Functional Brain MRI**  
Accepted to the Medical Imaging with Deep Learning (MIDL) 2025 conference

**Authors:**  
Omri Leshem, Atira Sara Bick, Nahum Kiryati, Netta Levin, Arnaldo Mayer  
Contact: Omri Leshem – omrileshem@mail.tau.ac.il  
Computational Imaging Lab (CILAB) - https://www.cilab.org.il/

This project introduces a deep learning framework for indirect segmentation of functional language areas (Broca and Wernicke) using only anatomical and diffusion MRI data, without relying on fMRI. The approach leverages multi-modal information from T1-weighted and DWI scans, enabling clinical applicability in settings where fMRI is unavailable or unreliable.

---

## 1. Setup

### Clone the repository
```bash
git clone https://github.com/OmriLeshem1234/Mapping-Functional-Language-Areas-with-non-Functional-Brain-MRI.git
```

### Navigate into the repository directory
```bash
cd Mapping-Functional-Language-Areas-with-non-Functional-Brain-MRI
```

### Create the conda environment
```bash
conda env create -f environment.yaml
```

### Activate the conda environment
```bash
conda activate fmri_env
```

---

## 2. Pre-trained Weights

Download model weights from the shared folder:

[model_weights – Google Drive link](https://drive.google.com/drive/u/4/folders/1NbgwwP-U-6f3rHkWj9JM_VY8gSa5uM97)

Copy the downloaded `model_weights` directory into the repository root:

```bash
cp -r /path/to/downloaded/model_weights ./model_weights
```

These pre-trained weights were trained on the HCP dataset using Brodmann area-derived functional labels. They serve as a starting point for fine-tuning on clinical or other domain-specific data.

Note: The `clinical_fMRI` directory contains model weights that were fine-tuned on actual clinical data using real fMRI-derived labels, as detailed in the paper.

After downloading and copying, the `model_weights/` directory should be placed directly under the project root. Its structure should appear as follows:

```
model_weights/                       
├── pre_trained_HCP/                 # Weights trained on HCP using Brodmann areas labels
│   ├── AGYnet/
│   │   ├── broca/
│   │   └── wernicke/
│   └── Swin-UNETR/
└── clinical_fMRI/                   # Fine-tuned weights using real fMRI-derived clinical labels
    ├── AGYnet/
    │   ├── broca/
    │   └── wernicke/
    └── Swin-UNETR/
```

Make sure the `model_weights/` folder is located at the top level of the repository, alongside files such as `main.py` and `environment.yaml`.

---

## 3. Atlas Files

This project uses the **SENSAAS atlas** for anatomical-functional alignment:

> Labache et al., *A SENtence Supramodal Areas AtlaS (SENSAAS) based on multiple task-induced activation mapping and graph analysis of intrinsic connectivity in 144 healthy right-handers*,  
> Brain Structure and Function (2019), Springer.  
> DOI: [10.1007/s00429-018-1810-2](https://doi.org/10.1007/s00429-018-1810-2)

Download the atlas folder from:

[sensaas_atlas – Google Drive link](https://drive.google.com/drive/folders/19r245NQPzGQSXtV6unnKmQanlooyhyfE?usp=drive_link)

Copy the downloaded `sensaas_atlas` folder into the `.data/` directory:

```bash
cp -r /path/to/downloaded/sensaas_atlas data/sensaas_atlas
```
After downloading and copying, the `data/` directory at the project root should be organized as follows:

```bash
data/
├── sensaas_atlas/
│   ├── broca_atlas.nii.gz
│   ├── wernicke_atlas.nii.gz
│   ├── SENSAAS.nii.gz
│   └── T1_FSL_atlas_2mm_RAS_MNI.nii.gz
├── case_lists/
└── json_datasets_examples/
```

Make sure the `sensaas_atlas/` folder is correctly placed under the `data/` directory. All four listed NIfTI files are required for atlas-based segmentation comparison.

---

## 4. Dataset Preparation

Generate dataset JSON files with:

```bash
python create_json_datasets.py
```

Output will be saved to:

```
./data/json_datasets/
```

Example JSON files are available under:

```
./data/json_datasets_examples/
```

> Note: The clinical dataset used in this project is currently unavailable for public release due to regulatory constraints.

---

## 5. Model Training

Train a model using:

```bash
python main.py --model <MODEL> [--region <REGION>]
```

### `--model` options:
- `swinunetr`
- `agynet` (requires `--region`)
- `all_models` (**recommended**: runs all models across all regions)

### `--region` (only required if `--model agynet`):
- `broca`
- `wernicke`

**Examples:**

```bash
python main.py --model all_models              # recommended
python main.py --model swinunetr
python main.py --model agynet --region broca
python main.py --model agynet --region wernicke
```

Trained weights will be saved automatically under:

```
./model_weights/experiment/
```

> **Training logs can be optionally logged to [Comet ML](https://www.comet.com/)**.  
> To enable Comet integration, set the `workspace` and `api_key` fields in the corresponding config file under:

```
training/config_files/
```

---

## 6. Model Evaluation (Inference)

Run test-time inference:

```bash
python main_test.py --model <MODEL> [--region <REGION>]
```

### `--model` options:
- `swinunetr`
- `agynet` (requires `--region`)
- `all_models` (**recommended**: evaluates all models for both regions)

**Examples:**

```bash
python main_test.py --model all_models         # recommended
python main_test.py --model swinunetr
python main_test.py --model agynet --region broca
python main_test.py --model agynet --region wernicke
```

Test results will be saved under the `./outputs/` directory, which is created automatically.

This includes:
- Saving predicted segmentation probabilities as NIfTI files (`.nii.gz`)
- Saving final binary segmentation masks as NIfTI files (`.nii.gz`)


---

## 7. Automatic Analysis After Testing

Once `main_test.py` has been executed for all models and both regions (broca and wernicke), the following analyses run automatically:

- Atlas-based segmentation comparison
- ROC curves and AUC computation per region/model
- Sensitivity margins analysis (5 mm clinical threshold)

All analysis outputs are stored under the `./outputs/` directory alongside the model predictions.

