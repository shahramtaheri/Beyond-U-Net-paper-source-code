# Beyond U-Net++: Residual and Attention-Enhanced Architecture for Pulmonary Nodule Segmentation

This repository provides the official PyTorch implementation of the model proposed in:

**“Beyond U-Net++: Residual and Attention-Enhanced Architecture for Pulmonary Nodule Segmentation”**  
(Submitted to *Neurocomputing*)

The proposed framework enhances U-Net++ by integrating residual blocks and attention gates, together with a hybrid Dice–Focal loss, to improve pulmonary nodule segmentation in CT images.

---

## Overview

Key contributions of this work include:
- Residual blocks integrated into the U-Net++ encoder–decoder pathway to improve training stability and feature propagation.
- Attention gates applied to skip connections to suppress irrelevant background and emphasize salient nodule regions.
- A hybrid Dice–Focal loss to address class imbalance and ambiguous nodule boundaries.
- Comprehensive evaluation on the LIDC-IDRI dataset using five-fold cross-validation.
- Detailed ablation study and computational complexity analysis (parameters, FLOPs, training time).

---

## Repository Structure

Beyond-U-Net-plusplus/
├── README.md
├── requirements.txt
├── configs/
├── scripts/
├── src/
│ ├── models/
│ ├── datasets/
│ ├── losses/
│ ├── metrics/
│ └── utils/
├── data/
│ └── README.md
├── results/
└── outputs/

yaml
Copy code

---

## Installation

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (recommended)
- MONAI
- Albumentations
- NumPy, SciPy, scikit-image

Install dependencies:
```bash
pip install -r requirements.txt
Dataset Preparation
LIDC-IDRI
This project uses the LIDC-IDRI dataset. Due to licensing restrictions, the dataset is not distributed with this repository.

Please download LIDC-IDRI from:
https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

Data Curation and Preprocessing
The preprocessing pipeline follows exactly the procedure described in the manuscript:

CT volumes are resampled to 1 mm × 1 mm × 1 mm isotropic resolution.

Only nodules annotated by at least three of four radiologists are retained.

Nodules smaller than 3 mm are excluded.

3D-to-2D conversion:

All axial slices containing foreground mask voxels are included.

Multi-slice nodules: all slices are preserved.

Multi-nodule slices: masks are combined.

No interpolation or slice skipping is performed.

Lung-field-based cropping:

A coarse lung mask is generated using HU thresholding and morphological operations.

A 512 × 512 crop is extracted centered on the lung-field bounding box.

Nodule centroid information is used only to identify nodule-bearing slices, not for cropping.

Run preprocessing:

bash
Copy code
python scripts/prepare_lidc.py --config configs/default.yaml
Cross-Validation Splits
Five-fold cross-validation is performed with patient-level stratification to avoid data leakage.

Generate splits:

bash
Copy code
python scripts/create_folds.py --config configs/default.yaml
Training
Train the model for a specific fold:

bash
Copy code
python -m src.train --config configs/lidc_2d_fold1.yaml
Training details:

Optimizer: Adam

Initial learning rate: 1e-4

Scheduler: ReduceLROnPlateau

Early stopping patience: 15 epochs

Maximum epochs: 100

Batch size: configurable via YAML

Evaluation
Evaluate a trained model:

bash
Copy code
python -m src.evaluate \
  --config configs/lidc_2d_fold1.yaml \
  --ckpt path/to/best_model.pt
Metrics reported:

Dice Similarity Coefficient (DSC)

Intersection over Union (IoU)

Sensitivity

Positive Predictive Value (PPV)

Hausdorff Distance (HD)

95th percentile Hausdorff Distance (HD95)

Average Symmetric Surface Distance (ASSD)

Inference
Run inference on a single image or directory:

bash
Copy code
python -m src.infer \
  --ckpt path/to/best_model.pt \
  --input path/to/images \
  --output path/to/save_predictions
Model Complexity
The repository includes scripts to reproduce the computational complexity analysis reported in the paper:

Number of parameters

FLOPs (512 × 512 input)

Training time per epoch (RTX 3090)

See results/complexity_table.csv.

Reproducibility
All experiments were conducted with fixed random seeds:

Python / NumPy seed: 42

PyTorch CPU & CUDA seed: 42

Cross-validation split seed: 1234

Data augmentation seed: 2023

cuDNN deterministic mode enabled

cuDNN benchmarking disabled

These settings ensure deterministic and reproducible results across runs.

Results
Quantitative results, ablation studies, and complexity comparisons reported in the manuscript can be reproduced using the scripts provided. Example tables and figures are stored under results/.

Citation
If you use this code, please cite:

bibtex
Copy code
@article{BeyondUNetPP2025,
  title={Beyond U-Net++: Residual and Attention-Enhanced Architecture for Pulmonary Nodule Segmentation},
  author={Golrizkhatami, Zahra and Taheri, Shahram},
  journal={Neurocomputing},
  year={2025}
}
License
This project is released under the MIT License. See LICENSE for details.

Contact
For questions or issues, please contact:
Zahra Golrizkhatami
z.golrizkhatami@antalya.edu.tr
