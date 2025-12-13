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


## Installation

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (recommended)
- MONAI
- Albumentations
- NumPy, SciPy, scikit-image

Install dependencies:
