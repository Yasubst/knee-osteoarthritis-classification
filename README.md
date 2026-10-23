# Knee Osteoarthritis Severity Classification Using Deep Learning

## Project Overview

This project explores automated knee osteoarthritis severity classification from X-ray images using deep learning.

The objective is to classify knee X-ray images into five Kellgren–Lawrence (KL) grades, ranging from KL0 to KL4.

The project includes knee region-of-interest (ROI) extraction, dataset analysis, preprocessing, ResNet18 training, cross-validation, and model evaluation.

## Dataset

The classification dataset contains **2,352 grayscale knee X-ray images**.

| KL Grade | Number of Images |
|---|---:|
| KL0 | 932 |
| KL1 | 428 |
| KL2 | 615 |
| KL3 | 306 |
| KL4 | 71 |
| **Total** | **2,352** |

The dataset is imbalanced, with considerably fewer KL4 images.

## Methodology

1. Extract knee regions of interest from sample DICOM images using BoneFinder landmarks.
2. Analyze image distributions and preprocessing.
3. Split the classification dataset into development (80%) and held-out test (20%) sets using stratification.
4. Perform stratified 5-fold cross-validation on the development set.
5. Train a baseline ResNet18 model from scratch.
6. Train an ImageNet-pretrained ResNet18 model.
7. Compare models using balanced accuracy, ROC-AUC, and average precision.
8. Evaluate the selected pretrained model on the held-out test set.

## Model Results

| Metric | Baseline ResNet18 | Pretrained ResNet18 |
|---|---:|---:|
| Mean CV balanced accuracy | 21.21% | 51.15% |
| Mean CV ROC-AUC | 0.5837 | 0.7937 |
| Mean CV average precision | 0.2458 | 0.5530 |

The pretrained ResNet18 achieved substantially better cross-validation results than the model trained from scratch.

### Held-Out Test Results

The pretrained model selected from fold 3 achieved:

| Metric | Result |
|---|---:|
| Accuracy | 43.52% |
| Balanced accuracy | 46.17% |
| ROC-AUC | 0.7646 |
| Average precision | 0.5109 |

## Visualizations

### Dataset Distribution

![KL Grade Distribution](outputs/plots/kl_grade_distribution.png)

### Pretrained Model — Fold 3 Training

![Balanced Accuracy](outputs/pretrained/fold_3_balanced_accuracy.png)

![ROC-AUC](outputs/pretrained/fold_3_roc_auc.png)

### Test Confusion Matrix

![Test Confusion Matrix](outputs/pretrained/test_confusion_matrix.png)

Additional training plots and evaluation CSV files are available in the `outputs/baseline/` and `outputs/pretrained/` directories.

## Technologies Used

- Python
- PyTorch and Torchvision
- Scikit-learn
- Pandas and NumPy
- Matplotlib
- Pydicom
- Pillow
- PyCharm
- Git and GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/Yasubst/knee-osteoarthritis-classification.git
cd knee-osteoarthritis-classification
```

Create a Python virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the requirements:

```bash
python -m pip install -r requirements.txt
```

The dataset and trained model checkpoints are not included in the repository.

## Limitations

- The dataset has a substantial class imbalance.
- Training results suggest overfitting.
- Patient-level independence between dataset splits has not yet been verified.
- The model is intended for educational research, not clinical diagnosis.

## Future Work

- Improve generalization and minority-class performance.
- Investigate patient-level data splitting.
- Explore additional model architectures.
- Implement tibia segmentation using U-Net.
- Compare classification and segmentation-based approaches.

## Project Status

**Knee osteoarthritis classification:** Completed initial experiments and evaluation.

**Tibia segmentation:** Planned.

This is an educational deep learning project developed as part of practical machine learning and medical image analysis studies.