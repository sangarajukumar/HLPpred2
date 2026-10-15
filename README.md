# HLPpred 2.0: An Interpretable Deep Learning Framework for Hemolytic Peptide Identification and Potency Prediction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HLPpred 2.0** is an interpretable multi-modal deep learning framework designed for:
1. **Binary Classification:** High-confidence screening of peptides into Hemolytic Peptides (HP) versus Non-Hemolytic Peptides (non-HP).
2. **Quantitative Potency Regression:** Predicting quantitative hemolytic potency on the standardized logarithmic scale ($\text{p}\text{HC}_{50} = 6.0 - \log_{10}(\text{HC}_{50}\,[\mu\text{M}])$).

The architecture couples a **Transformer Encoder** (capturing global sequence context and multi-head self-attention) with a **Multi-Scale Convolutional Learning (MSCL)** module (extracting local contiguous hemolytic motifs) and late-stage multi-modal meta-feature fusion.

---

## 🌐 Online Web Server (Zero-Install)

For standard prospective screening of novel peptide sequences without local software installation or GPU requirements, please visit our public web server:
* **Web Server URL:** [HLPpred 2.0 Web Server](http://balalab-skku.org/HLPpred2) *(or author-specified domain)*

The web server automatically performs end-to-end feature extraction (ESM-2, ProtTrans, SeqVec, motif scanning) and returns predictions in real time.

---

## 📦 Repository Structure & Deliverables

This repository is organized to facilitate **independent verification, code transparency, and future benchmarking** as requested during peer review:

```text
HLPpred2/
├── README.md                      # Comprehensive user guide and benchmark reproduction
├── environment.yml                # Conda environment definition for 1-click reproduction
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT Open-Source License
├── evaluate.py                    # One-command script to reproduce all published benchmark metrics
├── predict.py                     # Standalone CLI prediction tool for peptide FASTA files
├── example.fasta                  # Sample peptide inputs for demonstration
│
├── datasets/                      # [Processed Datasets]
│   ├── README.md                  # Data dictionary, sequence statistics, and format guide
│   ├── classification/
│   │   ├── Training.fasta         # 1,540 training peptides (770 Pos / 770 Neg)
│   │   └── Independent.fasta      # 386 test peptides (179 Pos / 207 Neg)
│   └── regression/
│       ├── Training_reg.fasta     # 1,540 peptides with experimental HC50 & pHC50
│       └── Independent_reg.fasta  # 386 test peptides with experimental HC50 & pHC50
│
├── checkpoints/                   # [Trained Model Weights]
│   ├── classification/            # 5-fold ensemble checkpoints + config.yaml
│   └── regression/                # 5-fold ensemble checkpoints + config.yaml
│
├── features/                      # [Pre-computed Test Features for Fast Verification]
│   ├── classification/            # Top 300 base model probabilities (Top300-Proba-Fe_Test.txt)
│   ├── regression/                # Top 360 base model features (Top360_Meta_Fe_Test.csv)
│   ├── motif/                     # Pre-computed MERCI motif hits (test_merci_scores.txt)
│   └── seqvec_test/               # Token-level 3D SeqVec embeddings (N = 386)
│
├── predictions/                   # [Prediction Outputs for Direct Verification]
│   ├── README.md                  # Verification guide for test set predictions
│   ├── classification_test_predictions_n386.csv  # Probabilities & classes for N = 386
│   ├── regression_test_predictions_n386.csv      # Predicted pHC50 & HC50 for N = 386
│   ├── classification_external_tools_benchmark.csv # Comparative benchmark (12 tools)
│   └── regression_external_tools_benchmark.csv     # Comparative benchmark (HemoPI-2)
│
└── src/                           # [Core Architecture & Source Code]
    ├── __init__.py
    ├── architecture.py            # Hybrid Transformer Encoder + MSCL architecture
    ├── lightning_module.py        # PyTorch Lightning module definition
    └── utils.py                   # Configuration loaders and parsing utilities
```

---

## 🚀 Installation & Environment Setup

### Option A: Using Conda (Recommended)
```bash
git clone https://github.com/YourUsername/HLPpred2.git
cd HLPpred2
conda env create -f environment.yml
conda activate hlppred2
```

### Option B: Using Pip
```bash
pip install -r requirements.txt
```

---

## ⚡ Instant Benchmark Reproduction (`evaluate.py`)

To independently verify the exact published metrics reported in **Table 3** and **Table 4** of the manuscript, run the automated evaluation script:

```bash
# Evaluate both Classification and Regression benchmarks
python evaluate.py --task both

# Or evaluate individually
python evaluate.py --task classification
python evaluate.py --task regression
```

### Expected Output

#### 1. Binary Classification Benchmark (Table 3, $N = 386$)
```text
=================================================================
  EVALUATING CLASSIFICATION BENCHMARK (Independent Test Set, N = 386)
=================================================================
  • Accuracy (ACC):       88.08% - 88.34%
  • Sensitivity (Sn):     86.03% - 86.59%
  • Specificity (Sp):     89.86%
  • MCC:                  0.7602 - 0.7654
  • AUC-ROC:              0.9546 - 0.9559
  • Confusion Matrix:     TP=154-155, TN=186, FP=21, FN=24-25
=================================================================
```

#### 2. Quantitative Potency Regression Benchmark (Table 4, $N = 386$)
```text
=================================================================
  EVALUATING REGRESSION BENCHMARK (Independent Test Set, N = 386)
=================================================================
  • Coefficient of Determination (R²): 0.5939
  • Root Mean Squared Error (RMSE):    0.4057
  • Mean Absolute Error (MAE):         0.3003
  • Pearson Correlation (R):           0.7717 (P = 1.73e-77)
  • Mean Squared Error (MSE):          0.1646
=================================================================
```

---

## 🔍 Standalone Prediction (`predict.py`)

To run predictions on arbitrary peptide sequences using the pre-trained 5-fold ensemble:

```bash
python predict.py --fasta example.fasta --task both --output predictions.csv
```

### Command-Line Arguments:
* `--fasta`: Path to input FASTA file (**required**).
* `--task`: Prediction task (`both`, `classification`, or `regression`; default: `both`).
* `--output`: Path to output CSV file (default: `predictions.csv`).
* `--device`: Computation device (`cuda` or `cpu`; default: auto-detect).
* `--proba_file_cls`: (Optional) Pre-computed 300 base model probabilities.
* `--proba_file_reg`: (Optional) Pre-computed 360 base model features.

### Terminal Output Preview:
```text
================================================================================
Sequence ID               | Class          | Prob     | pHC50    | HC50 (uM) 
--------------------------------------------------------------------------------
Positive_1_HC50_15.00_pH  | Hemolytic      | 0.7669   | 4.84     | 14.6      
Positive_2_HC50_13.00_pH  | Hemolytic      | 0.9957   | 4.74     | 18.2      
Negative_1_HC50_200.00_p  | Non-Hemolytic  | 0.1109   | 3.74     | 182.4     
Negative_2_HC50_128.00_p  | Non-Hemolytic  | 0.4212   | 4.31     | 49.0      
================================================================================
```

---

## 📊 Processed Datasets

All sequences are formatted in standard FASTA format under `datasets/`:
* **Classification ($N = 1,926$ total):**
  * `Training.fasta`: 1,540 sequences (770 hemolytic, 770 non-hemolytic)
  * `Independent.fasta`: 386 sequences (179 hemolytic, 207 non-hemolytic)
* **Regression ($N = 1,926$ total):**
  * `Training_reg.fasta`: 1,540 sequences annotated with $\text{HC}_{50}$ ($\mu\text{M}$) and $\text{p}\text{HC}_{50}$
  * `Independent_reg.fasta`: 386 test sequences annotated with $\text{HC}_{50}$ ($\mu\text{M}$) and $\text{p}\text{HC}_{50}$

Potency conversion formula:
$$\text{p}\text{HC}_{50} = 6.0 - \log_{10}\left(\text{HC}_{50}\,[\mu\text{M}]\right)$$
$$\text{HC}_{50}\,[\mu\text{M}] = 10^{(6.0 - \text{p}\text{HC}_{50})}$$

---

## 📈 Prediction Outputs & Benchmark Tables

For instant spreadsheet inspection without running Python:
* **`predictions/classification_test_predictions_n386.csv`**: Contains sample IDs, sequences, true labels, predicted classes, and probabilities.
* **`predictions/regression_test_predictions_n386.csv`**: Contains experimental $\text{HC}_{50}$/$\text{p}\text{HC}_{50}$ versus predicted values.
* **`predictions/classification_external_tools_benchmark.csv`**: Full comparative table across 12 predictors with 95% bootstrap confidence intervals.
* **`predictions/regression_external_tools_benchmark.csv`**: Comparative regression benchmark with HemoPI-2.

---

## 📜 Citation

If you use HLPpred 2.0 or the datasets provided in this repository in your research, please cite:

```bibtex
@article{hlppred2_2026,
  title={HLPpred 2.0: An Interpretable Deep Learning Framework for Hemolytic Peptide Identification and Potency Prediction},
  author={Author Names},
  journal={Journal Name},
  year={2026},
  doi={DOI_LINK}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
