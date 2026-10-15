# Datasets for HLPpred 2.0

This directory contains the curated training and independent hold-out test datasets for both the **binary classification** (hemolytic vs. non-hemolytic) and **potency regression** ($\text{p}\text{HC}_{50}$) tasks.

---

## 1. Classification Datasets (`datasets/classification/`)

* **`Training.fasta` ($N = 1,540$ peptides):**
  * 770 positive (hemolytic) peptides
  * 770 negative (non-hemolytic) peptides
  * Employed for feature representation screening, baseline model training, and 5-fold cross-validation.
* **`Independent.fasta` ($N = 386$ peptides):**
  * 179 positive (hemolytic) peptides
  * 207 negative (non-hemolytic) peptides
  * Strictly held-out independent test set used for final unbiased performance estimation and external tool benchmarking.

---

## 2. Regression Datasets (`datasets/regression/`)

* **`Training_reg.fasta` ($N = 1,540$ peptides):**
  * 1,540 peptides annotated with half-maximal hemolytic concentrations ($\text{HC}_{50}$) and negative logarithmic potencies ($\text{p}\text{HC}_{50}$).
* **`Independent_reg.fasta` ($N = 386$ peptides):**
  * 386 peptides annotated with experimental $\text{HC}_{50}$ and $\text{p}\text{HC}_{50}$ values.

### FASTA Header Format
Headers in the regression datasets follow this standardized format:
```text
>Positive_1_HC50_15.00_pHC50_4.8239
IISTIGKLVKWIIDTV
```
* **ID:** `Positive_1`
* **$\text{HC}_{50}$:** $15.00\ \mu\text{M}$ (micromolar concentration causing 50% lysis of red blood cells)
* **$\text{p}\text{HC}_{50}$:** $4.8239$

### Mathematical Transformation
$$\text{p}\text{HC}_{50} = -\log_{10}\left(\text{HC}_{50}\,[\mu\text{M}] \times 10^{-6}\right) = 6.0 - \log_{10}\left(\text{HC}_{50}\,[\mu\text{M}]\right)$$
$$\text{HC}_{50}\,[\mu\text{M}] = 10^{(6.0 - \text{p}\text{HC}_{50})}$$
