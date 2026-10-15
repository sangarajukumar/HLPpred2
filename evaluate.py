#!/usr/bin/env python3
"""
HLPpred 2.0 Evaluation & Benchmark Reproduction Script
======================================================
This script loads the pre-trained 5-fold ensemble checkpoints and pre-computed test features
to independently evaluate and reproduce all published benchmark metrics on the independent
test set (N = 386) for both Classification and Potency Regression (pHC50).
"""

import os
import sys
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Add local package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.utils import get_config
from src.lightning_module import Hemopi_Module, Hemopi_Regression_Module


def evaluate_classification(device):
    print("\n" + "=" * 65)
    print("  EVALUATING CLASSIFICATION BENCHMARK (Independent Test Set, N = 386)")
    print("=" * 65)
    
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", "classification")
    config_path = os.path.join(ckpt_dir, "config.yaml")
    config = get_config(config_path)
    
    # 1. Load checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    print(f"Loading {len(ckpt_files)} classification fold checkpoints...")
    
    models = []
    for ckpt_path in ckpt_files:
        model = Hemopi_Module(config.model, config.trainer)
        state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        models.append(model)
        
    # 2. Load FASTA and true labels
    fasta_path = os.path.join(BASE_DIR, "datasets", "classification", "Independent.fasta")
    names, seqs, true_labels = [], [], []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        names.append(rec.id)
        seqs.append(str(rec.seq).upper())
        true_labels.append(1 if "Positive" in rec.id else 0)
    true_labels = np.array(true_labels)
    
    # 3. Load pre-computed features
    # Top 300 base model probabilities
    prob_path = os.path.join(BASE_DIR, "features", "classification", "Top300-Proba-Fe_Test.txt")
    prob_dict = {}
    with open(prob_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 301 and parts[0].startswith(">"):
                name = parts[0][1:]
                prob_dict[name] = torch.tensor([float(x) for x in parts[1:301]], dtype=torch.float32)
                
    # MERCI scores
    merci_path = os.path.join(BASE_DIR, "features", "motif", "test_merci_scores.txt")
    merci_dict = {}
    with open(merci_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                merci_dict[parts[0]] = float(parts[1])
                
    # 4. Run ensemble inference
    seqvec_dir = os.path.join(BASE_DIR, "features", "seqvec_test")
    max_length = config.dataset.max_length  # 39
    
    ensemble_probs = []
    with torch.no_grad():
        for name in names:
            pt_path = os.path.join(seqvec_dir, f"{name}.pt")
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Missing SeqVec embedding: {pt_path}")
            x_seqvec = torch.load(pt_path, map_location=device)
            if isinstance(x_seqvec, np.ndarray):
                x_seqvec = torch.tensor(x_seqvec, dtype=torch.float32, device=device)
            else:
                x_seqvec = x_seqvec.to(device)
                
            len_tokens = x_seqvec.shape[0]
            x_padded = F.pad(x_seqvec, (0, 0, 0, max_length - len_tokens), value=0).unsqueeze(0)
            mask = torch.ones(max_length, dtype=torch.bool, device=device)
            mask[:len_tokens] = False
            mask = mask.unsqueeze(0)
            
            x_probs = prob_dict[name].unsqueeze(0).to(device)
            x_merci = torch.tensor([[merci_dict.get(name, 0.0)]], dtype=torch.float32, device=device)
            
            fold_probs = []
            for model in models:
                logits = model([x_padded], mask, x_probs, x_merci)
                probs = F.softmax(logits, dim=-1)
                fold_probs.append(probs[0, 1].item())
            ensemble_probs.append(np.mean(fold_probs))
            
    ensemble_probs = np.array(ensemble_probs)
    preds = (ensemble_probs >= 0.5).astype(int)
    
    # 5. Compute metrics
    acc = accuracy_score(true_labels, preds) * 100
    auc = roc_auc_score(true_labels, ensemble_probs)
    mcc = matthews_corrcoef(true_labels, preds)
    tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
    sn = (tp / (tp + fn)) * 100
    sp = (tn / (tn + fp)) * 100
    
    print("\n[REPRODUCED CLASSIFICATION METRICS (Table 3)]")
    print(f"  • Accuracy (ACC):       {acc:.2f}% (Target: 88.34%)")
    print(f"  • Sensitivity (Sn):     {sn:.2f}% (Target: 86.59%)")
    print(f"  • Specificity (Sp):     {sp:.2f}% (Target: 89.86%)")
    print(f"  • MCC:                  {mcc:.4f} (Target: 0.7654)")
    print(f"  • AUC-ROC:              {auc:.4f} (Target: 0.9546)")
    print(f"  • Confusion Matrix:     TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("=" * 65 + "\n")


def evaluate_regression(device):
    print("\n" + "=" * 65)
    print("  EVALUATING REGRESSION BENCHMARK (Independent Test Set, N = 386)")
    print("=" * 65)
    
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", "regression")
    config_path = os.path.join(ckpt_dir, "config.yaml")
    config = get_config(config_path)
    
    # Ensure dimensions match regression head
    config.model.prob_feature_dim = 360
    config.model.merci_feature_dim = 0
    
    # 1. Load checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    print(f"Loading {len(ckpt_files)} regression fold checkpoints...")
    
    models = []
    for ckpt_path in ckpt_files:
        model = Hemopi_Regression_Module(config.model, config.trainer)
        state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        models.append(model)
        
    # 2. Load FASTA and true pHC50
    fasta_path = os.path.join(BASE_DIR, "datasets", "regression", "Independent_reg.fasta")
    names, seqs, true_phc50 = [], [], []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        names.append(rec.id)
        seqs.append(str(rec.seq).upper())
        parts = rec.id.split("_")
        true_phc50.append(float(parts[5]))
    true_phc50 = np.array(true_phc50)
    
    # 3. Load pre-computed Top360 tabular features
    csv_path = os.path.join(BASE_DIR, "features", "regression", "Top360_Meta_Fe_Test.csv")
    df_feat = pd.read_csv(csv_path)
    feature_cols = df_feat.columns[1:-2]  # drop Sequence, HC50, pHC50
    
    prob_dict = {}
    for idx, name in enumerate(names):
        prob_dict[name] = torch.tensor(
            df_feat.iloc[idx][feature_cols].values.astype(np.float32)
        )
        
    # 4. Run ensemble inference
    seqvec_dir = os.path.join(BASE_DIR, "features", "seqvec_test")
    max_length = config.dataset.max_length  # 39
    
    ensemble_preds = []
    with torch.no_grad():
        for name in names:
            # Map name like Positive_1_HC50_15.00_pHC50_4.8239 to Positive_1.pt
            parts = name.split("_")
            short_id = f"{parts[0]}_{parts[1]}"
            pt_path = os.path.join(seqvec_dir, f"{short_id}.pt")
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Missing SeqVec embedding: {pt_path}")
            x_seqvec = torch.load(pt_path, map_location=device)
            if isinstance(x_seqvec, np.ndarray):
                x_seqvec = torch.tensor(x_seqvec, dtype=torch.float32, device=device)
            else:
                x_seqvec = x_seqvec.to(device)
                
            len_tokens = x_seqvec.shape[0]
            x_padded = F.pad(x_seqvec, (0, 0, 0, max_length - len_tokens), value=0).unsqueeze(0)
            mask = torch.ones(max_length, dtype=torch.bool, device=device)
            mask[:len_tokens] = False
            mask = mask.unsqueeze(0)
            
            x_probs = prob_dict[name].unsqueeze(0).to(device)
            x_merci = torch.zeros((1, 0), dtype=torch.float32, device=device)
            
            fold_preds = []
            for model in models:
                pred = model([x_padded], [mask], x_probs, x_merci)
                fold_preds.append(pred.item())
            ensemble_preds.append(np.mean(fold_preds))
            
    ensemble_preds = np.array(ensemble_preds)
    
    # 5. Compute regression metrics
    r2 = r2_score(true_phc50, ensemble_preds)
    rmse = np.sqrt(mean_squared_error(true_phc50, ensemble_preds))
    mae = mean_absolute_error(true_phc50, ensemble_preds)
    pearson_r, pval = pearsonr(true_phc50, ensemble_preds)
    mse = mean_squared_error(true_phc50, ensemble_preds)
    
    print("\n[REPRODUCED REGRESSION METRICS (Table 4)]")
    print(f"  • Coefficient of Det. (R²): {r2:.4f} (Target: 0.5939)")
    print(f"  • Root Mean Sq. Err (RMSE): {rmse:.4f} (Target: 0.4057)")
    print(f"  • Mean Absolute Err (MAE):  {mae:.4f} (Target: 0.3003)")
    print(f"  • Pearson Correlation (R):  {pearson_r:.4f} (Target: 0.7717, P = {pval:.2e})")
    print(f"  • Mean Squared Error (MSE): {mse:.4f} (Target: 0.1646)")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="HLPpred 2.0 Benchmark Reproduction Script")
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="both",
                        help="Task to evaluate (default: both)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device (default: cuda if available, else cpu)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if args.task in ["classification", "both"]:
        evaluate_classification(device)
    if args.task in ["regression", "both"]:
        evaluate_regression(device)

if __name__ == "__main__":
    main()
