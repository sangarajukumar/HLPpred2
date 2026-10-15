#!/usr/bin/env python3
"""
HLPpred 2.0 Standalone Inference Tool
=====================================
Predicts hemolytic activity (classification) and potency (pHC50 regression)
for peptide sequences using the 5-fold deep hybrid Transformer-MSCL ensemble.

Usage:
    python predict.py --fasta example.fasta --task both --output predictions.csv
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

# Add local package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.utils import get_config
from src.lightning_module import Hemopi_Module, Hemopi_Regression_Module


def load_classification_models(device):
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", "classification")
    config_path = os.path.join(ckpt_dir, "config.yaml")
    config = get_config(config_path)
    
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No classification checkpoints in {ckpt_dir}")
        
    models = []
    for ckpt_path in ckpt_files:
        model = Hemopi_Module(config.model, config.trainer)
        state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        models.append(model)
        
    return models, config


def load_regression_models(device):
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", "regression")
    config_path = os.path.join(ckpt_dir, "config.yaml")
    config = get_config(config_path)
    config.model.prob_feature_dim = 360
    config.model.merci_feature_dim = 0
    
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No regression checkpoints in {ckpt_dir}")
        
    models = []
    for ckpt_path in ckpt_files:
        model = Hemopi_Regression_Module(config.model, config.trainer)
        state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        models.append(model)
        
    return models, config


def get_seqvec_embedding(name, seq, device):
    """
    Attempts to retrieve cached SeqVec embedding or dynamically compute using bio_embeddings.
    """
    # 1. Try local cache
    cached_dir = os.path.join(BASE_DIR, "features", "seqvec_test")
    # Clean ID if contains attributes
    short_id = name.split()[0]
    if "_" in short_id:
        p = short_id.split("_")
        candidate = f"{p[0]}_{p[1]}"
    else:
        candidate = short_id
        
    for try_id in [short_id, candidate]:
        pt_path = os.path.join(cached_dir, f"{try_id}.pt")
        if os.path.exists(pt_path):
            x = torch.load(pt_path, map_location=device)
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=torch.float32, device=device)
            return x.to(device)
            
    # 2. Try dynamic bio_embeddings extraction
    try:
        from bio_embeddings.embed import SeqVecEmbedder
        embedder = SeqVecEmbedder()
        emb = embedder.embed(seq)
        if isinstance(emb, (list, tuple)) or hasattr(emb, '__iter__'):
            emb = np.array(list(emb))
        if emb.ndim == 3:
            emb = emb.mean(axis=0)  # [L, 1024]
        return torch.tensor(emb, dtype=torch.float32, device=device)
    except Exception as e:
        print(f"Warning: Could not compute dynamic SeqVec embedding for {name} ({e}).")
        print("Note: To run de novo sequences without cached embeddings, please use the public Web Server or install bio_embeddings.")
        # Fallback to zero embedding of length len(seq)
        return torch.zeros((len(seq), 1024), dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser(description="HLPpred 2.0 Peptide Prediction Tool")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="both",
                        help="Prediction task (default: both)")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV file")
    parser.add_argument("--proba_file_cls", default=None, help="Optional pre-computed 300 base model probabilities")
    parser.add_argument("--proba_file_reg", default=None, help="Optional pre-computed 360 base model features")
    parser.add_argument("--merci_file", default=None, help="Optional pre-computed MERCI motif scores")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for computation")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"\n[HLPpred 2.0] Initializing inference on {device}...")
    
    # 1. Parse FASTA
    records = list(SeqIO.parse(args.fasta, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {args.fasta}")
    print(f"Loaded {len(records)} sequence(s) from {args.fasta}")
    
    # 2. Pre-load cached feature tables if available
    cls_prob_dict = {}
    default_cls_prob = os.path.join(BASE_DIR, "features", "classification", "Top300-Proba-Fe_Test.txt")
    cls_file = args.proba_file_cls or (default_cls_prob if os.path.exists(default_cls_prob) else None)
    if cls_file and os.path.exists(cls_file):
        with open(cls_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 301 and parts[0].startswith(">"):
                    cls_prob_dict[parts[0][1:]] = torch.tensor([float(x) for x in parts[1:301]], dtype=torch.float32)
                    
    merci_dict = {}
    default_merci = os.path.join(BASE_DIR, "features", "motif", "test_merci_scores.txt")
    merci_file = args.merci_file or (default_merci if os.path.exists(default_merci) else None)
    if merci_file and os.path.exists(merci_file):
        with open(merci_file, "r") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) == 2:
                    merci_dict[p[0]] = float(p[1])
                    
    reg_prob_dict = {}
    default_reg_prob = os.path.join(BASE_DIR, "features", "regression", "Top360_Meta_Fe_Test.csv")
    reg_file = args.proba_file_reg or (default_reg_prob if os.path.exists(default_reg_prob) else None)
    if reg_file and os.path.exists(reg_file):
        df_reg = pd.read_csv(reg_file)
        feat_cols = df_reg.columns[1:-2]
        for _, row in df_reg.iterrows():
            reg_prob_dict[row['Sequence']] = torch.tensor(row[feat_cols].values.astype(np.float32))
            
    # 3. Load Models
    cls_models, cls_cfg = None, None
    if args.task in ["classification", "both"]:
        print("Loading classification 5-fold ensemble...")
        cls_models, cls_cfg = load_classification_models(device)
        
    reg_models, reg_cfg = None, None
    if args.task in ["regression", "both"]:
        print("Loading regression 5-fold ensemble...")
        reg_models, reg_cfg = load_regression_models(device)
        
    # 4. Predict
    results = []
    print("\nRunning inference...")
    with torch.no_grad():
        for rec in records:
            name = rec.id
            seq = str(rec.seq).upper()
            x_seqvec = get_seqvec_embedding(name, seq, device)
            len_tokens = x_seqvec.shape[0]
            max_len = 39
            x_padded = F.pad(x_seqvec, (0, 0, 0, max_len - len_tokens), value=0).unsqueeze(0)
            mask = torch.ones(max_len, dtype=torch.bool, device=device)
            mask[:len_tokens] = False
            mask = mask.unsqueeze(0)
            
            res_row = {
                "sequence_id": name,
                "sequence": seq,
                "length": len(seq)
            }
            
            # Classification inference
            if cls_models is not None:
                # Match name or clean name
                clean_id = name.split()[0]
                if "_" in clean_id:
                    cand = f"{clean_id.split('_')[0]}_{clean_id.split('_')[1]}"
                else:
                    cand = clean_id
                x_probs = cls_prob_dict.get(clean_id, cls_prob_dict.get(cand, torch.zeros(300))).unsqueeze(0).to(device)
                x_merci = torch.tensor([[merci_dict.get(clean_id, merci_dict.get(cand, 0.0))]], dtype=torch.float32, device=device)
                
                fold_probs = []
                for model in cls_models:
                    logits = model([x_padded], mask, x_probs, x_merci)
                    probs = F.softmax(logits, dim=-1)
                    fold_probs.append(probs[0, 1].item())
                prob_hemolytic = float(np.mean(fold_probs))
                call = "Hemolytic" if prob_hemolytic >= 0.5 else "Non-Hemolytic"
                res_row["predicted_class"] = call
                res_row["hemolytic_probability"] = round(prob_hemolytic, 4)
                
            # Regression inference
            if reg_models is not None:
                x_probs_reg = reg_prob_dict.get(seq, torch.zeros(360)).unsqueeze(0).to(device)
                x_merci_reg = torch.zeros((1, 0), dtype=torch.float32, device=device)
                
                fold_preds = []
                for model in reg_models:
                    pred = model([x_padded], [mask], x_probs_reg, x_merci_reg)
                    fold_preds.append(pred.item())
                pred_phc50 = float(np.mean(fold_preds))
                pred_hc50 = 10 ** (6.0 - pred_phc50)
                res_row["predicted_pHC50"] = round(pred_phc50, 4)
                res_row["predicted_HC50_uM"] = round(pred_hc50, 2)
                
            results.append(res_row)
            
    # 5. Output and Save
    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output, index=False)
    print(f"\nPredictions saved successfully to: {args.output}")
    
    # Print formatted terminal preview
    print("\n" + "=" * 80)
    print(f"{'Sequence ID':<25} | {'Class':<14} | {'Prob':<8} | {'pHC50':<8} | {'HC50 (uM)':<10}")
    print("-" * 80)
    for r in results:
        cid = r['sequence_id'][:24]
        c = r.get('predicted_class', '-')
        p = f"{r.get('hemolytic_probability', 0.0):.4f}" if 'hemolytic_probability' in r else '-'
        phc = f"{r.get('predicted_pHC50', 0.0):.2f}" if 'predicted_pHC50' in r else '-'
        hc = f"{r.get('predicted_HC50_uM', 0.0):.1f}" if 'predicted_HC50_uM' in r else '-'
        print(f"{cid:<25} | {c:<14} | {p:<8} | {phc:<8} | {hc:<10}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
