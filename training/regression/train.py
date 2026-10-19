from argparse import ArgumentParser
import pickle

import pandas as pd
from src.utils import get_config, extract_headers_phc50
from src.lightning_module import Hemopi_Regression_Module
from src.dataset import Hemopi_Regression_Dataset
import torch
from torch.utils.data import DataLoader, Subset
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os
import copy
import yaml
import numpy as np
from tqdm import tqdm
from lightning.pytorch.loggers import TensorBoardLogger
import glob
from Bio import SeqIO
import warnings
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import Callback
from typing import Any, Dict
warnings.filterwarnings("ignore")

def process_prob_features(prob_feature_file, fasta_file):
    """
    Load prob features from CSV, keyed by FASTA header ID.
    CSV and FASTA are in the same order, so we match by position.
    """
    from Bio import SeqIO
    header_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header_id = record.description.split('_HC50_')[0]
        header_ids.append(header_id)
    
    df = pd.read_csv(prob_feature_file)
    
    assert len(header_ids) == len(df), \
        f"Mismatch: FASTA has {len(header_ids)} entries but CSV has {len(df)} rows"
    
    feature_cols = df.columns[1:-2]
    
    prob_features = {}
    for idx, header_id in enumerate(header_ids):
        prob_features[header_id] = torch.tensor(
            df.iloc[idx][feature_cols].values.astype(np.float32)
        )
    
    return prob_features

def process_merci_features(merci_feature_file):
    """Load MERCI scores (continuous: -1.0 to +1.0)"""
    if not merci_feature_file or not os.path.exists(merci_feature_file):
        return None
    
    merci_features = {}
    with open(merci_feature_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                header = parts[0]
                merci_score = float(parts[1])
                merci_features[header] = torch.tensor([merci_score], dtype=torch.float)
    
    print(f"  Loaded MERCI features for {len(merci_features)} sequences")
    return merci_features

def test_kfold_models_regression(model, ckpt_dir, test_dataloader, device):
    """
    Test ensemble of K-fold models on test set (regression version)
    """
    MODELS_LIST = []
    
    for ckpt_file in os.listdir(ckpt_dir):
        if not ckpt_file.endswith('.ckpt'):
            continue
        
        pt_file_path = os.path.join(ckpt_dir, ckpt_file)
        state_dict = torch.load(pt_file_path)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        MODELS_LIST.append(copy.deepcopy(model))
    
    print(f"Loaded {len(MODELS_LIST)} models for ensemble prediction")
    
    ALL_Y_TRUE = []
    ALL_Y_PRED = []
    
    for batch in tqdm(test_dataloader, desc='Testing', total=len(test_dataloader)):
        X, X_probs, X_merci, masks_X, phc50_true = batch
        X = [x.to(device) for x in X]
        X_probs = X_probs.to(device)
        X_merci = X_merci.to(device)
        masks_X = [mask.to(device) for mask in masks_X]
        phc50_true = phc50_true.to(device)
        
        batch_predictions = []
        for model in MODELS_LIST:
            with torch.no_grad():
                phc50_pred = model(X, masks_X, X_probs, X_merci)
            batch_predictions.append(phc50_pred)
        
        avg_pred = torch.stack(batch_predictions).mean(dim=0)
        
        ALL_Y_PRED.append(avg_pred)
        ALL_Y_TRUE.append(phc50_true)
    
    ALL_Y_TRUE = torch.cat(ALL_Y_TRUE).detach().cpu().numpy()
    ALL_Y_PRED = torch.cat(ALL_Y_PRED).detach().cpu().numpy()
    
    mse = mean_squared_error(ALL_Y_TRUE, ALL_Y_PRED)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ALL_Y_TRUE, ALL_Y_PRED)
    r2 = r2_score(ALL_Y_TRUE, ALL_Y_PRED)
    
    try:
        pcc, pval = pearsonr(ALL_Y_TRUE, ALL_Y_PRED)
    except:
        pcc, pval = 0.0, 1.0
    
    with open(os.path.join(ckpt_dir, 'test_ensemble_results.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("ENSEMBLE TEST SET RESULTS (Average of all folds)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of models in ensemble: {len(MODELS_LIST)}\n")
        f.write(f"Test samples: {len(ALL_Y_TRUE)}\n\n")
        f.write("Regression Metrics:\n")
        f.write(f"  R² (R-squared):         {r2:.5f}\n")
        f.write(f"  RMSE:                   {rmse:.5f}\n")
        f.write(f"  MAE:                    {mae:.5f}\n")
        f.write(f"  Pearson Correlation:    {pcc:.5f}\n")
        f.write(f"  MSE:                    {mse:.5f}\n")
        f.write(f"  P-value:                {pval:.5e}\n")
        f.write("\n" + "="*60 + "\n")
        f.write(f"Target to beat (HemoPI-2): R² = 0.543\n")
        f.write(f"Your result: R² = {r2:.5f}\n")
        if r2 > 0.543:
            improvement = ((r2 - 0.543) / 0.543) * 100
            f.write(f"✓ Improvement: +{improvement:.1f}%\n")
        f.write("="*60 + "\n")
    
    print(f"\n{'='*60}")
    print("ENSEMBLE TEST RESULTS:")
    print(f"{'='*60}")
    print(f"R² = {r2:.5f}")
    print(f"RMSE = {rmse:.5f}")
    print(f"MAE = {mae:.5f}")
    print(f"PCC = {pcc:.5f}")
    print(f"{'='*60}\n")

def evaluate_model_regression(model, ckpt_dir, fold, dataloader, device):
    """
    Evaluate single fold model on validation set (regression version)
    """
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f'fold_{fold}' in f and f.endswith('.ckpt')]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint file found for fold {fold} in directory {ckpt_dir}")
    
    pt_file_path = os.path.join(ckpt_dir, ckpt_files[0])
    state_dict = torch.load(pt_file_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    ALL_Y_TRUE = []
    ALL_Y_PRED = []
    
    for batch in dataloader:
        X, X_probs, X_merci, masks_X, phc50_true = batch
        X = [x.to(device) for x in X]
        X_probs = X_probs.to(device)
        X_merci = X_merci.to(device)
        masks_X = [mask.to(device) for mask in masks_X]
        phc50_true = phc50_true.to(device)
        
        with torch.no_grad():
            phc50_pred = model(X, masks_X, X_probs, X_merci)
        
        ALL_Y_PRED.append(phc50_pred)
        ALL_Y_TRUE.append(phc50_true)
    
    ALL_Y_TRUE = torch.cat(ALL_Y_TRUE).detach().cpu().numpy()
    ALL_Y_PRED = torch.cat(ALL_Y_PRED).detach().cpu().numpy()
    
    mse = mean_squared_error(ALL_Y_TRUE, ALL_Y_PRED)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ALL_Y_TRUE, ALL_Y_PRED)
    r2 = r2_score(ALL_Y_TRUE, ALL_Y_PRED)
    
    try:
        pcc, _ = pearsonr(ALL_Y_TRUE, ALL_Y_PRED)
    except:
        pcc = 0.0
    
    with open(os.path.join(ckpt_dir, f'fold_{fold}_validation_results.txt'), 'w') as f:
        f.write(f"Fold {fold} Validation Results\n")
        f.write("="*40 + "\n")
        f.write(f"R²:   {r2:.5f}\n")
        f.write(f"RMSE: {rmse:.5f}\n")
        f.write(f"MAE:  {mae:.5f}\n")
        f.write(f"PCC:  {pcc:.5f}\n")
        f.write(f"MSE:  {mse:.5f}\n")

def main(config):
    pl.seed_everything(config.seed)
    device = torch.device(config.trainer.device)
    
    test_file_path = config.dataset.test_fasta_path
    train_file_path = config.dataset.train_fasta_path
    
    header_train, phc50_train = extract_headers_phc50(train_file_path)
    header_test, phc50_test = extract_headers_phc50(test_file_path)
    
    phc50_train = np.array(phc50_train)
    phc50_test = np.array(phc50_test)
    
    print(f"{'='*60}")
    print("REGRESSION TRAINING")
    print(f"{'='*60}")
    print(f"Training samples: {len(header_train)}")
    print(f"  pHC50 range: {phc50_train.min():.2f} - {phc50_train.max():.2f}")
    print(f"  pHC50 mean: {phc50_train.mean():.2f}")
    print(f"Test samples: {len(header_test)}")
    print(f"  pHC50 range: {phc50_test.min():.2f} - {phc50_test.max():.2f}")
    print(f"{'='*60}\n")
    
    train_prob_features = process_prob_features(config.dataset.probs_file_path_train, train_file_path)
    test_prob_features  = process_prob_features(config.dataset.probs_file_path_test,  test_file_path)
    
    print("\nLoading MERCI features...")
    if hasattr(config.dataset, 'merci_file_path_train') and config.dataset.merci_file_path_train:
        train_merci_features = process_merci_features(config.dataset.merci_file_path_train)
        test_merci_features  = process_merci_features(config.dataset.merci_file_path_test)
    else:
        print("  Warning: No MERCI file paths in config, using None")
        train_merci_features = None
        test_merci_features  = None
    
    test_dataset = Hemopi_Regression_Dataset(
        config.dataset, 
        header_test, 
        phc50_test,
        test_prob_features,
        test_merci_features, 
        mode='test'
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=6, 
        shuffle=False, 
        num_workers=config.trainer.num_workers, 
        pin_memory=True
    )
    
    if config.save_config:
        if not os.path.exists(config.trainer.output_path):
            os.makedirs(config.trainer.output_path)
        with open(os.path.join(config.trainer.output_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    
    kf = KFold(n_splits=config.trainer.n_splits, shuffle=True, random_state=config.seed)
    
    for idx, (train_index, val_index) in enumerate(kf.split(header_train)):
        print(f"\n{'='*60}")
        print(f"FOLD {idx+1}/{config.trainer.n_splits}")
        print(f"{'='*60}")
        
        tb_logger = TensorBoardLogger(
            save_dir=config.trainer.output_path,
            name=f'fold_{idx+1}'
        )
        
        header_train_fold = [header_train[i] for i in train_index]
        phc50_train_fold = phc50_train[train_index]
        header_val_fold = [header_train[i] for i in val_index]
        phc50_val_fold = phc50_train[val_index]
        
        print(f"Train: {len(header_train_fold)} samples")
        print(f"Val:   {len(header_val_fold)} samples")
        
        train_dataset = Hemopi_Regression_Dataset(
            config.dataset, 
            header_train_fold, 
            phc50_train_fold, 
            train_prob_features,
            train_merci_features, 
            mode='train'
        )
        val_dataset = Hemopi_Regression_Dataset(
            config.dataset, 
            header_val_fold, 
            phc50_val_fold, 
            train_prob_features, 
            train_merci_features, 
            mode='val'
        )
        
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=config.trainer.batch_size, 
            shuffle=True, 
            num_workers=config.trainer.num_workers, 
            pin_memory=True
        )
        val_dataloader = DataLoader(
            dataset=val_dataset, 
            batch_size=config.trainer.batch_size, 
            shuffle=False, 
            num_workers=config.trainer.num_workers, 
            pin_memory=True
        )
        
        model_module = Hemopi_Regression_Module(config.model, config.trainer)
        
        trainer = pl.Trainer(
            accelerator=config.trainer.device,
            max_epochs=config.trainer.epochs,
            val_check_interval=0.5,
            gradient_clip_val=10.,
            callbacks=[
                ModelCheckpoint(
                    monitor='rmse',
                    dirpath=config.trainer.output_path,
                    filename=f'fold_{idx+1}_{{epoch}}_{{r2:.5f}}_{{rmse:.5f}}',
                    save_top_k=1,
                    mode='min'
                ),
                LearningRateMonitor(logging_interval='epoch')
            ],
            logger=tb_logger
        )
        
        trainer.fit(model_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        evaluate_model_regression(model_module, config.trainer.output_path, idx+1, val_dataloader, device)
    
    print(f"\n{'='*60}")
    print("TESTING ENSEMBLE ON INDEPENDENT TEST SET")
    print(f"{'='*60}\n")
    test_kfold_models_regression(model_module, config.trainer.output_path, test_dataloader, device)
    
    print("\n✓ Training complete!")
    print(f"Results saved to: {config.trainer.output_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/regression_config.yaml')
    parser.add_argument('--save_config', action='store_true')
    args = parser.parse_args()
    config = get_config(args.config_path)
    
    config.save_config = args.save_config
    main(config)
