"""Five-fold Transformer ensembles: checkpoint loading and prediction."""
import glob
import os

import torch
import torch.nn.functional as F
import yaml

from .architecture import Hemopi_arch, Hemopi_Regression_arch


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_ensemble(task, model_dir, device, legacy_padding_mask):
    """Load the fold checkpoints (*.ckpt) and config.yaml in `model_dir`."""
    config_path = os.path.join(model_dir, 'config.yaml')
    checkpoints = sorted(glob.glob(os.path.join(model_dir, '*.ckpt')))
    if not os.path.exists(config_path) or not checkpoints:
        raise FileNotFoundError(f"No {task} checkpoints (*.ckpt) and config.yaml found in '{model_dir}'")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    arch = Hemopi_arch if task == 'classification' else Hemopi_Regression_arch
    models = []
    for path in checkpoints:
        state = {k[len('model.'):]: v for k, v in _load_checkpoint(path, device)['state_dict'].items()
                 if k.startswith('model.')}
        model = arch(**config['model'], legacy_padding_mask=legacy_padding_mask)
        model.load_state_dict(state)
        models.append(model.eval().to(device))
    return models, config


def _pad(embeddings, max_len, device):
    """Zero-pad per-residue embeddings to `max_len`; the mask is True at padded positions."""
    x = torch.zeros(len(embeddings), max_len, embeddings[0].shape[-1])
    mask = torch.ones(len(embeddings), max_len, dtype=torch.bool)
    for b, emb in enumerate(embeddings):
        x[b, :emb.shape[0]] = emb
        mask[b, :emb.shape[0]] = False
    return x.to(device), mask.to(device)


@torch.no_grad()
def predict_classification(models, config, ids, seqvec, base_probs, merci, device, batch_size=64):
    """Ensemble-mean P(hemolytic) per sequence."""
    out = {}
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        x, mask = _pad([seqvec[i] for i in batch], config['dataset']['max_length'], device)
        probs = torch.stack([base_probs[i] for i in batch]).to(device)
        scores = torch.tensor([[merci[i]] for i in batch], dtype=torch.float32, device=device)
        p = torch.stack([F.softmax(m([x], [mask], probs, scores), dim=-1)[:, 1] for m in models])
        out.update(zip(batch, p.mean(0).tolist()))
    return out


@torch.no_grad()
def predict_regression(models, config, ids, seqvec, base_preds, device, batch_size=64):
    """Ensemble-mean predicted pHC50 per sequence."""
    out = {}
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        x, mask = _pad([seqvec[i] for i in batch], config['dataset']['max_length'], device)
        preds = torch.stack([base_preds[i] for i in batch]).to(device)
        no_merci = torch.zeros((len(batch), 0), dtype=torch.float32, device=device)
        p = torch.stack([m([x], [mask], preds, no_merci).reshape(-1) for m in models])
        out.update(zip(batch, p.mean(0).tolist()))
    return out
