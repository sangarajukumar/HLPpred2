"""End-to-end inference: FASTA -> features -> base models -> Transformer ensembles."""
import logging
import os
import re

import pandas as pd
import yaml
from Bio import SeqIO

from .base_models import classification_base_probs, read_model_list, regression_base_preds
from .ensemble import load_ensemble, predict_classification, predict_regression
from .features import (check_environment, extract_merci_scores, extract_raw_features,
                       extract_seqvec_residue, required_feature_types)

log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, 'config')
MOTIF_DIR = os.path.join(ROOT, 'motif')
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
MAX_LENGTH = 39


def default_paths():
    return {
        'models_dir': os.path.join(ROOT, 'models'),
        'esm_dir': os.path.join(ROOT, 'tools', 'esm'),
        'embedder_weights_dir': None,
    }


def read_fasta(path):
    """Read and validate a FASTA file; returns (ids, sequences)."""
    names, seqs = [], []
    for rec in SeqIO.parse(path, 'fasta'):
        names.append(rec.id)
        seqs.append(re.sub(r'\s', '', str(rec.seq)).upper())
    if not names:
        raise ValueError(f'No sequences found in {path}')
    errors = []
    for name, seq in zip(names, seqs):
        invalid = sorted(set(seq) - STANDARD_AA)
        if invalid:
            errors.append(f'{name}: non-standard residue(s) {"".join(invalid)}')
        if not 1 <= len(seq) <= MAX_LENGTH:
            errors.append(f'{name}: length {len(seq)} (accepted: 1-{MAX_LENGTH})')
    if errors:
        raise ValueError('Invalid input sequences:\n  ' + '\n  '.join(errors))
    return names, seqs


def run(names, seqs, tasks, device='cpu', paths=None):
    """Predict for `tasks` (subset of {'classification', 'regression'}).

    Returns a DataFrame with one row per sequence and {task: base models that could not be scored}.
    """
    p = {**default_paths(), **{k: v for k, v in (paths or {}).items() if v}}
    with open(os.path.join(CONFIG_DIR, 'models.yaml')) as f:
        model_cfg = yaml.safe_load(f)
    check_environment(p['esm_dir'])

    ids = [f'seq{k}' for k in range(len(seqs))]
    base_names = {t: read_model_list(os.path.join(CONFIG_DIR, f'{t}_base_models.txt')) for t in tasks}

    merci = None
    if 'classification' in tasks:
        log.info('Scanning MERCI motifs')
        merci = extract_merci_scores(ids, seqs, os.path.join(MOTIF_DIR, 'MERCI_motif_locator.pl'), MOTIF_DIR)

    seqvec = {}
    needed = sorted({f for t in tasks for f in required_feature_types(base_names[t])})
    features = extract_raw_features(ids, seqs, needed, p['esm_dir'], p['embedder_weights_dir'], device, seqvec)

    meta, failed = {}, {}
    for task, scorer in (('classification', classification_base_probs), ('regression', regression_base_preds)):
        if task in tasks:
            log.info('Scoring %d %s base models', len(base_names[task]), task)
            meta[task], failed[task] = scorer(features, ids, base_names[task],
                                              os.path.join(p['models_dir'], model_cfg[task]['ml_dir']))
    del features
    seqvec = extract_seqvec_residue(ids, seqs, device, done=seqvec)

    result = pd.DataFrame({'sequence_id': names, 'sequence': seqs, 'length': [len(s) for s in seqs]})
    for task in ('classification', 'regression'):
        if task not in tasks:
            continue
        log.info('Running the %s ensemble', task)
        models, cfg = load_ensemble(task, os.path.join(p['models_dir'], model_cfg[task]['dl_dir']),
                                    device, model_cfg[task]['legacy_padding_mask'])
        if task == 'classification':
            prob = predict_classification(models, cfg, ids, seqvec, meta[task], merci, device)
            result['hemolytic_probability'] = [round(prob[i], 4) for i in ids]
            result['predicted_class'] = ['Hemolytic' if prob[i] >= 0.5 else 'Non-hemolytic' for i in ids]
            result['merci_score'] = [merci[i] for i in ids]
        else:
            phc50 = predict_regression(models, cfg, ids, seqvec, meta[task], device)
            result['predicted_pHC50'] = [round(phc50[i], 4) for i in ids]
            result['predicted_HC50_uM'] = [round(10 ** (6.0 - phc50[i]), 2) for i in ids]
    return result, failed
