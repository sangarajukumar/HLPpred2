"""Base (meta-feature) models.

Classification: 300 classifiers, each giving P(hemolytic).
Regression:     360 regressors, each giving a predicted pHC50.
Model names are '<ALGORITHM>_<FEATURE>'; their order is fixed by config/*_base_models.txt.
"""
import logging
import os
import warnings

import joblib
import numpy as np
import torch

log = logging.getLogger(__name__)


def read_model_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _patch_randomstate_unpickling():
    """Let numpy unpickle RandomState objects saved by a newer numpy (not used for prediction)."""
    def ctor(*args):
        return np.random.RandomState()

    targets = [np.random]
    for name in ('numpy.random.mtrand', 'numpy.random._pickle'):
        try:
            targets.append(__import__(name, fromlist=['_']))
        except ImportError:
            pass
    for target in targets:
        target.__randomstate_ctor = ctor


def _load_classifier(model_dir, name):
    """<NAME>_model_<ALGORITHM>.pkl, with <NAME>_scaler_<ALGORITHM>.pkl for scaled models (ANN, LRT)."""
    algo = name.split('_', 1)[0]
    model = joblib.load(os.path.join(model_dir, f'{name}_model_{algo}.pkl'))
    scaler_path = os.path.join(model_dir, f'{name}_scaler_{algo}.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return lambda X: model.predict_proba(X if scaler is None else scaler.transform(X))[:, 1]


def _load_regressor(model_dir, name):
    """<ALGORITHM>/<FEATURE>_model.joblib holding {'model': ..., 'scaler': ...}."""
    algo, feat = name.split('_', 1)
    bundle = joblib.load(os.path.join(model_dir, algo, f'{feat}_model.joblib'))
    model, scaler = bundle['model'], bundle.get('scaler')
    return lambda X: model.predict(X if scaler is None else scaler.transform(X))


def _score(features, ids, model_names, model_dir, loader):
    """Score every sequence with every base model.

    A model that cannot be scored contributes 0.0 and is reported in the returned list of failures.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Base model folder not found: '{model_dir}'")
    _patch_randomstate_unpickling()
    warnings.filterwarnings('ignore')
    out = {i: [] for i in ids}
    failed = []
    for name in model_names:
        feat = name.split('_', 1)[1]
        valid = [i for i in ids if i in features.get(feat, {})]
        values = {}
        try:
            if not valid:
                raise ValueError(f'feature {feat} unavailable')
            predict = loader(model_dir, name)
            values = dict(zip(valid, predict(np.array([features[feat][i] for i in valid]))))
            if len(valid) < len(ids):
                failed.append(f'{name}: {feat} missing for {len(ids) - len(valid)} sequence(s)')
        except Exception as e:
            failed.append(f'{name}: {e}')
        for i in ids:
            out[i].append(float(values.get(i, 0.0)))
    return {i: torch.tensor(v, dtype=torch.float32) for i, v in out.items()}, failed


def classification_base_probs(features, ids, model_names, model_dir):
    """Return ({id: tensor[300]}, failures)."""
    return _score(features, ids, model_names, model_dir, _load_classifier)


def regression_base_preds(features, ids, model_names, model_dir):
    """Return ({id: tensor[360]}, failures)."""
    return _score(features, ids, model_names, model_dir, _load_regressor)
