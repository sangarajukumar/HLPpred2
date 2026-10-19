from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import math
import numpy as np

def calculate_metrics(probs, targets, threshold=0.5):
    predictions = np.where(probs > threshold, 1, 0)
    tp = ((predictions == 1) & (targets == 1)).sum()
    tn = ((predictions == 0) & (targets == 0)).sum()
    fp = ((predictions == 1) & (targets == 0)).sum()
    fn = ((predictions == 0) & (targets == 1)).sum()

    sn = sensitivity(tp, fn)
    sp = specificity(tn, fp)
    fdr = false_discovery_rate(fp, tp)
    acc = accuracy(tp, tn, fp, fn)
    bacc = balanced_accuracy(sn, sp)
    f1 = f1_score(tp, fp, fn)
    mcc = matthews_corr_coef(tp, tn, fp, fn)
    auc_roc = calculate_auc_roc_auc_prc(targets, probs, predictions)

    return {
        'sn': float(sn) if np.isfinite(sn) else 0.0,
        'sp': float(sp) if np.isfinite(sp) else 0.0,
        'fdr': float(fdr) if np.isfinite(fdr) else 0.0,
        'acc': float(acc) if np.isfinite(acc) else 0.0,
        'bacc': float(bacc) if np.isfinite(bacc) else 0.0,
        'f1': float(f1) if np.isfinite(f1) else 0.0,
        'mcc': float(mcc) if np.isfinite(mcc) else 0.0,
        'auc_roc': float(auc_roc) if np.isfinite(auc_roc) else 0.0,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

def calculate_auc_roc_auc_prc(y_true, probs, y_pred):
    try:
        auc_roc = roc_auc_score(y_true, probs)
        return auc_roc
    except:
        return 0.0

def balanced_accuracy(sn, sp):
    if not np.isfinite(sn) or not np.isfinite(sp):
        return 0.0
    return (sn + sp) / 2

def sensitivity(tp, fn):
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def specificity(tn, fp):
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def false_discovery_rate(fp, tp):
    if (fp + tp) == 0:
        return 0.0
    return fp / (fp + tp)

def accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total

def f1_score(tp, fp, fn):
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if (prec + rec) == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)

def matthews_corr_coef(tp, tn, fp, fn):
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    numerator = (tp * tn) - (fp * fn)
    return numerator / denominator
