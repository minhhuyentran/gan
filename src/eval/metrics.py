import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def anomaly_metrics(y_true: np.ndarray, anomaly_score: np.ndarray):
    """
    y_true: 1 = anomaly, 0 = normal
    anomaly_score: higher = more anomalous
    """
    auroc = roc_auc_score(y_true, anomaly_score)
    auprc = average_precision_score(y_true, anomaly_score)
    return {"auroc": float(auroc), "auprc": float(auprc)}

def threshold_at_fpr(y_true, normal_score, target_fpr=0.01):
    """
    y_true: 1 anomaly, 0 normal
    normal_score: higher=more normal
    Choose threshold on normal_score using normal class only to get target FPR.
    """
    normal_scores = normal_score[y_true == 0]
    # FPR = fraction normal predicted anomaly => normal_score < thr
    thr = np.quantile(normal_scores, target_fpr)
    return float(thr)

def fpr_tpr_at_threshold(y_true, normal_score, thr):
    pred_anom = (normal_score < thr).astype(int)
    # y_true: 1 anomaly
    tp = ((pred_anom == 1) & (y_true == 1)).sum()
    fp = ((pred_anom == 1) & (y_true == 0)).sum()
    tn = ((pred_anom == 0) & (y_true == 0)).sum()
    fn = ((pred_anom == 0) & (y_true == 1)).sum()
    fpr = fp / max(1, (fp + tn))
    tpr = tp / max(1, (tp + fn))
    return {"fpr": float(fpr), "tpr": float(tpr)}
