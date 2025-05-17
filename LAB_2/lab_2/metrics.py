import numpy as np

def mean_squared_error_manual(y_true, y_pred):
    """
        Формула:
            MSE = (1/n) * Σi (y_true_i - y_pred_i)^2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error_manual(y_true, y_pred):
    return np.sqrt(mean_squared_error_manual(y_true, y_pred))

def mean_absolute_error_manual(y_true, y_pred):
    """
        MAE: Mean Absolute Error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def median_absolute_error_manual(y_true, y_pred):
    """
    MedAE: Median Absolute Error.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.median(np.abs(y_true - y_pred))

def r2_score_manual(y_true, y_pred):
    """R²: коэффициент детерминации.
    Формула:
        R² = 1 - (Σi (y_true_i - y_pred_i)^2) / (Σi (y_true_i - ȳ)^2)

    Чем ближе к 1 тем лучше модель предсказывает
    Чем ближе к 0 юзлесс модель
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

def mean_absolute_percentage_error_manual(y_true, y_pred):
    """MAPE: средняя абсолютная процентная ошибка.
    Недостаточк на малых y_tue улетит в космос
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        raise AttributeError
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return np.mean(ape) * 100

def symmetric_mean_absolute_percentage_error_manual(y_true, y_pred):
    """sMAPE: симметричная MAPE.
    Устойчивая штука в отличии от MAPE
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return np.nan
    smape_values = np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return np.mean(smape_values) * 100

def accuracy_manual(y_true, y_pred):
    """
        Accuracy = (TP + TN) / (P + N)
        """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def precision_manual(y_true, y_pred, positive=1):
    """
        Precision = TP / (TP + FP)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    fp = np.sum((y_pred == positive) & (y_true != positive))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_manual(y_true, y_pred, positive=1):
    """
    Recall = TP / (TP + FN)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    fn = np.sum((y_pred != positive) & (y_true == positive))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score_manual(y_true, y_pred, positive=1):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    p = precision_manual(y_true, y_pred, positive)
    r = recall_manual(y_true, y_pred, positive)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def roc_auc_score_manual(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    desc = np.argsort(-y_score)
    y_true = y_true[desc]
    y_score = y_score[desc]
    # begin point and probabilities
    thresholds = np.concatenate(([np.inf], y_score))
    tprs, fprs = [], []
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    for thr in thresholds:
        y_pred = y_score >= thr
        tp = np.sum((y_true == 1) & y_pred)
        fp = np.sum((y_true == 0) & y_pred)
        tprs.append(tp / P if P else 0.0)
        fprs.append(fp / N if N else 0.0)
    return np.trapezoid(tprs, fprs)

def average_precision_score_manual(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    desc = np.argsort(-y_score)
    y_true = y_true[desc]
    tp_cum = np.cumsum(y_true == 1)
    fp_cum = np.cumsum(y_true == 0)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / tp_cum[-1] if tp_cum[-1] else np.zeros_like(tp_cum)

    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return np.trapezoid(precision, recall)