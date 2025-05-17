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
