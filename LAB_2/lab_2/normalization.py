import numpy as np
import pandas as pd


def normalize_zscore(X, ref_stats=None):
    if ref_stats is None:
        mean = X.mean()
        std = X.std()
    else:
        mean, std = ref_stats

    if isinstance(std, pd.Series) or isinstance(std, pd.DataFrame):
        std = std.replace(0, 1)
    else:
        std = np.where(std == 0, 1, std)

    X_norm = (X - mean) / std
    return X_norm, (mean, std)


def normalize_minmax(X, ref_stats=None):
    """
    Min-Max normalization в диапазон [0, 1].
    """
    if ref_stats is None:
        X_min = X.min()
        X_max = X.max()
    else:
        X_min, X_max = ref_stats

    range_ = X_max - X_min
    if isinstance(range_, pd.Series) or isinstance(range_, pd.DataFrame):
        range_ = range_.replace(0, 1)
    else:
        range_ = np.where(range_ == 0, 1, range_)

    X_norm = (X - X_min) / range_
    return X_norm, (X_min, X_max)


def impute_median(X: pd.DataFrame) -> pd.DataFrame:
    medians = X.median()
    X_filled = X.fillna(medians)
    return X_filled