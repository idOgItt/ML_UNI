import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

def kfold_split(X, n_splits=5, shuffle=False, random_state=None):
    """
    Генерация индексов для K-Fold.
    Возвращает list из k кортежей (train_idx, test_idx).
    """
    X_arr = np.asarray(X)
    n_samples = len(X_arr)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    base_size = n_samples // n_splits
    extras = n_samples % n_splits
    fold_sizes = [base_size + (1 if i < extras else 0) for i in range(n_splits)]

    splits = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append((train_idx, test_idx))
        current = stop

    return splits


def loo_split(X):
    """
    Leave-One-Out: каждый тестовый набор — один пример.
    """
    X_arr = np.asarray(X)
    n_samples = len(X_arr)
    indicies = np.arange(n_samples)

    splits = []
    for i in indicies:
        test_idx = np.array([i])
        train_idx = np.delete(indicies, i)
        splits.append((train_idx, test_idx))

    return splits


def cross_val_score_manual(model, X, y, cv, scoring):
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    scores = []

    for train_idx, test_idx in cv:
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores.append(scoring(y_test, y_pred))

    return scores

def train_test_split_stratified(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
):
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, val_idx = next(sss.split(X, y))
    return X.iloc[train_idx], X.iloc[val_idx], y[train_idx], y[val_idx]