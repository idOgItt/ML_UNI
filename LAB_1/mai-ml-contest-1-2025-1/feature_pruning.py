import numpy as np
from cross_validation import cross_val_score_manual, kfold_split
from linear_regression import LinearRegressionManual
from metrics import mean_squared_error_manual

import matplotlib.pyplot as plt
import seaborn as sns


def auto_prune_correlated_features(df, target_column, feature_pairs, all_features,
                                   method="analytic", threshold=0.95, verbose=True):
    """
    Автоматический отбор признаков на основе сравнения качества модели по каждой паре
    коррелирующих признаков. Оставляет только один признак из каждой пары — тот, при
    котором модель показывает лучший результат по метрике (по умолчанию MSE).

    Parameters:
    - df: DataFrame
    - target_column: имя колонки с целевой переменной
    - feature_pairs: список пар коррелирующих признаков [(f1, f2), (f3, f4), ...]
    - all_features: исходный список всех числовых признаков
    - method: метод обучения модели ("analytic", "gradient", "sgd")
    - threshold: корреляционный порог (не используется в функции, но полезен при отборе пар)
    - verbose: выводить промежуточные сравнения

    Returns:
    - selected_features: список отобранных признаков (без выброшенных)
    """
    y = df[target_column].values
    to_drop = set()

    for f1, f2 in feature_pairs:
        if f1 in to_drop or f2 in to_drop:
            continue  # уже обработаны ранее

        base_features_A = [f for f in all_features if f != f2 and f not in to_drop]
        base_features_B = [f for f in all_features if f != f1 and f not in to_drop]

        X_A = df[base_features_A].fillna(df[base_features_A].median()).values
        X_B = df[base_features_B].fillna(df[base_features_B].median()).values

        model = lambda: LinearRegressionManual(method=method)
        cv = kfold_split(X_A, n_splits=5, shuffle=True, random_state=42)

        mse_A = np.mean(cross_val_score_manual(model(), X_A, y, cv=cv, scoring=mean_squared_error_manual))
        mse_B = np.mean(cross_val_score_manual(model(), X_B, y, cv=cv, scoring=mean_squared_error_manual))

        if verbose:
            print(f"{f1} vs {f2} → MSE_A={mse_A:.4f}, MSE_B={mse_B:.4f} → drop {'f2' if mse_A <= mse_B else 'f1'}")

        if mse_A <= mse_B:
            to_drop.add(f2)
        else:
            to_drop.add(f1)

    selected = [f for f in all_features if f not in to_drop]
    return selected
