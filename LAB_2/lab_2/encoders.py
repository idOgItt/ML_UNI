import numpy as np
import pandas as pd

def label_encode(X):
    """
    Label Encoding: каждой категории — целочисленное значение.
    """
    X_series = pd.Series(X, name=getattr(X, 'name', None))
    categories = X_series.unique().tolist()
    mapping = {cat: idx for idx, cat in enumerate(categories)}
    return X_series.map(mapping)

def one_hot_encode(X, drop_first=False):
    """
    One-Hot Encoding: бинарные столбцы под каждую категорию.
    Если drop_first=True, удаляем один столбец для избежания мультиколлинеарности.
    """
    X_series = pd.Series(X, name=getattr(X, 'name', 'x'))
    return pd.get_dummies(X_series, drop_first=drop_first)

def ordinal_encode(X, mapping=None):
    """
    Ordinal Encoding: категории упорядочены по заранее заданному mapping.
    mapping — dict {category: integer}.
    """
    X_series = pd.Series(X, name=getattr(X, 'name', None))
    if mapping is None:
        cats = sorted(X_series.unique())
        mapping = {cat : idx for idx, cat in enumerate(cats)}
    return X_series.map(mapping)

def target_encode(X, y, smoothing=1.0):
    """
    Target Encoding: среднее y для каждой категории с сглаживанием.
    smoothing — коэффициент регуляризации.
    """
    X_series = pd.Series(X, name=getattr(X, 'name', None))
    y_series = pd.Series(y).astype(float)

    stats = y_series.groupby(X_series).agg(['mean', 'count'])
    global_mean = y_series.mean()

    smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) \
             / (stats['count'] + smoothing)
    return X_series.map(smooth)