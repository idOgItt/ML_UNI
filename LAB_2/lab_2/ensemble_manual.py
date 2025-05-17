import numpy as np
import copy
from cross_validation import kfold_split
from sklearn.base import clone

def bootstrap_sample(X, y, n_samples=None, random_state=None):
    n = X.shape[0] if n_samples is None else n_samples
    rng = np.random.RandomState(random_state)
    idx = rng.randint(0, X.shape[0], size=n)
    return X[idx], y[idx]

class BaggingRegressorManual:
    def __init__(self, base_estimator, n_estimators=10,
                 max_samples=1.0, random_state=None):
        self.base = base_estimator
        self.n = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.models = []
        for i in range(self.n):
            rs = rng.randint(0, 2**32 - 1)
            size = (int(self.max_samples * X.shape[0])
                    if isinstance(self.max_samples, float)
                    else self.max_samples)
            Xb, yb = bootstrap_sample(X, y, n_samples=size,
                                      random_state=rs)
            m = copy.deepcopy(self.base)
            m.fit(Xb, yb)
            self.models.append(m)
        return self

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        # усреденее по строкам
        return np.mean(np.vstack(preds), axis=0)


class RandomForestRegressorManual(BaggingRegressorManual):

    def __init__(self, base_estimator, n_estimators=10,
                 max_samples=1.0, max_features='sqrt',
                 random_state=None):
        super().__init__(base_estimator, n_estimators,
                         max_samples, random_state)
        self.max_features = max_features
        self.models = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        self.models = []
        for i in range(self.n):
            rs = rng.randint(0, 2**32 - 1)
            size = (int(self.max_samples * X.shape[0])
                    if isinstance(self.max_samples, float)
                    else self.max_samples)
            Xb, yb = bootstrap_sample(X, y, n_samples=size,
                                      random_state=rs)
            # число фич
            if isinstance(self.max_features, float):
                k = int(self.max_features * n_features)
            elif self.max_features == 'sqrt':
                k = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                k = int(np.log2(n_features))
            else:
                k = int(self.max_features)
            feat_idx = rng.choice(n_features, size=k, replace=False)
            m = copy.deepcopy(self.base)
            m.fit(Xb[:, feat_idx], yb)
            self.models.append((m, feat_idx))
        return self

    def predict(self, X):
        preds = []
        for m, idx in self.models:
            preds.append(m.predict(X[:, idx]))
        return np.mean(np.vstack(preds), axis=0)


class AdaBoostRegressorManual:
    # Fix errors of others
    def __init__(self, base_estimator, n_estimators=50,
                 learning_rate=0.1):
        self.base = base_estimator
        self.n = n_estimators
        self.lr = learning_rate # your mistakes
        self.models = []

    def fit(self, X, y):
        self.models = []
        residual = y.copy().astype(float)
        for _ in range(self.n):
            m = copy.deepcopy(self.base)
            m.fit(X, residual)
            pred = m.predict(X)
            self.models.append(m)
            residual -= self.lr * pred # correction
        return self

    def predict(self, X):
        total = np.zeros(X.shape[0], dtype=float)
        for m in self.models:
            total += self.lr * m.predict(X) # correction with W
        return total


class GradientBoostingRegressorManual:
    def __init__(self, base_estimator, n_estimators=100,
                 learning_rate=0.1):
        self.base = base_estimator
        self.n = n_estimators
        self.lr = learning_rate
        self.models = []
        self.init_pred = None # base

    def fit(self, X, y):
        self.init_pred = np.full(y.shape, y.mean()) # base
        F = self.init_pred.copy()
        self.models = []
        for _ in range(self.n):
            residual = y - F
            m = copy.deepcopy(self.base)
            m.fit(X, residual)
            self.models.append(m)
            F += self.lr * m.predict(X)
        return self

    def predict(self, X):
        F = np.full(X.shape[0], self.init_pred[0])
        for m in self.models:
            F += self.lr * m.predict(X)
        return F


class StackingRegressorManual:
    def __init__(self, base_estimators, meta_estimator,
                 cv=5, shuffle=True, random_state=None):
        self.bases = base_estimators
        self.meta = meta_estimator
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        folds = kfold_split(X, n_splits=self.cv,
                            shuffle=self.shuffle,
                            random_state=self.random_state)
        n, k = X.shape[0], len(self.bases)
        meta_X = np.zeros((n, k), dtype=float)

        for i, base in enumerate(self.bases):
            for train_idx, test_idx in folds:
                m = copy.deepcopy(base)
                m.fit(X[train_idx], y[train_idx])
                meta_X[test_idx, i] = m.predict(X[test_idx])
            base.fit(X, y)

        self.meta.fit(meta_X, y)
        return self

    def predict(self, X):
        base_preds = np.column_stack([m.predict(X) for m in self.bases])
        return self.meta.predict(base_preds)


class BlendingRegressorManual:
    def __init__(self, base_estimators, meta_estimator,
                 holdout_fraction=0.2, random_state=None,
                 retrain_base=True):
        self.bases = base_estimators
        self.meta = meta_estimator
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state
        self.retrain_base = retrain_base
        self.train_idx = None
        self.holdout_idx = None

    def fit(self, X, y):
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)
        all_idx = np.arange(n)
        holdout_size = int(self.holdout_fraction * n)
        holdout_idx = rng.choice(all_idx, size=holdout_size, replace=False)
        train_idx = np.setdiff1d(all_idx, holdout_idx)
        self.train_idx, self.holdout_idx = train_idx, holdout_idx

        for m in self.bases:
            m.fit(X[train_idx], y[train_idx])

        meta_X = np.column_stack([
            m.predict(X[holdout_idx]) for m in self.bases
        ])
        y_hold = y[holdout_idx]

        self.meta.fit(meta_X, y_hold)

        if self.retrain_base:
            for m in self.bases:
                m.fit(X, y)

        return self

    def predict(self, X):
        base_preds = np.column_stack([m.predict(X) for m in self.bases])
        return self.meta.predict(base_preds)

class BaggingClassifierManual:
    def __init__(self, base_estimator, n_estimators=10,
                 max_samples=1.0, random_state=None):
        self.base = base_estimator
        self.n = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        sample_size = int(self.max_samples * n_samples) \
                        if isinstance(self.max_samples, float) else self.max_samples
        self.models = []
        for _ in range(self.n):
            rs = rng.randint(0, 2 ** 32 - 1)
            idx = rng.randint(0, n_samples, size=sample_size)
            Xb, yb = X[idx], y[idx]
            m = clone(self.base)
            m.fit(Xb, yb)
            self.models.append(m)
        return self

    def predict_proba(self, X):
        probs = np.stack([m.predict_proba(X)[:, 1] for m in self.models], axis=1)
        avg = np.mean(probs, axis=1)
        return np.vstack([1 - avg, avg]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class GradientBoostingClassifierManual:
    def __init__(self, base_estimator, n_estimators=100, learning_rate=0.1):
        self.base = base_estimator
        self.n = n_estimators
        self.lr = learning_rate
        self.models = []
        self.init_F = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        p0 = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.init_F = np.log(p0 / (1 - p0))

        F = np.full(y.shape, self.init_F, dtype=float)
        self.models = []

        for m in range(self.n):
            p = self._sigmoid(F)
            residual = y - p

            tree = clone(self.base)
            tree.fit(X, residual)
            self.models.append(tree)

            F += self.lr * tree.predict(X)

        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.init_F, dtype=float)
        for tree in self.models:
            F += self.lr * tree.predict(X)
        p = self._sigmoid(F)
        return np.vstack([1 - p, p]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)