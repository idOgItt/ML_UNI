import numpy as np

def pca_manual(X, n_components):

    Xc = np.asarray(X, dtype=float) - np.mean(X, axis=0)
    C = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1][:n_components]
    components = eigvecs[:, idx]
    explained_variance = eigvals[idx]
    X_pca = Xc.dot(components)
    return X_pca, components, explained_variance


def select_n_components(X, variance_threshold=0.95):

    Xc = np.asarray(X, dtype=float) - np.mean(X, axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals_sorted = np.sort(eigvals)[::-1]
    explained_ratio = eigvals_sorted / eigvals_sorted.sum()
    cumulative_ratio = np.cumsum(explained_ratio)
    n_components = int(np.searchsorted(cumulative_ratio, variance_threshold) + 1)
    return n_components, explained_ratio, cumulative_ratio