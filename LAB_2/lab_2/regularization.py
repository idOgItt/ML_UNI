import numpy as np


def lasso_regression(
        X, y, alpha=1.0, alpha_lr=0.01, n_iters=1000, tol=1e-6
):
    """
    Lasso (L1) regression via gradient descent.
    Loss = MSE + alpha * ||w||_1
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n, d = X_arr.shape

    w = np.zeros(d)
    weights_history = [w.copy()]
    loss_history = [np.mean((X_arr.dot(w) - y_arr) ** 2) + alpha * np.sum(np.abs(w))]

    thr = alpha_lr * alpha
    for _ in range(n_iters):
        # 1) градиент MSE
        grad = (2.0 / n) * X_arr.T.dot(X_arr.dot(w) - y_arr)
        w_temp = w - alpha_lr * grad
        # 2) soft-threshold
        w = np.sign(w_temp) * np.maximum(np.abs(w_temp) - thr, 0.0)

        loss = np.mean((X_arr.dot(w) - y_arr) ** 2) + alpha * np.sum(np.abs(w))
        weights_history.append(w.copy())
        loss_history.append(loss)
        if abs(loss_history[-1] - loss_history[-2]) < tol:
            break

    return weights_history, loss_history


def ridge_regression(
        X, y, alpha=1.0, alpha_lr=0.01, n_iters=1000, tol=1e-6
):
    """
       Ridge (L2) regression via gradient descent.
       Loss = MSE + alpha * ||w||_2^2
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_samples, n_features = X_arr.shape

    w = np.zeros(n_features)
    weights_history = [w.copy()]
    loss_history = [np.mean((X_arr.dot(w) - y_arr) ** 2) + alpha * np.sum(w ** 2)]

    for _ in range(n_iters):
        grad_mse = (2.0 / n_samples) * X_arr.T.dot(X_arr.dot(w) - y_arr)
        grad_l2 = 2 * alpha * w
        grad = grad_mse + grad_l2

        w = w - alpha_lr * grad
        loss = np.mean((X_arr.dot(w) - y_arr) ** 2) + alpha * np.sum(w ** 2)

        weights_history.append(w.copy())
        loss_history.append(loss)
        if abs(loss_history[-1] - loss_history[-2]) < tol:
            break

    return weights_history, loss_history


def elastic_net_regression(
        X, y, alpha1=1.0, alpha2=1.0, alpha_lr=0.01, n_iters=1000, tol=1e-6
):
    """
   Elastic Net via gradient descent.
   Loss = MSE + alpha1 * ||w||_1 + alpha2 * ||w||_2^2
   """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_samples, n_features = X_arr.shape

    w = np.zeros(n_features)
    weights_history = [w.copy()]
    loss_history = [np.mean((X_arr.dot(w) - y_arr) ** 2)
                + alpha1 * np.sum(np.abs(w))
                + alpha2 * np.sum(w ** 2)]
    for _ in range(n_iters):
        grad_mse = (2.0 / n_samples) * X_arr.T.dot(X_arr.dot(w) - y_arr)
        grad_l1 = alpha1 * np.sign(w)
        grad_l2 = 2 * alpha2 * w
        grad = grad_mse + grad_l1 + grad_l2

        w = w - alpha_lr * grad
        loss = (np.mean((X_arr.dot(w) - y_arr) ** 2)
                + alpha1 * np.sum(np.abs(w))
                + alpha2 * np.sum(w ** 2))

        weights_history.append(w.copy())
        loss_history.append(loss)
        if abs(loss_history[-1] - loss_history[-2]) < tol:
            break

    return weights_history, loss_history


def lp_regression(
        X, y, alpha=1.0, p=1.0, alpha_lr=0.01, n_iters=1000, tol=1e-6
):
    """
    Lp-regularization via gradient descent.
    Loss = MSE + alpha * ||w||_p^p
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_samples, n_features = X_arr.shape

    w = np.zeros(n_features)
    weights_history = [w.copy()]
    loss_history = [np.mean((X_arr.dot(w) - y_arr) ** 2) + alpha * np.sum(np.abs(w) ** p)]

    for _ in range(n_iters):
        grad_mse = (2.0 / n_samples) * X_arr.T.dot(X_arr.dot(w) - y_arr)
        grad_lp = alpha * p * np.sign(w) * np.abs(w)**(p-1)
        grad = grad_mse + grad_lp

        w = w - alpha_lr * grad
        loss = np.mean((X_arr.dot(w) - y_arr)**2) + alpha * np.sum(np.abs(w)**p)

        weights_history.append(w.copy())
        loss_history.append(loss)
        if abs(loss_history[-1] - loss_history[-2]) < tol:
            break

    return weights_history, loss_history