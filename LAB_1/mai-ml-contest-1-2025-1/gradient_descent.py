import numpy as np

def batch_gradient_descent(X, y, alpha=0.01, n_iters=1000, tol=1e-6):
    """
    Batch GD для линейной регрессии.
    Возвращает weights_history, loss_history.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_samples, n_features = X_arr.shape

    w = np.zeros(n_features)
    weights_history = [w.copy()]
    loss_history = [np.mean((X_arr.dot(w) - y_arr) ** 2)]

    for _ in range(n_iters):
        grad = (2.0 / n_samples) * X_arr.T.dot(X_arr.dot(w) - y_arr)
        w = w - alpha * grad
        current_loss = np.mean((X_arr.dot(w) - y_arr) ** 2)

        weights_history.append(w.copy())
        loss_history.append(current_loss)

        if abs(loss_history[-2] - loss_history[-1]) < tol:
            break

    return weights_history, loss_history


def stochastic_gradient_descent(X, y, alpha=0.01, n_iters=1000, tol=1e-6, shuffle=True):
    """
    SGD: обновление на каждом примере.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_samples, n_features = X_arr.shape

    w = np.zeros(n_features)
    weights_history = [w.copy()]
    loss_history = [np.mean((X_arr.dot(w) - y_arr) ** 2)]

    for epoch in range(n_iters):
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            xi = X_arr[i:i+1]
            yi = y_arr[i:i+1]
            grad = 2.0 * xi.T.dot(xi.dot(w) - yi)
            w = w - alpha * grad.flatten()
        current_loss = np.mean((X_arr.dot(w) - y_arr) ** 2)
        weights_history.append(w.copy())
        loss_history.append(current_loss)

        if abs(loss_history[-2] - loss_history[-1]) < tol:
            break

    return weights_history, loss_history


def mini_batch_gradient_descent(X, y, alpha=0.01, n_iters=1000,
                                batch_size=32, tol=1e-6, shuffle=True):
    """
    Mini-batch GD: блоки размера batch_size.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_samples, n_features = X_arr.shape

    w = np.zeros(n_features)
    weights_history = [w.copy()]
    loss_history = [np.mean((X_arr.dot(w) - y_arr) ** 2)]

    for epoch in range(n_iters):
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            xi = X_arr[batch_idx]
            yi = y_arr[batch_idx]
            grad = (2.0 / len(batch_idx)) * xi.T.dot(xi.dot(w) - yi)
            w = w - alpha * grad
        current_loss = np.mean((X_arr.dot(w) - y_arr) ** 2)
        weights_history.append(w.copy())
        loss_history.append(current_loss)

        if abs(loss_history[-2] - loss_history[-1]) < tol:
            break

    return weights_history, loss_history


def momentum_gradient_descent(X, y, alpha=0.01, beta=0.9,
                              n_iters=1000, tol=1e-6):
    """
    GD с импульсом (momentum).
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_samples, n_features = X_arr.shape

    w = np.zeros(n_features)
    v = np.zeros(n_features)
    weights_history = [w.copy()]
    loss_history = [np.mean((X_arr.dot(w) - y_arr) ** 2)]

    for _ in range(n_iters):
        grad = (2.0 / n_samples) * X_arr.T.dot(X_arr.dot(w) - y_arr)
        v = beta * v + grad
        w = w - alpha * v
        current_loss = np.mean((X_arr.dot(w) - y_arr) ** 2)

        weights_history.append(w.copy())
        loss_history.append(current_loss)

        if abs(loss_history[-2] - loss_history[-1]) < tol:
            break

    return weights_history, loss_history
