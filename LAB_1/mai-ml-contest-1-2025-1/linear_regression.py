import numpy as np
from gradient_descent import (
    batch_gradient_descent,
    stochastic_gradient_descent
)

class LinearRegressionManual:
    def __init__(
        self,
        method='analytic',
        alpha=0.01,
        n_iters=1000,
        tol=1e-6,
        shuffle=True,
        random_state=None
    ):
        self.method = method
        self.alpha = alpha
        self.n_iters = n_iters
        self.tol = tol
        self.shuffle = shuffle
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_arr = np.asarray(X, float)
        y_arr = np.asarray(y, float)
        n_samples, n_features = X_arr.shape

        X_b = np.hstack([np.ones((n_samples,1)), X_arr])

        if self.method == 'analytic':
            w = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_arr)

        elif self.method == 'gradient':
            w_hist, _ = batch_gradient_descent(
                X_b, y_arr,
                alpha=self.alpha,
                n_iters=self.n_iters,
                tol=self.tol
            )
            w = w_hist[-1]

        elif self.method == 'sgd':
            w_hist, _ = stochastic_gradient_descent(
                X_b, y_arr,
                alpha=self.alpha,
                n_iters=self.n_iters,
                tol=self.tol,
                shuffle=self.shuffle
            )
            w = w_hist[-1]

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.intercept_ = w[0]
        self.coef_      = w[1:]
        return self

    def predict(self, X):
        return np.asarray(X, float).dot(self.coef_) + self.intercept_
