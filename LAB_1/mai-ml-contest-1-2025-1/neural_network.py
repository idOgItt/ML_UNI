import numpy as np

def linear_forward(X, W, b):
    Z = X.dot(W) + b
    return Z, (X, W, b)

def relu_forward(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def identity_forward(Z):
    return Z, None

def linear_backward(dZ, cache):
    X, W, b = cache
    n = X.shape[0]
    dW = X.T.dot(dZ) / n
    db = dZ.sum(axis=0) / n
    dX = dZ.dot(W.T)
    return dX, dW, db

def relu_backward(dA, cache):
    Z = cache
    dZ = dA.copy()
    dZ[Z <= 0] = 0
    return dZ

def compute_loss_mse(A_pred, y_true):
    A = np.asarray(A_pred)
    y = np.asarray(y_true)
    if A.ndim > 1 and y.ndim == 1:
        y = y.reshape(-1, 1)

    m = y.shape[0]
    diff = A - y
    loss = np.mean(diff**2) / 2
    dA = diff / m
    return loss, dA

class TwoLayerNet:
    def __init__(self, n_input, n_hidden, n_output):
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(n_input, n_hidden) * 0.01
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, n_output) * 0.01
        self.b2 = np.zeros(n_output)

    def forward(self, X):
        Z1, cache1 = linear_forward(X, self.W1, self.b1)
        A1, cache_relu = relu_forward(Z1)
        Z2, cache2 = linear_forward(A1, self.W2, self.b2)
        A2, _      = identity_forward(Z2)
        self.caches = (cache1, cache_relu, cache2)
        return A2

    def backward(self, A2, y):
        cache1, cache_relu, cache2 = self.caches
        loss, dA2 = compute_loss_mse(A2, y)
        dZ2 = dA2
        dA1, dW2, db2 = None, None, None
        dA1, dW2, db2 = linear_backward(dZ2, cache2)
        dZ1 = relu_backward(dA1, cache_relu)
        dX, dW1, db1 = linear_backward(dZ1, cache1)
        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2
        }
        return loss, grads

    def update_parameters(self, grads, lr):
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']
