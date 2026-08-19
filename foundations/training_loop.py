import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        b = 0.0
        n, d = X.shape 
        w = np.zeros(d)
        half_n = 2/n
        Xt = X.T
        for _ in range(epochs):
            y_hat = np.dot(X, w) + b
            err = y_hat - y
            # Calculate gradients
            dw = half_n * np.dot(Xt, err)
            db = half_n * np.sum(err)
            w = w - lr * dw
            b = b - lr * db
        return (np.round(w, 5), round(b, 5))
