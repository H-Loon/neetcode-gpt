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

        b = 0.0
        n, d = X.shape 
        w = np.zeros(d)

        for _ in range(epochs):
            y_hat = X @ w + b
            err = y_hat - y

            # Calculate gradients
            dw = (2/n) * (X.T @ err)
            db = (2/n) * np.sum(err)

            # Update the weights and biase
            w = w - lr * dw
            b = b - lr * db

        return (np.round(w, 5), round(b, 5))
