import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(
        self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float
    ) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = x @ w + b
        y_hat = 1 / (1 + np.exp(-z))

        dL_db = (y_hat - y_true) * y_hat * (1 - y_hat)
        dL_dw = dL_db * x

        return (np.round(dL_dw, 5), round(dL_db, 5))
