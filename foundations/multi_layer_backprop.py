import numpy as np
from typing import List

class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:
        """
        Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> hat_y
        Loss: MSE = (hat_y - y_true)^2
        """
        out = {}

        # Convert inputs to numpy arrays
        x = np.asarray(x)
        W1 = np.asarray(W1)
        b1 = np.asarray(b1)
        W2 = np.asarray(W2)
        b2 = np.asarray(b2)
        y_true = np.asarray(y_true)

        # Forward pass
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        hat_y = z2

        # Loss calculation
        delta_loss = hat_y - y_true
        loss = float(np.mean(delta_loss ** 2))
        # Adding 0.0 forces -0.0 to 0.0
        out['loss'] = round(loss, 4)

        # Backward pass
        n = delta_loss.size if delta_loss.size > 0 else 1
        dz2 = 2 * delta_loss / n

        # Gradients
        dW2 = dz2[:, None] * a1[None, :]
        db2 = dz2
        da1 = dz2 @ W2
        mask = (z1 > 0).astype(float)
        dz1 = da1 * mask
        dW1 = dz1[:, None] * x[None, :]
        db1 = dz1

        # Patch: Adding 0.0 after rounding eliminates signed zero (-0.0)
        out['dW2'] = (np.round(dW2, 4) + 0.0).tolist()
        out['db2'] = (np.round(db2, 4) + 0.0).tolist()
        out['dW1'] = (np.round(dW1, 4) + 0.0).tolist()
        out['db1'] = (np.round(db1, 4) + 0.0).tolist()

        return out