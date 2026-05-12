import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        gamma, beta = np.array(gamma), np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training:
            # Calcule les stats du batch actuel (moyenne par colonne)
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # Mise à jour des running stats
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var = (1 - momentum) * running_var + momentum * var
            
            # Utilise les stats du batch pour la normalisation
            mu, sigma_sq = mean, var
        else:
            # Utilise les running stats pour l'inférence
            mu, sigma_sq = running_mean, running_var

        # Normalisation et transformation affine
        x_hat = (x - mu) / np.sqrt(sigma_sq + eps)
        y = gamma * x_hat + beta

        # Retourne le tuple converti en listes arrondies à 4 décimales
        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )