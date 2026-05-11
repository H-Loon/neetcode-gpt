import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std

        return [[round(x, 4) for x in row] for row in weights.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std

        return [[round(x, 4) for x in row] for row in weights.tolist()]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        
        # 1. Générer TOUS les poids d'abord (consomme le seed en premier)
        all_weights = []
        for i in range(num_layers):
            f_in = input_dim if i == 0 else hidden_dim
            f_out = hidden_dim
            
            match init_type:
                case "xavier":
                    std = math.sqrt(2.0 / (f_in + f_out))
                case "kaiming":
                    std = math.sqrt(2.0 / f_in)
                case "random":
                    std = 1.0
            
            all_weights.append(torch.randn(f_out, f_in) * std)

        # 2. Générer l'entrée APRES les poids
        current_x = torch.randn(1, input_dim)
        
        # 3. Propagation
        stds = []
        for weights in all_weights:
            z = current_x @ weights.T
            current_x = torch.relu(z)
            stds.append(round(torch.std(current_x).item(), 2))
            
        return stds
