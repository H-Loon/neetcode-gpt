import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean_val = round(x.mean().item(), 4)
                    std_val = round(x.std().item(), 4)
                    dead_neurons = (x <= 0).all(dim=0)
                    dead_fraction = round(dead_neurons.float().mean().item(), 4)
                    stats_dict = {'mean': mean_val, 'std': std_val, 'dead_fraction': dead_fraction}
                    stats.append(stats_dict)
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        loss_fn = nn.MSELoss()
        loss = loss_fn(model(x), y)
        loss.backward()

        stats = []
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                stats.append({
                    'mean': round(layer.weight.grad.mean().item(), 4),
                    'std': round(layer.weight.grad.std().item(), 4),
                    'norm': round(torch.norm(layer.weight.grad).item(), 4)
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(stats['dead_fraction'] > 0.5 for stats in activation_stats):
            return 'dead_neurons'

        elif any(stats['norm'] > 1000 for stats in gradient_stats):
            return 'exploding_gradients'
            
        elif (gradient_stats[-1]['norm'] < 1e-5):
            return 'vanishing_gradients'
        
        for stats in activation_stats:
            if stats['std'] < 0.1:
                return 'vanishing_gradients'
            if stats['std'] > 10.0:
                return 'exploding_gradients'
            
        return 'healthy'
