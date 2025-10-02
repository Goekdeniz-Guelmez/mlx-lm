# Copyright © 2024-2025 Gökdeniz Gülmez

from typing import List, Union, Callable
import mlx.core as mx
import mlx.nn as nn


class ActivationFactory:
    _ACTIVATIONS = {
        'silu': lambda: nn.SiLU(),
        'gelu': lambda: nn.GELU(),
        'relu': lambda: nn.ReLU(),
        'relu2': lambda: nn.ReLU2(),
        'relu6': lambda: nn.ReLU6(),
        'tanh': lambda: nn.Tanh(),
        'sigmoid': lambda: nn.Sigmoid(),
        'identity': lambda x: x,
    }
    
    @classmethod
    def create(cls, activation: Union[str, Callable, None]) -> Callable:
        """Create activation function from various input types."""
        if activation is None:
            return lambda x: x
        
        if isinstance(activation, str):
            act_name = activation.lower()
            if act_name not in cls._ACTIVATIONS:
                raise ValueError(f"Unsupported activation: {activation}. "
                               f"Available: {list(cls._ACTIVATIONS.keys())}")
            return cls._ACTIVATIONS[act_name]()
        
        if callable(activation):
            try:
                # Try to instantiate if it's a class
                return activation()
            except TypeError:
                # Already a function
                return activation
        
        return activation


class KANLinear(nn.Module):
    """Optimized KAN Linear layer with B-spline basis functions."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: Callable = nn.SiLU(),
        grid_range: List[float] = [-1.0, 1.0],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation
        self.grid_range = grid_range
        
        # Initialize grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = mx.arange(-spline_order, grid_size + spline_order + 1, dtype=mx.float32) * h + grid_range[0]
        self.grid = mx.broadcast_to(grid[:, None], (len(grid), in_features))
        
        # Initialize weights
        self.base_weight = mx.random.normal((out_features, in_features)) * (scale_base / (in_features ** 0.5))
        self.spline_weight = mx.random.normal((out_features, in_features, grid_size + spline_order)) * (
            scale_spline / ((grid_size + spline_order) ** 0.5)
        )
    
    def b_splines(self, x: mx.array) -> mx.array:
        """Compute B-spline basis using Cox-de Boor recursion."""
        x = mx.clip(x, self.grid_range[0], self.grid_range[1])
        x = x.reshape(-1, self.in_features)
        x_expanded = x[:, :, None]
        
        # Order 0: indicator functions
        grid_left = self.grid[:-1].T[None, :, :]
        grid_right = self.grid[1:].T[None, :, :]
        bases = ((x_expanded >= grid_left) & (x_expanded < grid_right)).astype(mx.float32)
        
        # Higher orders via Cox-de Boor recursion
        for k in range(1, self.spline_order + 1):
            if bases.shape[-1] <= 1:
                break
            
            denom1 = mx.maximum(self.grid[k:-1].T[None, :, :] - self.grid[:-(k+1)].T[None, :, :], 1e-8)
            denom2 = mx.maximum(self.grid[(k+1):].T[None, :, :] - self.grid[1:(-k)].T[None, :, :], 1e-8)
            
            left = (x_expanded - self.grid[:-(k+1)].T[None, :, :]) / denom1 * bases[:, :, :-1]
            right = (self.grid[(k+1):].T[None, :, :] - x_expanded) / denom2 * bases[:, :, 1:]
            bases = left + right
        
        return bases
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        
        # Base transformation
        base_out = mx.matmul(self.base_activation(x_flat), self.base_weight.T)
        
        # Spline transformation
        bases = self.b_splines(x_flat)
        bases_flat = bases.reshape(bases.shape[0], -1)
        spline_weight_flat = self.spline_weight.reshape(self.out_features, -1)
        spline_out = mx.matmul(bases_flat, spline_weight_flat.T)
        
        # Combine and reshape
        output = base_out + spline_out
        if len(original_shape) == 3:
            output = output.reshape(original_shape[0], original_shape[1], self.out_features)
        
        return output
    
    def regularization_loss(self, l1_penalty: float = 1.0, entropy_penalty: float = 1.0) -> mx.array:
        """Compute regularization loss."""
        l1 = mx.abs(self.spline_weight).mean(axis=-1).sum()
        
        probs = mx.abs(self.spline_weight).flatten()
        probs = probs / (probs.sum() + 1e-8)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8))
        
        return l1_penalty * l1 + entropy_penalty * entropy