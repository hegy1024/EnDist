import torch.nn as nn
from typing import Optional
from torch import Tensor
from torch.nn import Linear, Sequential

from .utils import create_normalization, create_activation

class Residual(nn.Module):
    """
    A residual block.
    """
    def __init__(self, fnc):
        super().__init__()
        self.fnc = fnc

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.fnc(x, *args, **kwargs) + x

class MLPBlock(nn.Module):
    """
    An implementation of MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int,
                 norm: str = 'ln', activation: str = 'relu', out_activation: Optional[str] = 'relu'):
        super(MLPBlock, self).__init__()
        self.in_proj = Linear(input_dim, hidden_dim)
        self.res_mlp = Residual(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                create_normalization(norm)(hidden_dim),
                create_activation(activation),
                Linear(hidden_dim, hidden_dim)
            )
        )
        self.out_proj = Linear(hidden_dim, out_dim)
        self.act = create_activation(out_activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        x = self.res_mlp(x)
        x = self.out_proj(x)
        x = self.act(x)

        return x

