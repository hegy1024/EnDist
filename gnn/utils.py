import torch
import torch.nn as nn
import torch_geometric.nn as pygnn

from munch import Munch
from typing import *

def masked_instance_norm2d(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    Instance normalization for 2D feature maps with mask
    :param x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    :param mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    return: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    zero_indices = torch.where(torch.sum(mask, dim=[1, 2]) < 0.5)[0].squeeze(-1)  # [N,]
    mean = torch.sum(x * mask, dim=[1, 2]) / (torch.sum(mask, dim=[1, 2]))  # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask) ** 2  # (N,L,L,C)
    var = torch.sum(var_term, dim=[1, 2]) / (torch.sum(mask, dim=[1, 2])) + 1e-5  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)  # (N, L, L, C)
    instance_norm = instance_norm * mask
    instance_norm[zero_indices, :, :, :] = 0  # 标准化
    return instance_norm

def create_normalization(name: Optional[str] = None):
    """
    Args:
        name:
            'lr': 表示layernorm
            'bn': 表示batchnorm
    """
    if name is None:
        # nothing to do
        return nn.Identity
    elif name == 'ln':
        # layer norm
        return nn.LayerNorm
    elif name == 'bn':
        # batch norm
        return nn.BatchNorm1d
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_activation(name: Optional[str] = None):
    if name is None:
        # nothing to do
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f"{name} is not implemented.")