import numpy as np
import torch
from torch import nn


def init_weight(
    out_dim: int,
    in_dim: int,
    **factory_kwargs
) -> torch.Tensor:
    """
    Initializes weights using truncated normal distribution.
    """
    weight = torch.empty(out_dim, in_dim, **factory_kwargs)
    std = np.sqrt(2.0 / (in_dim + out_dim))
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    return weight
