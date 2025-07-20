import torch
import torch.nn.functional as F
from torch import nn

from cs336_basics.linear_utils import init_weight


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Construct a linear transformation module.

        Args:
            in_features: int Final dimension of the input
            out_features: int Final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(init_weight(out_features, in_features, **factory_kwargs))


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return F.linear(x, self.weight)
