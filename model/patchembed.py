"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple
from einops import rearrange



class LevelPatchEmbed(nn.Module):
    """At either the surface or at a single pressure level, maps all variables into a single
    embedding."""

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        type:str = "phy",
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ) -> None:
        """Initialise.

        Args:
            var_names (tuple[str, ...]): Variables to embed.
            patch_size (int): Patch size.
            embed_dim (int): Embedding dimensionality.
            norm_layer (torch.nn.Module, optional): Normalisation layer to be applied at the very
                end. Defaults to no normalisation layer.
            flatten (bool): At the end of the forward pass, flatten the two spatial dimensions
                into a single dimension. See :meth:`LevelPatchEmbed.forward` for more details.
        """
        super().__init__()

        var_names= ['thetao', 'so', 'uo', 'vo'] if type=="phy" else ['dissic', 'ph', 'talk', 'fe', 'no3', 'po4', 'si', 'chl', 'phyc', 'o2', 'nppv']
        self.var_names = var_names
        self.kernel_size =  to_2tuple(patch_size)
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.weights = nn.ParameterList(
            [
                # Shape (C_out, C_in, T, H, W). `C_in = 1` here because we're embedding every
                # variable separately.
                nn.Parameter(torch.empty(embed_dim, 1, *self.kernel_size))
                for name in var_names
            ]
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        # Setting `a = sqrt(5)` in kaiming_uniform is the same as initialising with
        # `uniform(-1/sqrt(k), 1/sqrt(k))`, where `k = weight.size(1) * prod(*kernel_size)`.
        # For more details, see
        #
        #   https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        #
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # The following initialisation is taken from
        #
        #   https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d
        #
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights)))
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the embedding.

        Args:
            x (:class:`torch.Tensor`): Tensor to embed of a shape of `(B, V, T, H, W)`.
            var_names (tuple[str, ...]): Names of the variables in `x`. The length should be equal
                to `V`.

        Returns:
            :class:`torch.Tensor`: Embedded tensor a shape of `(B, L, D]) if flattened,
                where `L = H * W / P^2`. Otherwise, the shape is `(B, D, H', W')`.

        """
        B, V, L, H, W = x.shape
        x = rearrange(x, "B V L H W -> (B L) V H W")

        # Select the weights of the variables and history dimensions that are present in the batch.
        weight = torch.cat(
            [
                # (C_out, C_in, H, W)
                self.weights[i]
                for i in range(len(self.var_names))
            ],
            dim=1,
        )
        # Adjust the stride if history is smaller than maximum.
        stride =  self.kernel_size

        # The convolution maps (B, V, H, W) to (B, D, H/P, W/P)
        proj = F.conv2d(x, weight, self.bias, stride=stride)
        proj = proj.flatten(2,3)

        x = self.norm(proj)
        x = rearrange(x, "(B L) D N-> B L N D", L=L)
        return x
