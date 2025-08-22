import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_, Mlp, PatchEmbed
from functools import lru_cache
from einops import rearrange
from typing import Optional

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))
from model.attention import PerceiverResampler
from model.embedding import FourierExpansion
from model.utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, get_pad2d


class Decoder(nn.Module):
    """Multi-scale multi-source multi-variable decoder based on the Perceiver architecture."""

    def __init__(
        self,
        path="/home/mafzhang/code/bioocean",
        img_size=(681,1440),
        levels=23,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
    ) -> None:
        """Initialise.

        Args:
        """
        super().__init__()

        self.project_path = path
        padding = get_pad2d(img_size, (patch_size, patch_size))
        self.padding = padding
        img_size = (img_size[0]+padding[0]+padding[1], img_size[1]+padding[2]+padding[3])
        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = 11
        self.levels = levels
        self.embed_dim = embed_dim

        self.level_decoder = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=2,
            head_dim=64,
            num_heads=num_heads,
            drop=0.1,
            mlp_ratio=4.0,
            ln_eps=1e-5,
            ln_k_q=False,
        )
        
        self.levels = levels
        self.level_final = nn.Parameter(torch.randn(levels, embed_dim))

        self.head = nn.Linear(embed_dim, self.variables*patch_size**2)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def deaggregate_levels(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        B, _ , N, _ = x.shape
        latents = self.level_final
        latents = latents.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("blnd->bnld", x)
        x = x.flatten(0, 1)  # (B * N, L, D)
        latents = torch.einsum("blnd->bnld", latents)
        latents = latents.flatten(0, 1)  # (B * N, L_latent, D)

        x = self.level_decoder(latents, x)  # (B * N, L_latent, D)
        x = x.unflatten(dim=0, sizes=(B, N))  # (B, N, L_latent, D)
        x = torch.einsum("blcd->bcld", x)  # (B, N, L_latent, D)
        return x
    
    def unpatchify(self, x: torch.Tensor, h=None, w=None):
    
        p = self.patch_size
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p

        x = rearrange(x, "B L (H W) (V Ph Pw) -> B V L (H Ph) (W Pw)", H=h, W=w, V=self.variables, Ph=p, Pw = p)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
    
        B, L, N, D = x.shape
        x = self.deaggregate_levels(x) # (B, L_latent, N, D) to (B, L, N, D) 
        x = self.head(x) # (B, L, N, D) to (B, L, N, V*P**2)
        x = self.unpatchify(x) # (B, L, N, V*P**2) to (B, L, N, H, W)

        right_h = -self.padding[1] if self.padding[1]>0 else None
        right_w = -self.padding[3] if self.padding[3]>0 else None
        return x[:,:,:, self.padding[0]:right_h, self.padding[2]:right_w]



if __name__=="__main__":
    decoder = Decoder((33,65))
    input = torch.rand(1,8,512,1024)
    y_mark = torch.Tensor([[11, 30]])
    print(decoder(input, y_mark).shape)