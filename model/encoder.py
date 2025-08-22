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
from model.patchembed import LevelPatchEmbed
from model.utils import get_2d_sincos_pos_embed, get_pad2d
from model.embedding import FourierExpansion


class Encoder(nn.Module):
    def __init__(
        self,
        path="/home/mafzhang/code/bioocean",
        img_size=(681,1440),
        latent_levels=5,
        levels=23,
        patch_size=2,
        embed_dim=256,
        num_heads=16,
    ):
        super().__init__()

        self.project_path = path
        padding = get_pad2d(img_size, (patch_size, patch_size))
        self.padding = padding
        img_size = (img_size[0]+padding[0]+padding[1], img_size[1]+padding[2]+padding[3])
        self.img_size = img_size
        self.patch_size = patch_size
        self.phy_variables = 4 
        self.bio_variables = 11

        self.phy_token_embeds = LevelPatchEmbed(patch_size=patch_size, embed_dim=embed_dim, type="phy")
        self.bio_token_embeds = LevelPatchEmbed(patch_size=patch_size, embed_dim=embed_dim, type="bio")

        self.levels = levels
        self.latent_levels = latent_levels
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.level_latents = nn.Parameter(torch.randn(latent_levels, embed_dim))
        self.grid = nn.Parameter(self.make_lati_lon_meshgrid(), requires_grad=False)

        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=1,
            head_dim=64,
            num_heads=num_heads,
            drop=0.1,
            mlp_ratio=4.0,
            ln_eps=1e-5,
            ln_k_q=False,
        )

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(0.1)

        # geo position embedding (longitude, latitude)
        self.position_expansion = FourierExpansion(lower=0.01, upper=1440, d=embed_dim//2)
        self.absolute_time_expansion = FourierExpansion(lower=1, upper=366, d=embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed( self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        
        B, _ , N, _ = x.shape
        latents = self.level_latents
        latents = latents.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("blnd->bnld", x)
        x = x.flatten(0, 1)  # (B * N, L, D)
        latents = torch.einsum("blnd->bnld", latents)
        latents = latents.flatten(0, 1)  # (B * N, L_latent, D)

        x = self.level_agg(latents, x)  # (B * N, L_latent, D)
        x = x.unflatten(dim=0, sizes=(B, N))  # (B, N, L_latent, D)
        x = torch.einsum("blcd->bcld", x)  # (B, N, L_latent, D)
        return x

    def forward(self, x_phy: torch.Tensor, x_bio: torch.Tensor, time: torch.Tensor):

        # encode
        B, V, L, H, W = x_phy.shape
        x_phy = F.pad(x_phy, (0,0)+self.padding)
        x_bio = F.pad(x_bio, (0,0)+self.padding)
        x_phy = self.phy_token_embeds(x_phy) # (B, V, L, H, W) to (B L N D) N=H*W/p**2
        x_phy = self.aggregate_levels(x_phy)  

        x_bio = self.bio_token_embeds(x_bio) # (B, V, L, H, W) to (B L N D) N=H*W/p**2
        x_bio = self.aggregate_levels(x_bio)  

        # position embedding in ViT
        x_phy = rearrange(x_phy, "B L N D-> (B L) N D")
        x_phy = x_phy + self.pos_embed

        x_bio = rearrange(x_bio, "B L N D-> (B L) N D")
        x_bio = x_bio + self.pos_embed

        # earth embedding and time embedding
        earth_pos_embedding = self.earth_embedding()
        earth_pos_embedding = earth_pos_embedding.flatten(1,2)
        time_embedding  = self.absolute_time_expansion(time)
        time_embedding = time_embedding[:,None,...]
        x_phy = x_phy + earth_pos_embedding + time_embedding
        x_bio = x_bio + earth_pos_embedding + time_embedding

        x_phy = self.dropout(x_phy)
        x_bio = self.dropout(x_bio)
        x_phy = rearrange(x_phy, "(B L) N D-> B L N D", B=B)
        x_bio = rearrange(x_bio, "(B L) N D-> B L N D", B=B)
        return x_phy, x_bio
    
    def earth_embedding(self):

        grid_h = F.avg_pool2d(self.grid[0][None,...], self.patch_size)
        grid_w = F.avg_pool2d(self.grid[1][None,...], self.patch_size)
        encode_h = self.position_expansion(grid_h)
        encode_w = self.position_expansion(grid_w)
        pos_encode = torch.cat([encode_h, encode_w], dim=-1)
        return pos_encode

    def make_lati_lon_meshgrid(self):

        lati = np.load(os.path.join(self.project_path, "constant/lati.npy"))
        lati = lati.tolist()
        lati = [lati[0]]* self.padding[0] + lati + [lati[-1]]*self.padding[1]
        lati = torch.Tensor(lati)

        lon = np.load(os.path.join(self.project_path, "constant/lon.npy"))
        lon= lon.tolist()
        lon = [lon[0]]* self.padding[0] + lon + [lon[-1]]*self.padding[1]

        lon = torch.Tensor(lon)
        grid = torch.meshgrid(lati, lon, indexing='xy')
        grid = torch.stack(grid)
        grid = grid.permute(0, 2, 1)

        return grid


if __name__=="__main__":
    encoder = Encoder((32,64))
    input = torch.rand(1, 19, 30, 32, 64)
    print(encoder(input).shape)