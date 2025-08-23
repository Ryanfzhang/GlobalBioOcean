import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_, Mlp, PatchEmbed
from functools import lru_cache
import numpy as np
import os
import sys
from einops import rearrange
sys.path.insert(0, os.path.join(os.getcwd(), "../"))

from model.encoder import Encoder
from model.backbone import Block
from model.decoder import Decoder


class Multi_source_integrate(nn.Module):
    def __init__(self, 
        path="/home/mafzhang/code/bioocean",
        img_size=(681,1440),
        latent_levels=5,
        levels=23,
        patch_size=2,
        embed_dim=256,
        num_heads=16,
    ):
        super().__init__()
        
        # embedding
        self.encoder = Encoder(path=path, img_size=img_size, latent_levels=latent_levels, levels=levels, patch_size=patch_size, embed_dim=embed_dim)
        self.blocks = Block(hidden_size=embed_dim, num_heads=num_heads)
        self.decoder = Decoder(path=path, img_size=img_size, levels=levels, patch_size=patch_size, embed_dim=embed_dim)

    def forward(self, x_phy, x_bio, time):
        print(x_phy.shape)
        print(x_bio.shape)
        B, V, L, H, W = x_phy.shape
        
        x_phy, x_bio = self.encoder(x_phy, x_bio, time) # B, L, D
        x_phy_ = rearrange(x_phy, "B V L D ->(B V) L D")
        x_bio_ = rearrange(x_bio, "B V L D ->(B V) L D")
        x_phy_latent= self.blocks(x_phy_)
        x_bio_latent = self.blocks(x_bio_)
        
        x_phy_latent = rearrange(x_phy_latent, "(B V) L D ->B V L D", B=B)
        x_bio_latent = rearrange(x_bio_latent, "(B V) L D ->B V L D", B=B)
        x_bio_recon_f_phy = self.decoder(x_phy_latent)
        x_bio_recon_f_bio= self.decoder(x_bio_latent)
        
        return x_bio_recon_f_phy, x_bio_recon_f_bio, x_phy_latent, x_bio_latent