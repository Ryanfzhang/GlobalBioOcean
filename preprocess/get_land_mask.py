import numpy as np
import os
from tqdm import tqdm
import pandas as pd

def get_land_mask(phy_path, bio_path, project_path):
    phy = np.load(os.path.join(phy_path, "thetao-01-01.nc.npy"))
    phy_scale_land_mask = 1- np.isnan(phy[0,0]).astype(np.float32)
    bio = np.load(os.path.join(bio_path, "dissic-01-01.nc.npy"))
    bio_scale_land_mask = 1- np.isnan(bio[0,0]).astype(np.float32)
    np.save(os.path.join(project_path, "phy_scale_land_mask.npy"), phy_scale_land_mask)
    np.save(os.path.join(project_path, "bio_scale_land_mask.npy"), bio_scale_land_mask)

if __name__=="__main__":
    get_land_mask("/home/mafzhang/data/GLORYS12/2023/", "/home/mafzhang/data/GLORYS12_bio/2023/", "/home/mafzhang/code/bioocean/constant")