import xarray as xr
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import netCDF4 as nc

from typing import Tuple, List
import torch
import random
from torch.utils import data
from torchvision import transforms as T
import os

class GlorysDataset(data.Dataset):
    """Dataset class for the Glorys dataset."""

    def __init__(self,
                 nc_path='/home/mafzhang/data/',
                 project_path='/home/mafzhang/code/bioocean',
                 seed=1234,
                 training=True,
                 validation=False,
                 startDate='20230101',
                 endDate='20231231',
                 freq='D',
                 ):
        """Initialize."""
        
        self.nc_path = nc_path
        self.glorys_phy = os.path.join(nc_path, "GLORYS12/2023/")
        self.glorys_bio = os.path.join(nc_path, "GLORYS12_bio/2023/")

        self.phy_vars = ['thetao', 'so', 'uo', 'vo']
        self.bio_vars = ['dissic', 'ph', 'talk', 'fe', 'no3', 'po4', 'si', 'chl', 'phyc', 'o2', 'nppv']
        self.phy_vars_mean = np.load(os.path.join(project_path, "constant/mean_phys.npy"))
        self.phy_vars_std= np.load(os.path.join(project_path, "constant/std_phys.npy"))
        self.bio_vars_mean = np.load(os.path.join(project_path, "constant/mean_bio.npy"))
        self.bio_vars_std= np.load(os.path.join(project_path, "constant/std_bio.npy"))
        self.land_mask = np.load(os.path.join(project_path, "constant/bio_scale_land_mask.npy"))
        

        self.training = training
        self.validation = validation
        self.seed = 0


        keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))
        train_keys, _, test_keys, _ = train_test_split(keys, keys, test_size=0.3, random_state=self.seed)
        if self.training:
            self.keys = train_keys
        elif self.validation:
            self.keys = test_keys
        
        self.day_of_year = np.array([i.day_of_year for i in self.keys]).astype(np.int32)

        self.length = len(self.keys) - 1
        random.seed(seed)

    def LoadPhy(self, key):
        """
        Input
            key: datetime object, input time
        Return
            input: numpy
            target: numpy label
            (start_time_str, end_time_str): string
        """

        data = []
        for var in self.phy_vars:
            data.append(np.load(os.path.join(self.glorys_phy, "{}-{:02d}-{:02d}.nc.npy".format(var, key.month, key.day))))
        data = np.concatenate(data, axis=0)
        return data
    
    def LoadBio(self, key):
        """
        Input
            key: datetime object, input time
        Return
            input: numpy
            target: numpy label
            (start_time_str, end_time_str): string
        """
        data = []
        for var in self.bio_vars:
            data.append(np.load(os.path.join(self.glorys_bio, "{}-{:02d}-{:02d}.nc.npy".format(var, key.month, key.day))))

        data = np.concatenate(data, axis=0)
        return data

    def __getitem__(self, index):
        """Return input frames, target frames, and its corresponding time steps."""
        iii = self.keys[index]
        time_str = datetime.strftime(iii, '%Y%m%d')
        phy= self.LoadPhy(iii)
        bio = self.LoadBio(iii)

        normalized_phy, normalized_bio= self.normalize(phy, bio)
        normalized_phy= np.nan_to_num(normalized_phy, nan=0.)
        normalized_bio= np.nan_to_num(normalized_bio, nan=0.)
        day_of_year = self.day_of_year[index]

        return normalized_phy.astype(np.float32), normalized_bio.astype(np.float32), day_of_year, self.land_mask, self.bio_vars_mean, self.bio_vars_std, time_str, phy, bio
    
    def normalize(self, phy, bio):
        phy = (phy - self.phy_vars_mean[:, :, np.newaxis, np.newaxis])/(self.phy_vars_std[:, :, np.newaxis, np.newaxis]+1e-5)
        bio= (bio - self.bio_vars_mean[:, :, np.newaxis, np.newaxis])/(self.bio_vars_std[:, :, np.newaxis, np.newaxis]+1e-5)
        return phy, bio 

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


if __name__=="__main__":
    dataset = GlorysDataset()
    print(dataset.__getitem__(10)[-2][0,0].shape)
    print(dataset.__getitem__(10)[1][0,0].shape)