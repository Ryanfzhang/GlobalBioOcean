import netCDF4 as nc
import numpy as np
import os
from tqdm import tqdm

def preprocess(nc_path, ny_path):
    files = os.listdir(nc_path)
    if not os.path.exists(ny_path):
        os.makedirs(ny_path)
    vars = ['thetao', 'so', 'uo', 'vo']

    for i in tqdm(range(len(files))):
        data = nc.Dataset(os.path.join(nc_path, files[i]), 'r')
        _, month, day = files[i].split("-")
        for var in vars:
            tmp = np.array(data.variables[var]).astype(np.float32)
            tmp[np.bitwise_or(tmp>5000, tmp<-3000)]=np.nan
            np.save(os.path.join(ny_path, "{}-{}-{}.npy".format(var, month, day)), tmp)


if __name__=="__main__":
    preprocess("/home/mafzhang/data/GLORYS12/2023-nc", "/home/mafzhang/data/GLORYS12/2023")