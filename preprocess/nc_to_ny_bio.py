import netCDF4 as nc
import numpy as np
import os
from tqdm import tqdm

def preprocess(nc_path, ny_path):
    files = os.listdir(nc_path)
    if not os.path.exists(ny_path):
        os.makedirs(ny_path)
    vars_carbon = ['dissic', 'ph', 'talk']
    vars_nutrients = ['fe', 'no3', 'po4', 'si']
    vars_phyto = ['chl', 'phyc'] 
    vars_o2 = ['o2', 'nppv']

    for i in tqdm(range(len(files))):
        data = nc.Dataset(os.path.join(nc_path, files[i]), 'r')
        _, type, month, day = files[i].split("-")

        if type=="carbon":
            for var in vars_carbon:
                tmp = np.array(data.variables[var]).astype(np.float32)
                tmp[np.bitwise_or(tmp>5000, tmp<-3000)]=np.nan
                np.save(os.path.join(ny_path, "{}-{}-{}.npy".format(var, month, day)), tmp)

        elif type=="nutrients":
            for var in vars_nutrients:
                tmp = np.array(data.variables[var]).astype(np.float32)
                tmp[np.bitwise_or(tmp>5000, tmp<-3000)]=np.nan
                np.save(os.path.join(ny_path, "{}-{}-{}.npy".format(var, month, day)), tmp)

        elif type=="Phyto":
            for var in vars_phyto:
                tmp = np.array(data.variables[var]).astype(np.float32)
                tmp[np.bitwise_or(tmp>5000, tmp<-3000)]=np.nan
                np.save(os.path.join(ny_path, "{}-{}-{}.npy".format(var, month, day)), tmp)

        elif type=="o2":
            for var in vars_o2:
                tmp = np.array(data.variables[var]).astype(np.float32)
                tmp[np.bitwise_or(tmp>5000, tmp<-3000)]=np.nan
                np.save(os.path.join(ny_path, "{}-{}-{}.npy".format(var, month, day)), tmp)

if __name__=="__main__":
    preprocess("/home/mafzhang/data/GLORYS12_bio/2023-nc", "/home/mafzhang/data/GLORYS12_bio/2023")