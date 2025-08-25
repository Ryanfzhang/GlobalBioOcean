import numpy as np
import os
from tqdm import tqdm
import pandas as pd

def get_mean(path):
    phy_vars = ['thetao', 'so', 'uo', 'vo']
    dates= pd.date_range(start="20230101", end="20231231", freq="D")


    means= []
    for var in phy_vars:
        daily= []
        for i in tqdm(range(len(dates))):
            date = dates[i]
            data = np.load(os.path.join(path, "{}-{:02d}-{:02d}.nc.npy".format(var, date.month, date.day)))[0]
            daily.append(np.nanmean(data,axis=(1,2)))
        
        daily=  np.stack(daily, axis=0)
        mean = np.mean(daily, axis=0)
        means.append(mean)
    
    means = np.stack(means, axis=0)
    np.save("/home/mafzhang/code/bioocean/constant/mean_phys.npy", means)

if __name__=="__main__":
    get_mean("/home/mafzhang/data/GLORYS12/2023/")