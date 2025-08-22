import numpy as np
import os
from tqdm import tqdm
import pandas as pd

def get_mean_std(path):
    bio_vars = ['dissic', 'ph', 'talk', 'fe', 'no3', 'po4', 'si', 'chl', 'phyc', 'o2', 'nppv']
    dates= pd.date_range(start="20230101", end="20231231", freq="D")
    depth = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


    means= []
    stds= []
    for var in bio_vars:
        daily= []
        for i in tqdm(range(len(dates))):
            date = dates[i]
            data = np.load(os.path.join(path, "{}-{:02d}-{:02d}.nc.npy".format(var, date.month, date.day)))[0,depth]
            daily.append(np.nanmean(data,axis=(1,2)))
        
        daily=  np.stack(daily, axis=0)
        mean = np.mean(daily, axis=0)
        std = np.mean(daily, axis=0)
        means.append(mean)
        stds.append(std)
    
    means = np.stack(means, axis=0)
    stds= np.stack(stds, axis=0)
    np.save("/home/mafzhang/code/bioocean/constant/mean_bio.npy", means)
    np.save("/home/mafzhang/code/bioocean/constant/std_bio.npy", stds)

if __name__=="__main__":
    get_mean_std("/home/mafzhang/data/GLORYS12_bio/2023/")