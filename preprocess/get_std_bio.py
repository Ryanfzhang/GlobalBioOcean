import numpy as np
import os
from tqdm import tqdm
import pandas as pd

def get_mean(path):
    bio_vars = ['dissic', 'ph', 'talk', 'fe', 'no3', 'po4', 'si', 'chl', 'phyc', 'o2', 'nppv']
    dates= pd.date_range(start="20230101", end="20231231", freq="D")
    mean = np.load("/home/mafzhang/code/bioocean/constant/mean_bio.npy")


    stds= []
    for v in range(len(bio_vars)):
        var = bio_vars[v]
        daily= []
        for i in tqdm(range(len(dates))):
            date = dates[i]
            data = np.load(os.path.join(path, "{}-{:02d}-{:02d}.nc.npy".format(var, date.month, date.day)))[0]

            tmp = (data - mean[v][:,np.newaxis, np.newaxis])**2
            daily.append(np.nanmean(tmp,axis=(1,2)))
        
        daily=  np.stack(daily, axis=0)
        std = np.sqrt(np.mean(daily, axis=0))
        stds.append(std)
    
    stds = np.stack(stds, axis=0)
    np.save("/home/mafzhang/code/bioocean/constant/std_bio.npy", stds)

if __name__=="__main__":
    get_mean("/home/mafzhang/data/GLORYS12_bio/2023/")