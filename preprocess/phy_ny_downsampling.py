import numpy as np
import os
from tqdm import tqdm
import torch

def preprocess(ny_path, new_ny_path):
    files = os.listdir(ny_path)
    if not os.path.exists(new_ny_path):
        os.makedirs(new_ny_path)

    for i in tqdm(range(len(files))):
        data = np.load(os.path.join(ny_path, files[i]))
        data = torch.from_numpy(data)
        data = torch.nn.functional.pad(data, (0,0,1,1))
        tmp = torch.nn.functional.avg_pool2d(data, 3, 3)
        np.save(os.path.join(new_ny_path, files[i]), tmp.numpy())


if __name__=="__main__":
    preprocess("/home/mafzhang/data/GLORYS12/2023-bak-2", "/home/mafzhang/data/GLORYS12/2023")