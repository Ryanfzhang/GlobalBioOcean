import argparse
import os
import time
import torch
import random
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from timm.utils import AverageMeter
import torch.nn.functional as F
import pandas as pd
from transformers import get_cosine_schedule_with_warmup


from utils import check_dir, seed_everything
from data.glorys_dataset import GlorysDataset
from model.model import Multi_source_integrate

fix_seed = 2025
seed_everything(fix_seed)

parser = argparse.ArgumentParser(description='Ocean Forecasting')

# data loader
parser.add_argument('--checkpoints', type=str, default='./checkpoints/multi_source_integrate/', help='location of model checkpoints')
parser.add_argument('--dataset_path', type=str, default='/home/mafzhang/data/', help='location of dataset')

parser.add_argument('--levels', type=int, default=23, help='input sequence length')
parser.add_argument('--patch_size', type=int, default=24, help='input sequence length')

# model
parser.add_argument('--hidden_size', type=int, default=256, help='input sequence length')

# optimization
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='optimizer wd')
parser.add_argument('--loss', type=str, default='mae', help='loss function')

args = parser.parse_args()


train_dataset = GlorysDataset(nc_path=args.dataset_path, project_path="/home/mafzhang/GlobalBioOcean")
train_dloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=1)
test_dataset = GlorysDataset(nc_path=args.dataset_path, project_path="/home/mafzhang/GlobalBioOcean")
test_dloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4, prefetch_factor=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Multi_source_integrate(patch_size=args.patch_size, levels=args.levels, embed_dim=args.hidden_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.995))
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps= 1000, 
    num_training_steps=len(train_dloader) * args.train_epochs,
)

accelerator = Accelerator()
device = accelerator.device
train_dloader = accelerator.prepare_data_loader(train_dloader)
test_dloader = accelerator.prepare_data_loader(test_dloader)
model = accelerator.prepare_model(model)
optimizer = accelerator.prepare_optimizer(optimizer)
lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

best_mse_sst, best_mse_salt = 100, 100

# if accelerator.is_main_process:
check_dir(args.checkpoints)
# accelerator.print("Training start")
print("Training start")
criteria = torch.nn.MSELoss(reduction='none')

for epoch in range(args.train_epochs):
    train_loss = AverageMeter()
    model.train()
    epoch_time = time.time()
    for i, (x_phy, x_bio, time, land_mask, _, _, _) in tqdm(enumerate(train_dloader), total=len(train_dloader), disable=(not accelerator.is_local_main_process)):
    # for i, (x_phy, x_bio, time, land_mask, _, _, _) in tqdm(enumerate(train_dloader), total=len(train_dloader)):
        x_phy, x_bio, time, land_mask = x_phy.to(device), x_bio.to(device), time.to(device), land_mask.to(device)
        x_bio_recon_f_phy, x_bio_recon_f_bio, x_phy_latent, x_bio_latent = model(x_phy, x_bio, time)

        recon_loss= criteria(x_bio_recon_f_bio, x_bio) + criteria(x_bio_recon_f_bio, x_bio)
        teacher_loss = criteria(x_phy_latent, x_bio_latent.detach())

        loss = (land_mask * recon_loss).mean() + 0.1 * teacher_loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss.update(loss.detach().cpu().item())
        torch.cuda.empty_cache()

    print("Epoch: {} | Train Loss: {:.4f}, Cost Time: {:.4f}".format(epoch, train_loss.avg, time.time()-epoch_time))

    train_loss.reset()

    if epoch%10==0: 
        with torch.no_grad():
            recon_f_bio_mse_list, recon_f_phy_mse_list = [], []
            for i, (x_phy, x_bio, time, land_mask, bio_mean, bio_std, _) in tqdm(enumerate(test_dloader), total=len(test_dloader), disable=(not accelerator.is_local_main_process)):
                x_bio_recon_f_phy, x_bio_recon_f_bio, x_phy_latent, x_bio_latent = model(x_phy, x_bio, time)
                land_mask = land_mask[:,None, None, ...]

                # denormalize
                x_bio_recon_f_phy = x_bio_recon_f_phy * bio_std[...,None, None] + bio_mean[..., None, None]
                x_bio_recon_f_bio= x_bio_recon_f_bio * bio_std[...,None, None] + bio_mean[..., None, None]

                recon_f_phy= criteria(x_bio_recon_f_phy, x_bio)
                recon_f_phy_mse = (land_mask * recon_f_phy).sum([3,4])/ land_mask.float().sum([3,4])

                recon_f_bio= criteria(x_bio_recon_f_bio, x_bio)
                recon_f_bio_mse = (land_mask * recon_f_bio).sum([3,4])/ land_mask.float().sum([3,4])

                recon_f_phy_mse = accelerator.gather(recon_f_phy_mse)
                recon_f_bio_mse = accelerator.gather(recon_f_bio_mse)
                recon_f_bio_mse_list.append(recon_f_bio_mse.flatten(0,1).detach().cpu().numpy())
                recon_f_phy_mse_list.append(recon_f_phy_mse.flatten(0,1).detach().cpu().numpy())
            
            mean_recon_f_bio_mse = np.concatenate(recon_f_bio_mse_list, axis=0).mean(0)
            mean_recon_f_phy_mse = np.concatenate(recon_f_phy_mse_list, axis=0).mean(0)

            accelerator.print("Test" + "*"*48)
            accelerator.print(mean_recon_f_bio_mse)
            accelerator.print(mean_recon_f_phy_mse)
