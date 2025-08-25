#! /bin/bash
#SBATCH --job-name=Bio
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -p normal
#SBATCH --mail-user=mafzhang@ust.hk
#SBATCH --time=72:00:00
#SBATCH --account=earthhk
#SBATCH --output=logs/Bioocean.first
#SBATCH --cpus-per-gpu=8

###############################

accelerate launch train.py --dataset_path="/project/earthhk/" --project_path="/home/mafzhang/GlobalBioOcean"

###############################