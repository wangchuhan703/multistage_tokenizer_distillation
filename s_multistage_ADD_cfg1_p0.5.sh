#!/bin/bash
#SBATCH --job-name="train_lpips_dino"
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=80G
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate efficientvit

cd /work/hdd/bcey/hchen10/rule_tokenizer-main
torchrun --master_port=29501 --nproc_per_node=2 train/train_tokenizer_diff_ADD.py  --config experiments/checkpoints/0240000.pt --p_loss_weight 0.5