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
torchrun --master_port=29502 --nproc_per_node=2 train/train_tokenizer_diff_ADD.py --config configs/diff_in1k/exp015-aediff16-latent_128d32-enc_mmditd12-dec_mmditd12_ms3-cfg-cross_rope--sigmoidweight-lognormal-percepstart_2.yaml 