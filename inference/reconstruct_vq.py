import torch
import time
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import os
import os
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image
import numpy as np
import argparse
import itertools
import ruamel.yaml as yaml
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from utils.model import build_tokenizer
import random

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed import init_distributed_mode
from utils.data import center_crop_arr
from modelling.tokenizer import SoftVQModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import wandb

def visualize_random_grids(sample_dir, wandb_logger, grid_size=8, num_grids=10):
    """
    Visualize and save 10 random 8x8 image grids from the sample folder.
    """
    all_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png')])
    if len(all_files) < grid_size * grid_size:
        print(f"Not enough images to build a grid: found {len(all_files)}.")
        return

    name = sample_dir.split('/')[-1]
    group = sample_dir.split('/')[-2]

    
    for grid_idx in range(num_grids):
        chosen_files = random.sample(all_files, grid_size * grid_size)
        images = [TF.to_tensor(Image.open(os.path.join(sample_dir, f))) for f in chosen_files]
        grid = make_grid(images, nrow=grid_size, padding=2)
        grid_image = TF.to_pil_image(grid)
        # grid_path = os.path.join(save_dir, f"grid_{grid_idx}.png")
        # grid_image.save(grid_path)
        wandb_logger.log({"recon_images": [wandb.Image(grid_image)]}, step=grid_idx)
        # print(f"Saved image grid to {grid_path}")
        

def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path



def main(args):
    total_gen_time = 0.0  # 总生成时间
    total_images = 0      # 总生成图像数
    batch_times = []      # 每个 batch 的时间
    
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    vq_model, _, _, _, _, _, _  = build_tokenizer(args.config, args.vq_ckpt)
    vq_model.eval()
    vq_model = vq_model.to(device)
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
        
    # Create folder to save samples:
    folder_name = '/'.join(args.vq_ckpt.split('/')[-3:-2])
    # folder_name = (f"{args.vq_model}-{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}-seed-{args.global_seed}")
    sample_folder_dir = f"{args.sample_dir}/{folder_name}/cfg{args.cfg_scale}-steps{args.num_inference_steps}"
    if args.clip: sample_folder_dir += '-clip'
    # sample_grid_dir = f"recon_vis/{folder_name}/grid-cfg{args.cfg_scale}-steps{args.num_inference_steps}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
        name = f'cfg{args.cfg_scale}-steps{args.num_inference_steps}'
        if args.clip: name += '-clip'
        group = folder_name
        wandb_logger = wandb.init(project='diff_recon_vis', name=name, group=group)
        wandb.define_metric("cfg")
        # set all other train/ metrics to use this step
        wandb.define_metric("rFID", step_metric="cfg")
        wandb.define_metric("PSNR", step_metric="cfg")
        wandb.define_metric("SSIM", step_metric="cfg")
    dist.barrier()

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(args.data_path, transform=transform)
    num_fid_samples = 50000
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )    

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    
    psnr_val_rgb = []
    ssim_val_rgb = []
    compute_fid_score = FrechetInceptionDistance(normalize=False).cuda()
    all_z = []
    all_zq = []
    loader = tqdm(loader) if rank == 0 else loader
    total = 0
    for x, _ in loader:
        start_time = time.time()
        if args.image_size_eval != args.image_size:
            rgb_gts = F.interpolate(x, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        else:
            rgb_gts = x
        rgb_gts = (rgb_gts.permute(0, 2, 3, 1).to("cpu").numpy() + 1.0) / 2.0 # rgb_gt value is between [0, 1]
        x = x.to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            
            # TODO: change this for diffusion decoder
            # samples = vq_model(x)
            # start_time = time.time()
            quant, _, _ = vq_model.module.encode(x)
            samples = vq_model.module.decode(quant, num_steps=args.num_inference_steps, cfg_scale=args.cfg_scale, clip=args.clip)
            torch.cuda.synchronize()  # 等待 CUDA 操作完成
            end_time = time.time()
            elapsed = end_time - start_time
            total_gen_time += elapsed
            total_images += x.shape[0]
            batch_times.append(elapsed)

            if args.image_size_eval != args.image_size:
                samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        
        all_z.append(quant.float().detach().cpu().numpy())
            
        samples = torch.clamp(127.5 * samples + 128, 0, 255).to(dtype=torch.uint8)
        x = torch.clamp(127.5 * x + 128, 0, 255).to(dtype=torch.uint8)
    
        compute_fid_score.update(x, real=True)
        compute_fid_score.update(samples, real=False)
        
        samples = samples.permute(0, 2, 3, 1).to("cpu").numpy()

        # Save samples to disk as individual .png files
        for i, (sample, rgb_gt) in enumerate(zip(samples, rgb_gts)):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            # metric
            rgb_restored = sample.astype(np.float32) / 255. # rgb_restored value is between [0, 1]
            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)
            
        total += global_batch_size
    
    all_z = np.concatenate(all_z, axis=0)
    fid = compute_fid_score.compute().detach()
    
    # ------------------------------------
    #       Summary
    # ------------------------------------
    # Make sure all processes have finished saving their samples
    dist.barrier()
    world_size = dist.get_world_size()
    gather_psnr_val = [None for _ in range(world_size)]
    gather_ssim_val = [None for _ in range(world_size)]
    gather_all_z = [None for _ in range(world_size)]
    dist.all_gather_object(gather_psnr_val, psnr_val_rgb)
    dist.all_gather_object(gather_ssim_val, ssim_val_rgb)
    dist.all_gather_object(gather_all_z, all_z)
    

    # print(gather_latents)
    if rank == 0:
        throughput = total_images / total_gen_time
        avg_batch_time = np.mean(batch_times)
        print(f"Throughput: {throughput:.2f} images/s over {total_images} images")
        print(f"Average batch time: {avg_batch_time:.4f} s")

        # 写入日志文件
        with open(f"{sample_folder_dir}/throughput_log.txt", "w") as f:
            f.write(f"total_images: {total_images}\n")
            f.write(f"total_time: {total_gen_time:.4f}\n")
            f.write(f"throughput: {throughput:.4f} images/s\n")
            f.write(f"avg_batch_time: {avg_batch_time:.4f} s\n")

    if rank == 0:
        gather_psnr_val = list(itertools.chain(*gather_psnr_val))
        gather_ssim_val = list(itertools.chain(*gather_ssim_val))   
        # gather_fid_val = list(itertools.chain(*gather_fid_val))     
        psnr_val_rgb = sum(gather_psnr_val) / len(gather_psnr_val)
        ssim_val_rgb = sum(gather_ssim_val) / len(gather_ssim_val)
        gather_all_z = np.concatenate(gather_all_z, axis=0)
        # statistics 
        print(f"zq shape: {gather_all_z.shape}")
        print("zq mean: ", np.mean(gather_all_z))
        print("zq std: ", np.std(gather_all_z))
        save_latent_stat_dir = '/'.join(args.vq_ckpt.split('/')[:-2])
        np.savez(f'{save_latent_stat_dir}/latent_stat', mean=np.mean(gather_all_z), std=np.std(gather_all_z))
        # fid_val = sum(gather_fid_val) / len(gather_fid_val)
        
        print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
        print("FID: %f" % (fid))
        wandb_logger.log({
            'cfg': args.cfg_scale,
            'rFID': fid,
            'PSNR': psnr_val_rgb,
            'SSIM': ssim_val_rgb,
        })
        
        result_file = f"{sample_folder_dir}_results.txt"
        print("writing results to {}".format(result_file))
        with open(result_file, 'w') as f:
            print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb), file=f)
            print("FID: %f " % (fid), file=f)
          
        visualize_random_grids(sample_folder_dir, wandb_logger, grid_size=8, num_grids=10)  
        # create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
        print("Done.")

    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/work/hdd/bcey/hchen10/datasets/imagenet/ImageNet/val")
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco'], default='imagenet')
    parser.add_argument("--config", type=str, default="/work/hdd/bcey/hchen10/rule_tokenizer-main/configs/diff_in1k/exp017-aediff8_rfid.yaml")
    parser.add_argument("--vq-ckpt", type=str, default="/work/hdd/bcey/hchen10/rule_tokenizer-main/experiments/in1k/exp017-cfg2-aediff8-latent_128d32-enc_mmditd12-dec_mmditd12_ms4-cfg-cross_rope-sigmoidweight-lognormal-percepstart_2/ckpts_ADD_cfg2.0_p0.5_continue14k/0003200.pt")
    parser.add_argument("--vq-model", type=str, default="AE-Diff-16")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--sample-dir", type=str, default="reconstructions")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=1.0, help='classifier free guidance scale')
    parser.add_argument("--num-inference-steps", type=int, default=1, help='number of inference steps')
    parser.add_argument("--clip", action='store_true', default=False)
    args = parser.parse_args()
    main(args)