# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
from torch import nn
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import wandb
import ruamel.yaml as yaml
import numpy as np
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
from glob import glob
from copy import deepcopy

from timm.scheduler import create_scheduler_v2 as create_scheduler
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.logger_func import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.misc import str2bool, manage_checkpoints, load_model_state_dict
from utils.optim import param_groups_weight_decay
from utils.data import random_crop_arr, center_crop_arr
from modelling.tokenizer_ADD import VQ_models
from modelling.discriminators import DinoDiscriminator
from modelling.lpips import LPIPS, LPIPSTimm
from utils.diff_aug import DiffAugment
from losses.vq_loss import VQLoss
from torchmetrics.image.fid import FrechetInceptionDistance

import warnings
warnings.filterwarnings('ignore')


def calculate_adaptive_weight(nll_loss, g_loss, last_layer):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

    return d_weight.detach()
#################################################################################
#                                  Training Loop                                #
#################################################################################

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_disc_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def calculate_adaptive_weight(nll_loss, g_loss, last_layer):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

    return d_weight.detach()

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        if args.exp_index is not None:
            experiment_index = int(args.exp_index)
        else:
            experiment_index = int(args.config.split('/')[-1].split('-')[0][3:]) # len(glob(f"{args.results_dir}/*"))
        if args.config is not None:
            model_string_name = '.'.join(args.config.split('/')[-1].split('.')[:-1])
            if model_string_name.startswith('exp'):
                model_string_name = '-'.join(model_string_name.split('-')[1:])
        else:
            model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/exp{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/ckpts_ADD_cfg{args.cfg_scale}_p{args.p_loss_weight}"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        experiment_config = vars(args)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
        
        wandb_logger = wandb.init(project='diff_toknenizer_ADD', name=f'exp{experiment_index:03d}-{model_string_name}')
    else:
        logger = create_logger(None)
        wandb_logger = None

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model_t = VQ_models[args.vq_model](
        image_size=args.image_size,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        vq_loss_ratio=args.vq_loss_ratio,
        kl_loss_weight=args.kl_loss_weight,
        dropout_p=args.dropout_p,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        num_latent_tokens=args.num_latent_tokens,
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        enc_cls_token=args.encoder_cls_token,
        dec_cls_token=args.decoder_cls_token,
        tau=args.tau,
        repa=args.repa,
        repa_model=args.repa_model,
        repa_patch_size=args.repa_patch_size,
        repa_proj_dim=args.repa_proj_dim,
        repa_loss_weight=args.repa_loss_weight,
        repa_align=args.repa_align,
        num_codebooks=args.num_codebooks,
        # diffusion decoder
        # perceptual_type=args.perceptual_type,
        perceptual_type=args.perceptual_type,
        perceptual_loss_weight=args.perceptual_loss_weight,
        using_cfg=args.using_cfg,       
        latent_pos_encoding_type=args.latent_pos_embed_type, 
        pos_embed_max_size=args.pos_embed_max_size,
        multiscale_decoder=args.multiscale_decoder,
        num_multiscale_stages=args.num_multiscale_stages,
        sigmoid_weighting=args.sigmoid_weighting,
        flow_weighting=args.flow_weighting
    )
    vq_model_s = VQ_models[args.vq_model](
        image_size=args.image_size,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        vq_loss_ratio=args.vq_loss_ratio,
        kl_loss_weight=args.kl_loss_weight,
        dropout_p=args.dropout_p,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        num_latent_tokens=args.num_latent_tokens,
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        enc_cls_token=args.encoder_cls_token,
        dec_cls_token=args.decoder_cls_token,
        tau=args.tau,
        repa=args.repa,
        repa_model=args.repa_model,
        repa_patch_size=args.repa_patch_size,
        repa_proj_dim=args.repa_proj_dim,
        repa_loss_weight=args.repa_loss_weight,
        repa_align=args.repa_align,
        num_codebooks=args.num_codebooks,
        # diffusion decoder
        # perceptual_type=args.perceptual_type,
        perceptual_type=args.perceptual_type,
        perceptual_loss_weight=args.perceptual_loss_weight,
        using_cfg=args.using_cfg,       
        latent_pos_encoding_type=args.latent_pos_embed_type, 
        pos_embed_max_size=args.pos_embed_max_size,
        multiscale_decoder=args.multiscale_decoder,
        num_multiscale_stages=args.num_multiscale_stages,
        sigmoid_weighting=args.sigmoid_weighting,
        flow_weighting=args.flow_weighting
    )

    # Prepare models for training:
    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
        keys = vq_model_t.load_state_dict(load_model_state_dict(checkpoint['model']), strict=False)
        vq_model_s.load_state_dict(load_model_state_dict(checkpoint['model']), strict=False)
        print(keys)
    
    D = DinoDiscriminator()

    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model_s.parameters() if p.requires_grad):,}")
    logger.info(f"Discriminator trainable params (before freeze): {sum(p.numel() for p in D.parameters() if p.requires_grad):,}")
    if args.ema:
        ema = deepcopy(vq_model_s).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters() if p.requires_grad):,}")
    vq_model_s = vq_model_s.to(device)
    vq_model_t = vq_model_t.to(device)
    D = D.to(device)

    # freeze encoder
    if args.freeze_encoder:
        logger.info("freeze encoder...")
        for name, param in vq_model_s.named_parameters():
            if name.startswith('encoder'): #  or name.startswith('quant_conv'):
                param.requires_grad = False

    # scaling lr
    args.lr_student = args.lr_student * args.global_batch_size / 256
    args.lr_disc = args.lr_disc * args.global_batch_size / 256

    # scaling lr
    # args.lr = args.lr * args.global_batch_size / 256
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    if args.optimizer == 'adam':
        s_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vq_model_s.parameters()), lr=args.lr_student, betas=(args.beta1, args.beta2))
        D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr_disc, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamw':
        s_optimizer = torch.optim.AdamW(param_groups_weight_decay(vq_model_s, weight_decay=args.weight_decay), lr=args.lr_student, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        D_optimizer = torch.optim.AdamW(param_groups_weight_decay(D, weight_decay=args.weight_decay), lr=args.lr_disc, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # Setup data:
    if args.dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
        num_update_steps_per_epoch = len(loader)
        max_train_steps = args.epochs * num_update_steps_per_epoch
    elif args.dataset == 'azure://syn_data_wbs':
        from datasets.dataset_az import AzureWebDatasetPipeline
        dataset = AzureWebDatasetPipeline(
            prefix='vision_datasets/syn_data_wbs',
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            num_workers=args.num_workers,
            resolution=args.image_size,
            shuffle_samples=512,
            min_size=0,
            max_ar=None,
            max_pwatermark=None,
            center_crop=True,
        )
        loader = dataset.get_dataloader()
        num_update_steps_per_epoch = 5000
        max_train_steps = args.epochs * num_update_steps_per_epoch
    else:
        pass 

    if args.val_data_path is not None:
        transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        val_dataset = ImageFolder(args.val_data_path, transform=transform)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=rank % torch.cuda.device_count(),
            shuffle=False,
            seed=args.global_seed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
    else:
        val_loader = None

    # create lr scheduler
    if args.lr_scheduler == 'none':
        vqvae_lr_scheduler = None
        disc_lr_scheduler = None
    else:
        vqvae_lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=s_optimizer,
            patience_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs,
            warmup_epochs=args.lr_warmup_epochs,
            min_lr=args.lr_student * 0.1,
        ) 
        disc_lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=D_optimizer,
            patience_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs,
            warmup_epochs=args.lr_warmup_epochs,
            min_lr=args.lr_disc * 0.1,
        )
    logger.info(f"num_update_steps_per_epoch {num_update_steps_per_epoch:,} max_train_steps ({max_train_steps})")

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        
        keys = vq_model_s.load_state_dict(load_model_state_dict(checkpoint['model']), strict=False)
        print(keys)
        
        if args.ema:
            ema.load_state_dict(checkpoint["ema"], strict=False)
        
        if not args.finetune:
            s_optimizer.load_state_dict(checkpoint["optimizer"])
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            if args.dataset == 'imagenet':
                start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size)) + 1
                train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
            else:
                start_epoch = 0
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model_s, decay=0)  # Ensure EMA is initialized with synced weights
 
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model_s = torch.compile(vq_model_s, mode="reduce-overhead") # requires PyTorch 2.0      
        D = torch.compile(D)  
        logger.info("compiling done.")

    
    vq_model_s = DDP(vq_model_s.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vq_model_s.train()
    D = DDP(D.to(device), device_ids=[args.gpu], find_unused_parameters=False, broadcast_buffers=False)
    D.train()
    vq_model_t.eval()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    percept_loss = 0
    reconst_loss = 0
    discriminator_loss = 0
    start_time = time.time()
    curr_fid = None 
    compute_fid_score = FrechetInceptionDistance(normalize=False).cuda()

    reconstruction_loss=nn.MSELoss().to(device)
    lambd = 2.5
    perceptual_loss = LPIPS().eval()
    perceptual_loss = perceptual_loss.to(device)

    logger.info(f"Training for {args.epochs} epochs...")
    print("args.num_inference_steps: ", args.num_inference_steps)
    for epoch in range(start_epoch, args.epochs):
        if args.dataset in ['imagenet']:
            sampler.set_epoch(epoch)
            
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
                
            imgs = x.to(device, non_blocking=True)

            # generator training
            s_optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                # x_end_student_list, x_student, codebook_loss, info = vq_model_s.module.forward_student(imgs)
                # x_end_teacher_list, start_stage, u, codebook_loss, info = vq_model_t.forward_teacher(imgs, x_student)
                x_end_student_list, x_student, codebook_loss, info = vq_model_s.module.decode_student(imgs, num_steps=1, cfg_scale=args.cfg_scale)
                x_end_teacher_list, start_stage, u, codebook_loss, info = vq_model_t.decode_teacher(imgs, x_student, num_steps=1, cfg_scale=args.cfg_scale)
                # print("start_stage: ", start_stage)
                # print("u: ", u)
                # print("x_end_student_list: ", len(x_end_student_list))
                # print("x_end_teacher_list: ", len(x_end_teacher_list))
                logits_fake = D(x_student.contiguous()) 
                generator_adv_loss = -torch.mean(logits_fake)

                r_loss = torch.tensor(0.0, device=device)
                # c = (1.0 / ((1-u) + 1)).to(device)
                c = (1.0 / (u + 1)).to(device)
                # c = 1.0
                student_aligned = x_end_student_list[start_stage:]  # 保留从 start_stage 开始的 student 输出
                teacher_aligned = x_end_teacher_list
                # print("Student aligned shapes:")
                # for i, x in enumerate(student_aligned):
                #     print(f"student_aligned  [{i}] shape: {x.shape}")

                # print("Teacher aligned shapes:")
                # for i, x in enumerate(teacher_aligned):
                #     print(f"teacher_aligned  [{i}] shape: {x.shape}")
                for student_imgs, teacher_imgs in zip(student_aligned, teacher_aligned):
                    r_loss += reconstruction_loss(student_imgs, teacher_imgs)

                p_loss = perceptual_loss(x_student.contiguous(), imgs.contiguous())
                p_loss = torch.mean(p_loss)

                null_loss = 1.0 * r_loss + args.p_loss_weight * p_loss
                disc_adaptive_weight = calculate_adaptive_weight(nll_loss=null_loss, g_loss=generator_adv_loss, last_layer=vq_model_s.module.decoder.final_layer.get_last_layer_weight())
                disc_weight = adopt_weight(1, train_steps+1, threshold=args.disc_start)
                S_loss = disc_adaptive_weight * disc_weight * generator_adv_loss + lambd * (r_loss * c + args.p_loss_weight * p_loss)

            S_loss = S_loss.mean()
            scaler.scale(S_loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(s_optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model_s.parameters(), args.max_grad_norm)
            scaler.step(s_optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, vq_model_s.module._orig_mod if args.compile else vq_model_s.module)

            # Train Discriminator
            D_optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
            # with torch.cuda.amp.autocast(dtype=ptdtype):
                real_pred = D(imgs)
                logits_real = D(imgs.contiguous().detach())
                logits_fake = D(x_student.contiguous().detach())
                disc_weight = adopt_weight(0.5, train_steps+1, threshold=args.disc_start)
                d_adversarial_loss = disc_weight * hinge_disc_loss(logits_real, logits_fake)
                # print('d_adversarial_loss: ', d_adversarial_loss)

                if args.disc_cr_loss_weight:
                    logits_real_s = D(DiffAugment(imgs.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                    logits_fake_s = D(DiffAugment(x_student.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                    disc_cr_loss_weight = args.disc_cr_loss_weight if (train_steps+1) >= args.disc_start else 0.0
                    d_cr = F.mse_loss(torch.cat([logits_real, logits_fake], dim=0), torch.cat([logits_real_s, logits_fake_s])) * disc_cr_loss_weight
                    d_adversarial_loss += d_cr
                    # print('d_adversarial_loss: ', d_adversarial_loss)

            scaler_disc.scale(d_adversarial_loss).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(D_optimizer)
                torch.nn.utils.clip_grad_norm_(D.parameters(), args.max_grad_norm)
            scaler_disc.step(D_optimizer)
            scaler_disc.update()
            discriminator_loss += d_adversarial_loss.item()


            # # Log loss values:
            running_loss += S_loss.item()
            percept_loss += p_loss.item()
            reconst_loss += r_loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_p_loss = torch.tensor(percept_loss / log_steps, device=device)
                avg_r_loss = torch.tensor(reconst_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_p_loss, op=dist.ReduceOp.SUM)
                avg_p_loss = avg_p_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_r_loss, op=dist.ReduceOp.SUM)
                avg_r_loss = avg_r_loss.item() / dist.get_world_size()
                # discriminator loss history
                avg_discriminator_loss = torch.tensor(discriminator_loss / log_steps, device=device)
                dist.all_reduce(avg_discriminator_loss, op=dist.ReduceOp.SUM)
                avg_discriminator_loss = avg_discriminator_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/total_steps:{max_train_steps:07d}) Train Loss: {avg_loss:.4f}, Reconstruction Loss: {avg_r_loss:.4f}, Perceptual Loss: {avg_p_loss:.4f}, Discriminator Loss: {avg_discriminator_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                percept_loss = 0
                reconst_loss = 0
                discriminator_loss = 0
                log_steps = 0
                start_time = time.time()
            
                if rank == 0 and wandb_logger is not None:
                    log_dict = {"lr": s_optimizer.param_groups[0]["lr"], "train_loss": avg_loss, "reconstruction_loss": avg_r_loss, "perceptual_loss": avg_p_loss, "discriminator_loss": avg_discriminator_loss}
                    if args.vq_model == 'VQ-Diff-16':
                        log_dict['vq/vq_loss'] = codebook_loss[3].item()
                        log_dict['vq/commit_loss'] = codebook_loss[4].item()
                        log_dict['vq/ent_loss'] = codebook_loss[5].item()
                        log_dict['vq/usage'] = codebook_loss[6]
                    wandb_logger.log(log_dict,
                        step=train_steps
                    )
                
            if train_steps % args.vis_every == 0:
                # evaluate
                vq_model_s.eval()
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=ptdtype):
                    input_images = imgs[:8]
                    quant, _, _ = vq_model_s.module.encode(input_images)
                    recons_imgs = vq_model_s.module.decode(quant, num_steps=args.num_inference_steps, cfg_scale=args.cfg_scale, guidance_high=1.0)
                image = torch.cat([input_images, recons_imgs], dim=0)
                image = torch.clamp(image, min=-1, max=1)
                image = make_grid((image + 1) / 2, nrow=4, padding=0, pad_value=1.0)
                image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
                image = Image.fromarray(image.astype(np.uint8))
                vq_model_s.train()

                if rank == 0:
                    wandb_logger.log({"recon_images": [wandb.Image(image)]}, step=train_steps)

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:

                if rank == 0:
                    if args.compile:
                        model_weight = vq_model_s.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model_s.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": s_optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    # if not args.no_local_save:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    manage_checkpoints(checkpoint_dir)
                dist.barrier()

                if args.val_data_path is not None:
                    vq_model_s.eval()
                    total = 0
                    for x, _ in tqdm(val_loader, desc=f'evaluation for step {train_steps:07d}', disable=not rank == 0):

                        with torch.no_grad(), torch.cuda.amp.autocast(dtype=ptdtype):
                            x = x.to(device, non_blocking=True)
                            quant, _, _ = vq_model_s.module.encode(x)
                            recon_x = vq_model_s.module.decode(quant, num_steps=args.num_inference_steps, cfg_scale=args.cfg_scale, guidance_high=1.0)
                            
                        x = torch.clamp(127.5 * x + 128.0, 0, 255).to(dtype=torch.uint8)
                        recon_x = torch.clamp(127.5 * recon_x + 128.0, 0, 255).to(dtype=torch.uint8)

                        compute_fid_score.update(x, real=True)
                        compute_fid_score.update(recon_x, real=False)

                        total += recon_x.shape[0] * torch.cuda.device_count()
                    vq_model_s.train()
                    logger.info(f"Ealuate total {total} files.")
                    
                    FID = compute_fid_score.compute().detach()
                
                    logger.info(f"traing step: {train_steps:07d}, FID {FID:07f}")
                    # eval code, delete prev if not the best
                    if curr_fid == None:
                        curr_fid = [FID, train_steps]
                    elif FID <= curr_fid[0]:
                        curr_fid = [FID, train_steps]
                        
                    if rank == 0 and wandb_logger is not None:
                        wandb_logger.log({"rFID": FID}, step=train_steps)
                        
                    compute_fid_score.reset()
                

            if vqvae_lr_scheduler is not None:
                vqvae_lr_scheduler.step_update(train_steps)


    vq_model_s.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/tokenizer/cnn_llamagen_vq16.yaml', help="config file used to specify parameters")
    
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    
    
    parser.add_argument("--data-path", type=str, default="ImageNet2012/train")
    
    parser.add_argument("--val-data-path", type=str, default=None)
    parser.add_argument("--cloud-save-path", type=str, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", type=str2bool, default=False, help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", type=str2bool, default=False, help="finetune a pre-trained vq model")
    parser.add_argument("--ema", type=str2bool, default=True, help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", type=str2bool, default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--vq-loss-ratio", type=float, default=1.0, help="vq loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--kl-loss-weight", type=float, default=0.000001)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num-codebooks", type=int, default=1)
    
    parser.add_argument("--perceptual-loss-weight", type=float, default=0.1, help="perceptual loss weight of LPIPS")
    parser.add_argument("--perceptual-type", type=str, default='vgg', help="perceptual loss type of LPIPS", choices=['vgg', 'timm', 'tv', 'dinov2'])
    parser.add_argument("--perceptual-model", type=str, default='vgg', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-dino-variants", type=str, default='depth12_no_train', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-intermediate-loss", type=str2bool, default=False, help="perceptual loss compute at intermedia features of LPIPS")
    parser.add_argument("--perceptual-logit-loss", type=str2bool, default=False, help="perceptual loss compute at logits of LPIPS")
    parser.add_argument("--perceptual-resize", type=str2bool, default=False, help="perceptual loss compute at resized images of LPIPS")
    parser.add_argument("--perceptual-warmup", type=int, default=None, help="iteration to warmup perceptual loss")
    parser.add_argument("--p_loss_weight", type=float, default=0.5)

    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=0, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-dim", type=int, default=64, help="discriminator channel base dimension")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan', 'maskbit', 'dino'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--lecam-loss-weight", type=float, default=None)
    parser.add_argument("--use-diff-aug",type=str2bool, default=False)
    parser.add_argument("--disc-cr-loss-weight", type=float, default=0.0, help="discriminator consistency loss weight for gan training")
    parser.add_argument("--disc-adaptive-weight",type=str2bool, default=True)
    
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr_student", type=float, default=1e-4)
    parser.add_argument("--lr_disc", type=float, default=5e-5) # 5e-5
    parser.add_argument("--lr_warmup_epochs", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default='none')
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    parser.add_argument("--enc-type", type=str, default="cnn")
    parser.add_argument("--dec-type", type=str, default="cnn")
    parser.add_argument("--num-latent-tokens", type=int, default=None)
    parser.add_argument("--encoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='encoder model name')
    parser.add_argument("--decoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='decoder model name')
    parser.add_argument("--encoder-tuning-method", type=str, default='full', help='tuning method for encoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--decoder-tuning-method", type=str, default='full', help='tuning method for decoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--encoder-pretrained", type=str2bool, default=True, help='load pre-trained weight for encoder')
    parser.add_argument("--decoder-pretrained", type=str2bool, default=False, help='load pre-trained weight for decoder')
    parser.add_argument("--encoder-patch-size", type=int, default=16, help='encoder patch size')
    parser.add_argument("--decoder-patch-size", type=int, default=16, help='decoder patch size')
    parser.add_argument("--encoder-cls-token", type=str2bool, default=True, help='encoder class token')
    parser.add_argument("--decoder-cls-token", type=str2bool, default=True, help='decoder class token')
    parser.add_argument("--freeze-encoder", type=str2bool, default=True, help='freeze encoder')
    
    # repa
    parser.add_argument("--repa", type=str2bool, default=False, help='use repa')
    parser.add_argument('--repa-model', type=str, default='vit_base_patch16_224', help='repa model name')
    parser.add_argument('--repa-patch-size', type=int, default=16, help='repa patch size')
    parser.add_argument('--repa-proj-dim', type=int, default=1024, help='repa embed dim')
    parser.add_argument('--repa-loss-weight', type=float, default=0.1, help='repa loss weight')
    parser.add_argument('--repa-align', type=str, default='global', help='align repa feature', choices=['global', 'avg_1d', 'avg_2d', 'avg_1d_shuffle'])
    
    # diffusion decoder
    parser.add_argument("--using-cfg", type=str2bool, default=False, help='use classifier free guidance')
    parser.add_argument("--cfg-scale", type=float, default=2.0, help='classifier free guidance scale')
    parser.add_argument("--num_inference_steps", type=int, default=50, help='number of inference steps')
    parser.add_argument("--latent-pos-embed-type", type=str, default='none', help='latent token position embedding type')
    parser.add_argument("--pos_embed_max_size", type=int, default=None)
    parser.add_argument("--multiscale_decoder", type=str2bool, default=False, help='multiscale diffusion decoder')
    parser.add_argument("--num_multiscale_stages", type=int, default=None, help='number of multiscale stages')
    parser.add_argument("--multiscale_perceptual_start", type=int, default=0, help='percetual loss start stage')
    parser.add_argument("--sigmoid_weighting", type=str2bool, default=False, help='sigmoid weighting of noise')
    parser.add_argument("--flow_weighting", type=str, default='uniform', help='flow weighting of time')
    
    parser.add_argument("--model_checkpoint", type=str, default='/work/hdd/bcey/hchen10/rule_tokenizer-main/experiments/in1k/exp015-aediff16-latent_128d32-enc_mmditd12-dec_mmditd12_ms3-cfg-cross_rope--sigmoidweight-lognormal-percepstart_2/checkpoints_ADD_cfg1_p0.1/0004700.pt', help='load model from checkpoint')
    
    #fFirst parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    main(args)
