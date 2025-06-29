import math
import torch
import numpy as np
import torch.nn.functional as F
from modelling.lpips import LPIPS
from torchvision import transforms
import torchvision.transforms.functional as TF

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def cal_rectify_ratio(start_t, gamma):
    return 1 / (math.sqrt(1 + (1 / gamma)) * (1 - start_t) + start_t)


def logsnr_linear(t):
    # t should be a tensor in (0, 1)
    return 2 * torch.log((1 - t) / t)

def dlogsnr_dt_linear(t):
    # Assumes t is a tensor in (0, 1)
    return -2 * (1 / (1 - t) + 1 / t)

class MultiScaleSILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            num_stages=3,
            gamma=1/3,
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            perceptual_type='vgg_lpips',
            perceptual_loss_weight=0.0,
            perceptual_start_stage=0,
            sigmoid_weighting=False
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.num_stages = num_stages
        self.gamma = gamma
        self.perceptual_start_stage = perceptual_start_stage
        self.sigmoid_weighting = sigmoid_weighting

        self.perceptual_loss_weight = perceptual_loss_weight
        if self.perceptual_loss_weight > 0:
            self.lpips_loss = LPIPS().eval().cuda()
            
        # discrete timesteps and t
        num_train_timesteps = 1000
        self.num_train_timesteps = num_train_timesteps
        shift = 1.0
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        t = timesteps / num_train_timesteps
        t = shift * t / (1 + (shift - 1) * t)
        self.timesteps = t * num_train_timesteps
        self.t = t.to("cpu")

        
        # multi scales
        self.start_t, self.end_t = dict(), dict()
        self.orig_start_t = dict()
        stage_distance = list()
        
        # uniform stage range
        stage_range = 1 / self.num_stages
        for i_s in range(num_stages):
            start_indice = int(stage_range * i_s  * num_train_timesteps)
            start_indice = max(start_indice, 0)
            end_indice = int(stage_range * (i_s + 1) * num_train_timesteps)
            end_indice = min(end_indice, num_train_timesteps)
            start_t = self.t[start_indice].item()
            end_t = self.t[end_indice].item() if end_indice < num_train_timesteps else 0.0
            self.orig_start_t[i_s] = start_t 
            
            if i_s != 0:
                start_t = 1.0 - cal_rectify_ratio(1 - start_t, gamma) * (1 - start_t)
            
            stage_distance.append(start_t - end_t)
            self.start_t[i_s] = start_t
            self.end_t[i_s] = end_t 
        
        # print("start_t", self.start_t)
        # print("end_t", self.end_t)
        # print("orig_start_t", self.orig_start_t)

        self.timestep_ratios = dict()
        # determine the ratio of each stage according to flow length
        tot_distance = sum(stage_distance)
        for i_s in range(num_stages):
            if i_s == 0:
                start_ratio = 0.0
            else:
                start_ratio = sum(stage_distance[:i_s]) / tot_distance
                
            if i_s == num_stages - 1:
                end_ratio = 1.0
            else:
                end_ratio = sum(stage_distance[:i_s+1]) / tot_distance

            self.timestep_ratios[i_s] = (start_ratio, end_ratio)
        
        # print("timestep_ratios", self.timestep_ratios)
                
        self.timesteps_per_stage = dict()
        self.t_per_stage = dict()
        # determine the timesteps and sigmas for each stage
        for i_s in range(num_stages):
            timestep_ratio = self.timestep_ratios[i_s]
            timestep_max = self.timesteps[int(timestep_ratio[0] * num_train_timesteps)]
            timestep_min = self.timesteps[min(int(timestep_ratio[1] * num_train_timesteps), num_train_timesteps - 1)]
            timesteps = np.linspace(
                timestep_max, timestep_min, num_train_timesteps + 1,
            )
            self.timesteps_per_stage[i_s] = timesteps[:-1] if isinstance(timesteps, torch.Tensor) else torch.from_numpy(timesteps[:-1])
            stage_sigmas = np.linspace(
                1, 0, num_train_timesteps + 1,
            )
            self.t_per_stage[i_s] = torch.from_numpy(stage_sigmas[:-1])
        

    
    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t


    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        
        bsz, c, h, w = images.shape
        dtype = images.dtype
        device = images.device
        
        total_loss = 0.
        total_denoise_loss = 0.
        total_proj_loss = 0.
        total_percep_loss = 0.

        # generate random stage assignment per sample
        stage_indices = list(range(self.num_stages)) * (bsz // self.num_stages + 1)
        stage_indices = stage_indices[:bsz]
        np.random.shuffle(stage_indices)
        stage_indices = torch.tensor(stage_indices, device=device)
        
        # iterative through each stage, from low res to high res
        for i_s in range(self.num_stages):
            
            selected = (stage_indices == i_s)
            if selected.sum() == 0:
                continue
            
            imgs_sel = images[selected]
            bsz_sel = imgs_sel.shape[0]
            start_t = self.start_t[i_s]
            end_t = self.end_t[i_s]
            
            x_end = imgs_sel
            # print(f"x_end start {x_end.shape}")
            for d in range(1, self.num_stages - i_s):
                # print(f"x_end down {d}")
                x_end = F.interpolate(x_end, size=(h // (2 ** d), w // (2 ** d)), mode="bilinear")
            # print(f"x_end {x_end.shape}")
            
            # downsample to start, then upsample
            x_start = imgs_sel
            # print(f"x_start start {x_start.shape}")
            for d in range(1, self.num_stages - i_s + 1):
                # print(f"x_start down {d}")
                x_start = F.interpolate(x_start, size=(h // (2 ** d), w // (2 ** d)), mode="bilinear")
            x_start = F.interpolate(x_start, size=x_end.shape[-2:], mode="nearest")
            # print(f"x_start {x_start.shape}")
            
            # mix with noise
            noises = torch.randn_like(x_start)
            x_end_noisy = (1 - end_t) * x_end + end_t * noises
            x_start_noisy = (1 - start_t) * x_start + start_t * noises
            
            # setup input and target
            if self.weighting == 'uniform':
                u = torch.rand(size=(bsz_sel,), device="cpu")
            elif self.weighting == 'lognormal':
                rnd_normal = torch.randn((bsz_sel, ))
                sigma = rnd_normal.exp()
                u = sigma / (1 + sigma)
            # TODO: change to continuous time sampling
            indices = (u * self.num_train_timesteps).long()   # Totally 1000 training steps per stage
            indices = indices.clamp(0, self.num_train_timesteps-1)
            timesteps = self.timesteps_per_stage[i_s][indices].to(device=device, dtype=dtype) / self.num_train_timesteps
            t = self.t_per_stage[i_s][indices].to(device=device, dtype=dtype).view(-1, 1, 1, 1)
            # print("timesteps", timesteps)
            # print("timesteps / 1000", timesteps / 1000) 
            # print("t", t)
            
            xt = (1 - t) * x_end_noisy + t * x_start_noisy
            target = x_start_noisy - x_end_noisy
            
            # model cal
            stage_model_kwargs = {k:v[selected] if v is not None else None for k,v in model_kwargs.items()}
            model_out, zs_tilde = model(xt, timesteps.view(bsz_sel), **stage_model_kwargs)
            denoise_loss = mean_flat((model_out - target) ** 2)
            if self.sigmoid_weighting:
                bias = -3
                if self.path_type == 'linear':
                    logsnr = logsnr_linear(t)
                    dlogsnr_dt = dlogsnr_dt_linear(t)
                    weight = -0.5 * dlogsnr_dt * math.exp(bias) * torch.sigmoid(logsnr - bias)
                denoise_loss = denoise_loss * weight.flatten()
            denoise_loss = denoise_loss.mean()
            
            # Projection loss
            proj_loss = torch.tensor(0., device=device, dtype=images.dtype)
            if zs:
                for z, zt in zip(zs, zs_tilde):
                    for z_j, zt_j in zip(z, zt):
                        l, d = zt_j.shape
                        zt_j = zt_j.unsqueeze(1).expand(-1, z_j.size(0) // l, -1).reshape(-1, d)
                        z_j = F.normalize(z_j, dim=-1)
                        zt_j = F.normalize(zt_j, dim=-1)
                        proj_loss += mean_flat(-(z_j * zt_j).sum(dim=-1))
                proj_loss /= (len(zs) * bsz_sel)

            # Perceptual loss
            if self.perceptual_loss_weight > 0 and i_s == self.num_stages - 1: # starts last
                images_pred = xt - model_out * (t - end_t)
                
                # crop
                if images.size(-1) >= 512:
                    rec_patch = []
                    x_patch = []
                    for _ in range(4):
                        c_top, c_left, _, _ = transforms.RandomCrop.get_params(images_pred, output_size=(256, 256))
                        rec_patch.append(TF.crop(images_pred, c_top, c_left, 256, 256))
                        x_patch.append(TF.crop(x_end_noisy.detach(), c_top, c_left, 256, 256))
                    pred = torch.cat(rec_patch, dim=0)
                    gt = torch.cat(x_patch, dim=0)
                else:
                    pred = images_pred
                    gt = x_end_noisy
                
                # perceptual_loss = self.lpips_loss(F.interpolate(x_end_noisy, size=images.shape[-2:], mode="bilinear").detach(), F.interpolate(images_pred, size=images.shape[-2:], mode="bilinear")).mean()
                perceptual_loss = self.lpips_loss(gt, pred).mean()
            else:
                perceptual_loss = torch.tensor(0., device=device, dtype=images.dtype)

            total_denoise_loss += denoise_loss * bsz_sel
            total_proj_loss += proj_loss * bsz_sel
            total_percep_loss += perceptual_loss * bsz_sel
        
        total_denoise_loss /= bsz
        total_proj_loss /= bsz
        total_percep_loss /= bsz
            
        total_loss = total_denoise_loss + total_percep_loss * self.perceptual_loss_weight + total_proj_loss
        

        return total_loss, total_denoise_loss, total_percep_loss, total_proj_loss

    # ------------------------------------------------------------------------
    # batch operation
    # def __call__(self, model, images, model_kwargs=None, zs=None):
    #     if model_kwargs is None:
    #         model_kwargs = {}

    #     bsz, _, H, W = images.shape
    #     dtype, device = images.dtype, images.device

    #     # ---- prepare per‑sample stage assignment ---------------------------
    #     stage_idx_full = torch.randperm(bsz) % self.num_stages
    #     #  The tensor above chooses a stage (0 … num_stages‑1) for every sample.

    #     # lists that accumulate *per stage* tensors so that we can call the
    #     # model exactly once.
    #     xt_list, target_list, t_list, timestep_list = [], [], [], []

    #     # Gather slices of model‑kwargs matching the new ordering
    #     concat_kwargs = {k: [] for k in model_kwargs}

    #     for i_s in range(self.num_stages):
    #         sel = stage_idx_full == i_s
    #         if not sel.any():
    #             continue
    #         sel_idx = sel.nonzero(as_tuple=True)[0]
    #         imgs_sel = images[sel]
    #         B_sel = imgs_sel.size(0)

    #         # --------------------------------------------------------------
    #         # Down/up‑sampling to obtain x_end, x_start (same as original)
    #         # --------------------------------------------------------------
    #         x_end = imgs_sel
    #         for d in range(1, self.num_stages - i_s):
    #             x_end = F.interpolate(x_end, size=(H // 2**d, W // 2**d), mode="bilinear")
    #         # print(f"stage {i_s}, x_end {x_end.shape}")

    #         x_start = imgs_sel
    #         for d in range(1, self.num_stages - i_s + 1):
    #             x_start = F.interpolate(x_start, size=(H // 2**d, W // 2**d), mode="bilinear")
    #         x_start = F.interpolate(x_start, size=x_end.shape[-2:], mode="nearest")
    #         # print(f"stage {i_s}, x_start {x_start.shape}")
            
    #         # noise injection
    #         noises = torch.randn_like(x_start)
    #         x_end_noisy = (1 - self.end_t[i_s]) * x_end + self.end_t[i_s] * noises
    #         x_start_noisy = (1 - self.start_t[i_s]) * x_start + self.start_t[i_s] * noises

    #         # time‑step sampling (uniform or log‑normal)
    #         if self.weighting == "uniform":
    #             u = torch.rand(B_sel)
    #         else:  # lognormal
    #             rnd_normal = torch.randn(B_sel)
    #             sigma = rnd_normal.exp()
    #             u = sigma / (1 + sigma)

    #         idx = (u * self.num_train_timesteps).long().clamp_(0, self.num_train_timesteps - 1)
    #         timesteps = self.timesteps_per_stage[i_s][idx].to(device=device, dtype=dtype)
    #         t = self.t_per_stage[i_s][idx].to(device=device, dtype=dtype).view(-1, 1, 1, 1)

    #         xt = (1 - t) * x_end_noisy + t * x_start_noisy
    #         target = x_start_noisy - x_end_noisy

    #         # record
    #         xt_list.append(xt)
    #         target_list.append(target)
    #         t_list.append(timesteps)
    #         timestep_list.append(t)  # needed later for perceptual / weight

    #         # gather kwargs slices
    #         for k, v in model_kwargs.items():
    #             concat_kwargs[k].append(v[sel] if v is not None else None)

    #     # final concatenation -----------------------------
    #     x_in = xt_list     # keep as list (variable spatial shapes)
    #     t_in = t_list
    #     for k in concat_kwargs:
    #         if concat_kwargs[k] and concat_kwargs[k][0] is not None:
    #             concat_kwargs[k] = torch.cat(concat_kwargs[k], dim=0)
    #         else:
    #             concat_kwargs[k] = None

    #     # --------------- single forward -------------------
    #     model_out_list, zs_tilde = model(x_in, t_in, **concat_kwargs)

    #     # --------------- compute losses -------------------
    #     # model returns list aligned with x_in order, so iterate zip‑wise
    #     total_denoise_loss = images.new_zeros(())
    #     total_proj_loss = images.new_zeros(())
    #     total_percep_loss = images.new_zeros(())

    #     for xt, target, t_scalar, model_out in zip(xt_list, target_list, timestep_list, model_out_list):
    #         denoise_loss = mean_flat((model_out - target) ** 2)

    #         if self.sigmoid_weighting:
    #             bias = -3
    #             if self.path_type == "linear":
    #                 logsnr = logsnr_linear(t_scalar)
    #                 dlogsnr_dt = dlogsnr_dt_linear(t_scalar)
    #                 weight = -0.5 * dlogsnr_dt * math.exp(bias) * torch.sigmoid(logsnr - bias)
    #                 denoise_loss = denoise_loss * weight.flatten()
    #         total_denoise_loss += denoise_loss.sum()

    #     # Projection loss (NOTE: not implemented yet)
    #     proj_loss = torch.tensor(0., device=device, dtype=images.dtype)
        
    #     # Perceptual loss (directly implement for last stage only)        
    #     if self.perceptual_loss_weight > 0:
    #         images_pred = xt - model_out * t # (t - end_t)
    #         # TODO: crop
    #         # print(f"percep images_pred {images_pred.shape}")
    #         # print(f"percep x_end_noisy {x_end_noisy.shape}")
            
    #         # perceptual_loss = self.lpips_loss(F.interpolate(x_end_noisy, size=images.shape[-2:], mode="bilinear").detach(), F.interpolate(images_pred, size=images.shape[-2:], mode="bilinear")).mean()
    #         perceptual_loss = self.lpips_loss(x_end_noisy.detach(), images_pred).mean()
    #     else:
    #         perceptual_loss = torch.tensor(0., device=device, dtype=images.dtype)

    #     # normalise by batch size
    #     total_denoise_loss /= images.size(0)
    #     total_percep_loss = perceptual_loss
    #     total_proj_loss = proj_loss
        
    #     total_loss = total_denoise_loss + total_percep_loss * self.perceptual_loss_weight + total_proj_loss
    #     return total_loss, total_denoise_loss, total_percep_loss, total_proj_loss


if __name__ == '__main__':
    
    sit_loss = MultiScaleSILoss()