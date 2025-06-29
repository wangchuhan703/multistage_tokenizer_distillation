import math
import torch
import numpy as np
import torch.nn.functional as F
from modelling.lpips import LPIPS, LPIPSTimm

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

def logsnr_linear(t):
    # t should be a tensor in (0, 1)
    return 2 * torch.log((1 - t) / t)

def dlogsnr_dt_linear(t):
    # Assumes t is a tensor in (0, 1)
    return -2 * (1 / (1 - t) + 1 / t)



class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            perceptual_type='vgg_lpips',
            perceptual_loss_weight=0.0,
            sigmoid_weighting=False,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.sigmoid_weighting = sigmoid_weighting

        self.perceptual_loss_weight = perceptual_loss_weight
        if self.perceptual_loss_weight > 0:
            if perceptual_type == 'dinov2':
                self.lpips_loss = LPIPSTimm('vit_small_patch14_dinov2.lvd142m', intermediate_loss=True, resize=True, logit_loss=False, eval=True, dino_variants='depth5').eval().cuda()
            elif perceptual_type == 'resnet50':
                self.lpips_loss = LPIPSTimm('resnet50', intermediate_loss=True, resize=True, logit_loss=True, eval=False).eval().cuda()
            else:
                self.lpips_loss = LPIPS().eval().cuda()
            
    
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
        # sample timesteps
        if self.weighting == "uniform":
            if images.ndim == 4:
                time_input = torch.rand((images.shape[0], 1, 1, 1))
            else:
                time_input = torch.rand((images.shape[0], 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            if images.ndim == 4:
                rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            else:
                rnd_normal = torch.randn((images.shape[0], 1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        
        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        if self.sigmoid_weighting:
            bias = -3
            if self.path_type == 'linear':
                logsnr = logsnr_linear(time_input)
                dlogsnr_dt = dlogsnr_dt_linear(time_input)
                weight = -0.5 * dlogsnr_dt * math.exp(bias) * torch.sigmoid(logsnr - bias)
            denoising_loss = denoising_loss * weight
        denoising_loss = denoising_loss.mean()

        if len(zs) == 0:
            proj_loss = torch.tensor(0., device=images.device, dtype=images.dtype)
        else:
            # projection loss
            proj_loss = 0.
            bsz = zs[0].shape[0]
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):         
                    
                    # avg 1d pool
                    # z_j = F.adaptive_avg_pool1d(z_j.permute(1, 0), z_tilde_j.shape[0]).permute(1, 0)
                    
                    # global pool
                    # z_j = z_j.mean(dim=0, keepdim=True)
                    # z_tilde_j = z_tilde_j.mean(dim=0, keepdim=True)
                    
                    # replicate laten tokens
                    l, d = z_tilde_j.shape
                    z_tilde_j = z_tilde_j.unsqueeze(1).expand(-1, z_j.size(0) // l, -1).reshape(-1, d)
                        
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)
        
        if self.perceptual_loss_weight > 0:
            images_pred = model_input - model_output * time_input
            perceptual_loss = self.lpips_loss(images, images_pred).mean()
            perceptual_loss = perceptual_loss 
        else:
            perceptual_loss = torch.tensor(0., device=images.device, dtype=images.dtype)

        total_loss = denoising_loss + perceptual_loss * self.perceptual_loss_weight + proj_loss

        return total_loss, denoising_loss, perceptual_loss, proj_loss
