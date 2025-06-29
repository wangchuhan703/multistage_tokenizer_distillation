from math import pi

import torch
from torch import nn

from einops import rearrange, repeat
import numpy as np

from functools import partial

from math import pi

import torch
from torch import nn

from einops import rearrange, repeat
import numpy as np


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)



# def rotate_half(x):
#     x = rearrange(x, '... (d r) -> ... d r', r = 2)
#     x1, x2 = x.unbind(dim = -1)
#     x = torch.stack((-x2, x1), dim = -1)
#     return rearrange(x, '... d r -> ... (d r)')



class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)



class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        clusters=4,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.pt_seq_len=pt_seq_len
        self.register_buffer("freqs", freqs)
        ft_seq_len = self.pt_seq_len#int(np.sqrt(x.shape[2]))
        t = torch.arange(ft_seq_len) / ft_seq_len * self.pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])# N C
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])# N C
        N,C = freqs_cos.shape
        H=W=int(np.sqrt(N))
        freqs_cos=freqs_cos.reshape(2, H//2, 2, W//2,C)
        freqs_sin=freqs_sin.reshape(2, H//2, 2, W//2,C)
        freqs_cos = torch.einsum('hpwqc->hwpqc', freqs_cos).reshape(N, C)
        freqs_sin = torch.einsum('hpwqc->hwpqc', freqs_sin).reshape(N, C)
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)
        self.clusters=1
        self.seq_len=256

    def forward(self, x,scale_index=None): 
        if scale_index is None:
            return  x * self.freqs_cos + rotate_half(x) * self.freqs_sin
        else:
            return x * self.freqs_cos[(scale_index+1)*self.seq_len//self.clusters-x.shape[2]:(scale_index+1)*self.seq_len//self.clusters] + rotate_half(x) * self.freqs_sin[(scale_index+1)*self.seq_len//self.clusters-x.shape[2]:(scale_index+1)*self.seq_len//self.clusters]


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
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

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

    def __call__(self, model, images, condition):
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1))
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output  = model(model_input, time_input.flatten(), condition)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        return denoising_loss

def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur

@torch.no_grad()
def euler_sampler(
        model,
        latents,
        y,
        scale_index,
        condition=None,
        num_steps=50,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        mask=None
        ):
    # setup conditioning
    _dtype = latents.dtype    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device
    cfg = cfg_scale
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            #cfg_scale = (cfg-1.)*i/num_steps+1
            x_cur = x_next
            if condition is not None and i==0:
                model_input = torch.cat([condition, x_next], dim=1)
            else:
                model_input = x_next
            if t_cur <= guidance_high or i==0:
                model_input = torch.cat([model_input, model_input], dim=0)
                y_cur = torch.cat([y, torch.ones_like(y).cuda()*1000], dim=0)
            else:
                y_cur = y
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            if i==0:
                d_cur = model(
                model_input.to(dtype=_dtype), y_cur, 
                torch.cat([torch.zeros(model_input.size(0)*(1 if scale_index else 0)).to(device=device, dtype=torch.float64), time_input.to(dtype=_dtype)], dim=0), 
                True, scale_index, mask).to(torch.float64)
            else:
                d_cur = model(
                model_input.to(dtype=_dtype), y_cur, 
                time_input.to(dtype=_dtype), 
                False, scale_index).to(torch.float64)
            if t_cur <= guidance_high or i ==0:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)  
                else:
                    d_cur = d_cur_cond    
            x_next = x_cur + (t_next - t_cur) * d_cur          
    return x_next

@torch.no_grad()
def euler_maruyama_sampler(
        model,
        latents,
        y,
        scale_index,
        condition=None,
        num_steps=50,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        mask=None,
        seq_len=128,
        clusters=4
        ):
    # setup conditioning

    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next

            if condition is not None and i==0:
                model_input = torch.cat([condition, x_next], dim=1)
            else:
                model_input = x_next
            if t_cur <= guidance_high or i==0:
                model_input = torch.cat([model_input, model_input], dim=0)
                y_cur = torch.cat([y, torch.ones_like(y).cuda()*1000], dim=0)
            else:
                y_cur = y

            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)            
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))
            
            # compute drift
            if i==0:
                v_cur = model(
                model_input.to(dtype=_dtype), y_cur, 
                torch.cat([torch.zeros(model_input.size(0)*(1 if scale_index else 0)).to(device=device, dtype=torch.float64), time_input.to(dtype=_dtype)], dim=0), 
                True, scale_index, mask).to(torch.float64)
            else:
                v_cur = model(
                model_input.to(dtype=_dtype), y_cur, 
                time_input.to(dtype=_dtype), 
                False, scale_index).to(torch.float64)
            model_input=model_input[:, -seq_len//clusters:]
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur

            if t_cur <= guidance_high or i ==0:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)  
                else:
                    d_cur = d_cur_cond    
            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
    
    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    
    if condition is not None and i==0:
        model_input = torch.cat([condition, x_next], dim=1)
    else:
        model_input = x_next
    if t_cur <= guidance_high or i==0:
        model_input = torch.cat([model_input, model_input], dim=0)
        y_cur = torch.cat([y, torch.ones_like(y).cuda()*1000], dim=0)
    else:
        y_cur = y


    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    if i==0:
        v_cur = model(
        model_input.to(dtype=_dtype), y_cur, 
        torch.cat([torch.zeros(model_input.size(0)*(1 if scale_index else 0)).to(device=device, dtype=torch.float64), time_input.to(dtype=_dtype)], dim=0), 
        True, scale_index).to(torch.float64)
    else:
        v_cur = model(
        model_input.to(dtype=_dtype), y_cur, 
        time_input.to(dtype=_dtype), 
        False, scale_index).to(torch.float64)
    # v_cur = model(
    #             model_input.to(dtype=_dtype), y_cur, 
    #             torch.cat([torch.zeros(model_input.size(0)*scale_index).to(device=device, dtype=torch.float64), time_input.to(dtype=_dtype)], dim=0), 
    #             ).to(torch.float64)
    model_input=model_input[:, -seq_len//clusters::]
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur

    if t_cur <= guidance_high or i ==0:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)  
        else:
            d_cur = d_cur_cond    
    # if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
    #     d_cur_cond, d_cur_uncond = d_cur.chunk(2)
    #     d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
                    
    return mean_x

    
import numpy as np
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn as nn
from timm.layers import SwiGLU
from timm.models.vision_transformer import LayerScale, DropPath
from typing import Optional
import torch.nn.functional as F
# from .rope import *
# from .flow import SILoss
# import models.sampler as sampler
import torch.nn as nn
import math
import torch.utils.checkpoint
from flash_attn import flash_attn_func
from scipy.stats import norm



def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


from einops import repeat

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Split last dim into two halves and rotate:  (x1, x2) -> (-x2, x1)
    Works on any leading batch/sequence dims.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding1D(nn.Module):
    """
    1-D (language-style) Rotary Positional Embedding.
    Args
    ----
    dim        : embedding dimension (must be even)
    seq_len    : maximum sequence length to pre-compute
    theta      : base in 1 / theta^{2i/dim} (≈ 10 000 for GPT-style RoPE)
    """
    def __init__(self, dim: int, seq_len: int = 128, clusters: int = 4, theta: float = 10_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("`dim` must be even for RoPE.")
            
        #   ω_i = 1 / theta^(2i/dim)   for i = 0 .. dim/2-1
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))   # (dim/2,)
        
        #   t   = 0 .. seq_len-1
        t = torch.arange(seq_len).float()                                  # (seq_len,)
        
        #   outer product → (seq_len, dim/2)
        freqs = torch.einsum('n , d -> n d', t, freqs)                     # (N, D/2)
        freqs = repeat(freqs, 'n d -> n (d r)', r=2)                       # duplicate for cos/sin pair
        
        # cache
        self.register_buffer('cos_cached',  freqs.cos(),  persistent=False)  # (N, dim)
        self.register_buffer('sin_cached',  freqs.sin(),  persistent=False)  # (N, dim)
        self.seq_len = seq_len
        self.clusters = clusters

    def forward(self, x: torch.Tensor, scale_index=None) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)  – applies RoPE in-place on the last dimension.
        """
        if scale_index is None:
            seq_len = x.size(2)
            cos = self.cos_cached[:seq_len]   # (seq_len, dim)
            sin = self.sin_cached[:seq_len]   # (seq_len, dim)
        else:
            cos = self.cos_cached[(scale_index+1)*self.seq_len//self.clusters-x.shape[2]:(scale_index+1)*self.seq_len//self.clusters]
            sin = self.sin_cached[(scale_index+1)*self.seq_len//self.clusters-x.shape[2]:(scale_index+1)*self.seq_len//self.clusters]
        
        # broadcast cos/sin to (batch, seq_len, dim)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        return  x * cos + rotate_half(x) * sin
    
        
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps: float = 1e-6, weight=False):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight=None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is None:
            return output
        else:
            return output * self.weight




class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            scale=None,
            seq_len=128,
            clusters=4,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.seq_len = seq_len
        self.clusters = clusters

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # half_head_dim = dim // num_heads // 2
        # hw_seq_len = 16
        # self.rope = VisionRotaryEmbeddingFast(
        #     dim=half_head_dim,
        #     pt_seq_len=hw_seq_len,
        # )
        self.rope = RotaryEmbedding1D(
            dim= dim // num_heads,
            seq_len=seq_len,
            clusters=clusters
        )
        self.resolusion = scale
        self.k=None
        self.v=None

    def forward(self, x: torch.Tensor, mask, update_cache=True,scale_index=None) -> torch.Tensor:
        B, N, C = x.shape
        if self.training:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = self.rope(q)
            k = self.rope(k)
            x = F.scaled_dot_product_attention(
                    q, k, v,attn_mask=mask,
                    dropout_p=self.attn_drop if self.training else 0.,
                )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = self.rope(q, scale_index)
            k = self.rope(k, scale_index)
            if self.k is not None:
                k,v = torch.cat([self.k[:k.shape[0]],k], dim=2), torch.cat([self.v[:k.shape[0]], v], dim=2)
            if update_cache:
                self.k,self.v=k[:, :, :-self.seq_len//self.clusters], v[:, :, :-self.seq_len//self.clusters]

    
            if update_cache:
                x = F.scaled_dot_product_attention(q,k,v,attn_mask=mask).permute(0,2,1,3).reshape(B, N, C)
            else:
                x = F.scaled_dot_product_attention(q,k,v).permute(0,2,1,3).reshape(B, N, C)
    
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = SwiGLU,
            scale=None,
            seq_len: int = 128,
            clusters: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.clusters = clusters
        self.cluster_size = seq_len // clusters
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            scale=scale,
            seq_len=seq_len,
            clusters=clusters
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = RMSNorm(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio*2/3.),
            act_layer=act_layer,
            drop=proj_drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(dim, 6*dim))
        self.dim=dim


    def forward(self, x: torch.Tensor, condition, mask, update_cache=True,scale_index=None) -> torch.Tensor:
        num_scales = condition.shape[0]//x.shape[0]
        condition = self.ada_lin(condition).view(-1, 1, 6, self.dim).chunk(num_scales, dim=0)
        condition=torch.cat([condition[i].repeat(1, self.cluster_size, 1,1) for i in range(num_scales)], dim=1)
        gamma1, gamma2, scale1, scale2, shift1, shift2 = condition.unbind(2)
        x = x + self.drop_path1(self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1), mask,update_cache=update_cache,scale_index=scale_index).mul_(gamma1))
        x = x + self.drop_path2(self.mlp(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x


class xAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 diffusion_batch_mul=4, 
                 clusters=1,
                 vae_1d=True,
                 vae_seq_len=127
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.vae_1d = vae_1d
        if not vae_1d:
            self.vae_stride = vae_stride
            self.patch_size = patch_size
            self.seq_h = self.seq_w = img_size // vae_stride // patch_size
            self.seq_len = self.seq_h * self.seq_w
        else:
            self.vae_stride = 1
            self.patch_size = 1
            self.seq_len = vae_seq_len
        if not vae_1d:
            self.cluster_h, self.cluster_w = int(np.sqrt(clusters)), int(np.sqrt(clusters))
        self.clusters = clusters
        self.token_embed_dim = vae_embed_dim * patch_size**2
        

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(1000+1, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.time_embed = TimestepEmbedder(encoder_embed_dim)
        # Fake class embedding for CFG's unconditional generation
        #self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))


        # --------------------------------------------------------------------------
        # encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = RMSNorm(encoder_embed_dim, weight=True)#nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.mask_ratio_generator = stats.truncnorm((0.7 - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout, seq_len=self.seq_len, clusters=self.clusters) for _ in range(encoder_depth)])
        self.encoder_norm =  RMSNorm(encoder_embed_dim, weight=True)

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout, seq_len=self.seq_len, clusters=self.clusters) for _ in range(decoder_depth)])

        self.decoder_norm =  RMSNorm(decoder_embed_dim, weight=True) 
        self.pred = nn.Linear(decoder_embed_dim, vae_embed_dim)
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        
        self.flow_loss_fn = SILoss()
        self.diffusion_batch_mul = diffusion_batch_mul

        attention_mask = []
        start=0
        total_length = self.seq_len
        for pz in range(clusters):
            start+=self.seq_len//clusters
            attention_mask.append(torch.cat([torch.ones((self.seq_len//clusters, start)),
                                             torch.zeros((self.seq_len//clusters, total_length - start))], dim=-1))
        # self.variable('constant', 'attention_mask', lambda :jnp.concatenate(attention_mask, axis=0))
        attention_mask = torch.cat(attention_mask, dim=0)
        attention_mask = torch.where(attention_mask == 0, -torch.inf, attention_mask)
        attention_mask = torch.where(attention_mask == 1, 0, attention_mask)
        # self.attention_mask = attention_mask
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer('mask', attention_mask.contiguous())


    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def clusterify(self, x):
        if self.vae_1d:
            bsz, l, c = x.shape
            p = l // self.clusters
            x = x.reshape(bsz, -1, p, c)
            return [i.squeeze() for i in x.chunk(self.clusters, dim=1)]
        else:
            bsz, c, h, w = x.shape
            p = h//self.cluster_h
            h_, w_ = h // p, w // p

            x = x.reshape(bsz, c, h_, p, w_, p)
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(bsz, h_ * w_, p ** 2, c)
            return [i.squeeze() for i in x.chunk(self.clusters, dim=1)]

    def unclusterify(self, x):
        b, n, c=x.shape
        if self.vae_1d:
            x = x.reshape(b, -1, c)
        else:
            x = x.reshape(b, 2,2, 8,8, c)
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(b, c, 16, 16)
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]


    def forward_mae_encoder(self, x, condition, mask, update_cache=True,scale_index=0):
        if self.training:
            encoder_pos_embed_learned=self.encoder_pos_embed_learned
        else:
            encoder_pos_embed_learned =self.encoder_pos_embed_learned[:, (scale_index+1)*self.seq_len//self.clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.clusters]
        if (not self.training) and (mask is not None):
            mask = mask[:,:,(scale_index+1)*self.seq_len//self.clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.clusters, :]
        x = x + encoder_pos_embed_learned
        x = self.z_proj_ln(x)
        for blk in self.encoder_blocks:
            x = blk(x, condition, mask, update_cache=update_cache, scale_index=scale_index)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, condition, mask, update_cache=True, scale_index=0):
        x = self.decoder_embed(x)
        # decoder position embedding
        if self.training:
            decoder_pos_embed_learned=self.decoder_pos_embed_learned
        else:
            decoder_pos_embed_learned=self.decoder_pos_embed_learned[:, (scale_index+1)*self.seq_len//self.clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.clusters]
        if (not self.training) and (mask is not None):
            mask = mask[:,:,(scale_index+1)*self.seq_len//self.clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.clusters, :(scale_index+1)*self.seq_len//self.clusters]
        x = x + decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, condition, mask,update_cache=update_cache,scale_index=scale_index)
        x = self.decoder_norm(x)
        x = self.pred(x)
        return x

    def mean_flat(self, x):
        """
        Take the mean over all non-batch dimensions.
        """
        return torch.mean(x, dim=list(range(1, len(x.size()))))
    
    def sample_logit_normal(self, mu=0, sigma=1, size=1):
        # Generate samples from the normal distribution
        samples = norm.rvs(loc=mu, scale=sigma, size=size)
        
        # Transform samples to be in the range (0, 1) using the logistic function
        samples = 1 / (1 + np.exp(-samples))

        # Numpy to Tensor
        samples = torch.tensor(samples, dtype=torch.float32)

        return samples
        
    def forward(self, imgs, labels):
        # B,C,_,_=imgs.shape
        B, _, C = imgs.shape
        label_drop = torch.rand(imgs.shape[0],).cuda()<self.label_drop_prob
        fake_label = torch.ones(imgs.shape[0],).cuda()*1000
        labels = torch.where(label_drop, fake_label, labels)

        patches = self.clusterify(imgs) #B N C
        time_input = self.sample_logit_normal(size=patches[0].shape[0]*self.clusters)
        time_input = time_input.unsqueeze(-1).unsqueeze(-1)
        time_input = time_input.to(device=patches[0].device, dtype=patches[0].dtype)
        
        noises = [torch.randn_like(patches[0]) for i in range(self.clusters)]
        alpha_t = 1 - time_input
        sigma_t = time_input
        d_alpha_t = -1
        d_sigma_t =  1

        alpha_t = alpha_t.chunk(self.clusters, dim=0)
        sigma_t = sigma_t.chunk(self.clusters, dim=0)

        model_input = [alpha_t[i] * patches[i] + sigma_t[i] * noises[i] for i in range(len(patches))]
        model_target = [d_alpha_t * patches[i] + d_sigma_t * noises[i] for i in range(len(patches))]
        

        time_input = (time_input*1000).long() 
        time_input = self.time_embed(time_input).squeeze()
        class_embedding = self.class_emb(labels.long()).squeeze()
        model_input= torch.stack(model_input, dim=1).reshape(B,-1, C)
        model_input = self.z_proj(model_input)
        x = self.forward_mae_encoder(model_input, class_embedding.repeat(self.clusters,1)+time_input, self.mask)
        model_output = self.forward_mae_decoder(x, class_embedding.repeat(self.clusters,1)+time_input, self.mask)

        model_target = torch.stack(model_target, dim=1).reshape(B,-1, C)
        denoising_loss = self.mean_flat((model_output - model_target) ** 2).mean()
        return denoising_loss



    def sample_inference(self, x, label, time_input, update_cache, scale_index, mask=None):
        class_embedding = self.class_emb(label.long()).squeeze()
        time_input = (time_input*1000).long() 
        time_input = self.time_embed(time_input).squeeze()
        step =time_input.shape[0]//x.shape[0]
        condition = class_embedding.repeat(step,1)+time_input
        model_input = self.z_proj(x)
        x = self.forward_mae_encoder(model_input, condition, mask, update_cache=update_cache, scale_index=scale_index)
        model_output = self.forward_mae_decoder(x, condition, mask, update_cache=update_cache, scale_index=scale_index)
        if update_cache:
            model_output=model_output[:, -self.seq_len//self.clusters:]
        return model_output



    def sample_tokens(self, num_steps, cfg=1.0, label=None):
        
        if label is None:
            label = torch.ones_like(label).cuda()*1000
 
        indices = list(range(self.clusters))
        sequence = [self.seq_len // self.clusters for i in range(self.clusters)]
        sequence = torch.cumsum(torch.tensor(sequence),dim=0)
        prev_cond=None
        generated=None
        for blk in self.encoder_blocks:
            blk.attn.k=None
            blk.attn.v=None
        for blk in self.decoder_blocks:
            blk.attn.k=None
            blk.attn.v=None
        for step in indices:
            # print("step", step)
            scaled_cfg = (cfg-1)*step/(self.clusters - 1) +1  #(cfg-1)*(step+1)/self.clusters+1 ##
            latents= torch.randn([label.shape[0], self.seq_len//self.clusters, self.vae_embed_dim]).cuda()
            z_sample = euler_maruyama_sampler(self.sample_inference, latents, label, step, num_steps=num_steps, condition=prev_cond, cfg_scale=scaled_cfg, mask= self.mask[:,:,:sequence[step], :sequence[step]], seq_len=self.seq_len, clusters=self.clusters).float()
            prev_cond = z_sample
            if generated is None:
                generated = z_sample
            else:
                generated = torch.cat([generated, z_sample], dim=1)
        #     print("encoder k cache", self.encoder_blocks[0].attn.k.shape)
        #     print("decoder k cache", self.decoder_blocks[0].attn.k.shape)
        # # unpatchify
        # tokens = self.unclusterify(generated)
        tokens = generated
        return tokens


def xar_base(**kwargs):
    model = xAR(
        encoder_embed_dim=768, encoder_depth=8, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def xar_large(**kwargs):
    model = xAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def xar_huge(**kwargs):
    model = xAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model