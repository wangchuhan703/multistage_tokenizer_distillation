# ### This file contains impls for MM-DiT, the core model component of SD3
# #### Modifications: Added 2D RoPE for x, and choice of Learnable PE or Cross-Modality RoPE for context ####

import math
from typing import Dict, Optional, Tuple, Literal, Union # Added Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from masks.utils import apply_masks_with_mask_token, apply_masks, reverse_masks

# (Assuming other imports like transformers are available if needed)

#################################################################################################
### Core/Utility
#################################################################################################

# RoPE Helper Functions
#--------------------------------------------------------------------------

def _calculate_rope_base_freqs(dim: int, base: int = 10000, device=None, dtype=None) -> torch.Tensor:
    """Calculates the base frequencies for RoPE dimensions."""
    # Calculates frequencies for dim // 2 dimensions
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype)[: (dim // 2)] / dim))
    return freqs

def get_rope_freqs_for_positions(
        positions: torch.Tensor, # Shape: (seq_len, 2) where columns are (h, w)
        dim: int, # Typically head_dim
        base: int = 10000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
    """
    Calculates RoPE frequencies (cis) for specific 2D positions.
    Args:
        positions: Tensor of shape (seq_len, 2) containing (h, w) coordinates.
        dim: The dimension of the features RoPE is applied to (e.g., head_dim).
        base: The RoPE base frequency.
        device: Torch device.
        dtype: Torch dtype.
    Returns:
        Tensor of shape (seq_len, dim // 2) with complex RoPE frequencies (cis).
    """
    if device is None:
        device = positions.device
    if dtype is None:
        dtype = positions.dtype

    # Ensure positions are float for multiplication
    positions = positions.float()

    # Calculate base frequencies for H and W dimensions (each gets dim // 4)
    half_dim = dim // 2
    freqs_h = _calculate_rope_base_freqs(half_dim, base, device=device, dtype=dtype) # (dim // 4)
    freqs_w = _calculate_rope_base_freqs(half_dim, base, device=device, dtype=dtype) # (dim // 4)

    # Calculate theta values: position * frequency
    # positions[:, 0] is h_pos (seq_len,), positions[:, 1] is w_pos (seq_len,)
    # theta_h: (seq_len, 1) * (1, dim // 4) -> (seq_len, dim // 4)
    # theta_w: (seq_len, 1) * (1, dim // 4) -> (seq_len, dim // 4)
    theta_h = positions[:, 0].unsqueeze(1) * freqs_h.unsqueeze(0)
    theta_w = positions[:, 1].unsqueeze(1) * freqs_w.unsqueeze(0)

    # Convert to complex numbers (cis = cos + i*sin)
    freqs_cis_h = torch.polar(torch.ones_like(theta_h), theta_h) # (seq_len, dim // 4)
    freqs_cis_w = torch.polar(torch.ones_like(theta_w), theta_w) # (seq_len, dim // 4)

    # Concatenate H and W frequencies
    freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1) # (seq_len, dim // 2)

    return freqs_cis

def precompute_rope_freqs_2d(dim: int, max_h: int, max_w: int, base: int = 10000, device=None, dtype=None) -> torch.Tensor:
    """Precomputes 2D RoPE frequencies for a full grid up to max_h, max_w."""
    # Create all grid positions (h, w) from (0,0) to (max_h-1, max_w-1)
    h_coords = torch.arange(max_h, device=device)
    w_coords = torch.arange(max_w, device=device)
    grid = torch.stack(torch.meshgrid(h_coords, w_coords, indexing='ij'), dim=-1) # (max_h, max_w, 2)
    grid_positions = grid.reshape(-1, 2) # (max_h * max_w, 2)

    # Calculate frequencies for these grid positions
    freqs_cis = get_rope_freqs_for_positions(grid_positions, dim, base, device=device, dtype=dtype)
    return freqs_cis # (max_h * max_w, dim // 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshapes freqs_cis for broadcasting compatibility with attention tensors."""
    ndim = x.ndim
    assert ndim >= 4
    shape = [1] * ndim
    shape[1] = freqs_cis.shape[0] # Match sequence length dimension
    shape[-1] = freqs_cis.shape[-1] # Match frequency dimension
    return freqs_cis.view(*shape)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies RoPE rotations to the input tensor."""
    # x shape: (..., seq_len, ..., dim) -> e.g., (b, h, s, d) or (b, s, h, d) after split_qkv
    # freqs_cis shape: (seq_len, dim // 2)
    x_ = x.float().reshape(*x.shape[:-1], -1, 2) # (..., seq_len, ..., dim // 2, 2)
    x_complex = torch.view_as_complex(x_) # (..., seq_len, ..., dim // 2)

    # Reshape freqs for broadcasting (e.g., to (1, 1, seq_len, 1, dim//2) if x is (b,h,s,d))
    # Let's assume x is (b, s, h, d) for simplicity here, needs adjustment if layout differs
    # Target freqs shape: (1, seq_len, 1, dim // 2) for broadcasting
    if freqs_cis.ndim == 2:
        freqs_cis_reshaped = reshape_for_broadcast(freqs_cis, x_complex) # (1, s, 1, d//2)
    else:
        freqs_cis_reshaped = freqs_cis.unsqueeze(2) # Already in the right shape

    # Apply rotation
    x_rotated = x_complex * freqs_cis_reshaped.to(x_complex.device) # Ensure device match
    x_out = torch.view_as_real(x_rotated) # (..., dim // 2, 2)
    x_out = x_out.flatten(-2) # (..., dim)

    return x_out.type_as(x)

#--------------------------------------------------------------------------

# --- (attention, Mlp, build_mlp, PatchEmbed, modulate remain the same) ---
def attention(q, k, v, heads, mask=None):
    b, _, dim = q.shape # q shape is (b, seq_len, heads * dim_head)
    dim_head = dim // heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v)) # (b, heads, seq_len, dim_head)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, dim) # (b, seq_len, heads * dim_head)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=None, device=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

def build_mlp(hidden_size, projector_dim, z_dim):
    # ... (implementation)
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim), nn.SiLU(),
                nn.Linear(projector_dim, projector_dim), nn.SiLU(),
                nn.Linear(projector_dim, z_dim))

class PatchEmbed(nn.Module):
    # ... (implementation)
    def __init__(self, img_size: Optional[int] = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, flatten: bool = True, bias: bool = True, strict_img_size: bool = True, dynamic_img_pad: bool = False, dtype=None, device=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else: self.img_size = self.grid_size = self.num_patches = None
        self.flatten = flatten; self.strict_img_size = strict_img_size; self.dynamic_img_pad = dynamic_img_pad
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)
    def forward(self, x):
        B, C, H, W = x.shape
        if self.grid_size is None: # Dynamic grid calc
             gh, gw = H // self.patch_size[0], W // self.patch_size[1]
             self.grid_size = (gh, gw); self.num_patches = gh * gw
        if self.strict_img_size and self.img_size is not None and (H != self.img_size[0] or W != self.img_size[1]): raise ValueError("Input image size mismatch")
        x = self.proj(x)
        if self.flatten: x = x.flatten(2).transpose(1, 2)
        return x

def modulate(x, shift, scale):
    if shift is None: shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

# --- (TimestepEmbedder, VectorEmbedder remain the same) ---
class TimestepEmbedder(nn.Module):
    # ... (implementation)
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device))
        self.frequency_embedding_size = frequency_embedding_size
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # ... (implementation)
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    def forward(self, t, dtype, **kwargs):
        t_freq = self.timestep_embedding(t.to(self.mlp[0].weight.device), self.frequency_embedding_size).to(dtype)
        return self.mlp(t_freq)

class VectorEmbedder(nn.Module):
    # ... (implementation)
    def __init__(self, input_dim: int, hidden_size: int, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.to(device=self.mlp[0].weight.device, dtype=self.mlp[0].weight.dtype))

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

# --- (split_qkv remains the same) ---
def split_qkv(qkv, head_dim):
    qkv = qkv.view(qkv.shape[0], qkv.shape[1], 3, -1, head_dim)
    qkv = qkv.permute(2, 0, 1, 3, 4) # 3, b, s, h, d
    return qkv[0], qkv[1], qkv[2] # Each (b, s, h, d)

# --- SelfAttention remains the same (accepts freqs_cis) ---
class SelfAttention(nn.Module):
    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_mode: str = "torch", pre_only: bool = False, qk_norm: Optional[str] = None, rmsnorm: bool = False, dtype=None, device=None):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = nn.Linear(dim, dim, dtype=dtype, device=device) if not pre_only else nn.Identity()
        self.pre_only = pre_only
        self.attn_mode = attn_mode
        self.qk_norm = qk_norm
        norm_layer = RMSNorm if qk_norm == "rms" else nn.LayerNorm if qk_norm == "ln" else nn.Identity
        if qk_norm:
            norm_args = {'eps': 1e-6, 'dtype': dtype, 'device': device}
            # RMSNorm specific args if needed
            if qk_norm == "rms": norm_args['elementwise_affine'] = True # Example adjustment
            self.ln_q = norm_layer(self.head_dim, **norm_args)
            self.ln_k = norm_layer(self.head_dim, **norm_args)
        else: self.ln_q = self.ln_k = nn.Identity()

    def pre_attention(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None):
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim) # Each (B, L, H, D_head)

        # Apply QK Norm per head
        if self.qk_norm:
            q_dtype = q.dtype 
            q = self.ln_q(q).to(q_dtype)
            k = self.ln_k(k).to(q_dtype)

        # Apply RoPE if freqs_cis are provided
        if freqs_cis is not None:
            # Ensure freqs_cis has the right shape and device (L, D_head // 2)
            q = apply_rope(q, freqs_cis=freqs_cis.to(q.device))
            k = apply_rope(k, freqs_cis=freqs_cis.to(k.device))

        # Reshape back to (B, L, C)
        q = q.reshape(B, L, C); k = k.reshape(B, L, C); v = v.reshape(B, L, C)
        return q, k, v

    def post_attention(self, x: torch.Tensor) -> torch.Tensor: return self.proj(x)
    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = self.pre_attention(x, freqs_cis=freqs_cis)
        x = attention(q, k, v, self.num_heads)
        x = self.post_attention(x)
        return x

# --- (RMSNorm, SwiGLUFeedForward remain the same) ---
class RMSNorm(torch.nn.Module):
    # ... (implementation)
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale: self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        else: self.register_parameter("weight", None)
    def _norm(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.to(device=x.device, dtype=x.dtype) if self.learnable_scale else output

class SwiGLUFeedForward(nn.Module):
    # ... (implementation)
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256, ffn_dim_multiplier: Optional[float] = None, dtype=None, device=None, bias: bool = False):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None: hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias, dtype=dtype, device=device)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias, dtype=dtype, device=device)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias, dtype=dtype, device=device)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

# --- (DismantledBlock remains the same - already handles freqs_cis) ---
class DismantledBlock(nn.Module):
    # ... (implementation)
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, attn_mode: str = "torch", qkv_bias: bool = False, pre_only: bool = False, rmsnorm: bool = False, scale_mod_only: bool = False, swiglu: bool = False, qk_norm: Optional[str] = None, dtype=None, device=None, disable_adaLN: bool = False, **block_kwargs):
        super().__init__()
        norm_layer = RMSNorm if rmsnorm else nn.LayerNorm
        self.norm1 = norm_layer(hidden_size, elementwise_affine=disable_adaLN, eps=1e-6, dtype=dtype, device=device)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=pre_only, qk_norm=qk_norm, rmsnorm=rmsnorm, dtype=dtype, device=device)
        self.pre_only = pre_only
        if not pre_only:
            self.norm2 = norm_layer(hidden_size, elementwise_affine=disable_adaLN, eps=1e-6, dtype=dtype, device=device)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            mlp_module = SwiGLUFeedForward if swiglu else Mlp
            mlp_kwargs = {'dim': hidden_size, 'hidden_dim': mlp_hidden_dim} if swiglu else {'in_features': hidden_size, 'hidden_features': mlp_hidden_dim, 'act_layer': nn.GELU}
            self.mlp = mlp_module(**mlp_kwargs, dtype=dtype, device=device) # Pass common args
        self.scale_mod_only = scale_mod_only
        n_mods = (4 if not pre_only else 1) if scale_mod_only else (6 if not pre_only else 2)
        self.disable_adaLN = disable_adaLN
        if not disable_adaLN:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))
        else:
            self.shift, self.scale = nn.Parameter(torch.zeros((1,  hidden_size), device=device, dtype=dtype), requires_grad=False), nn.Parameter(torch.zeros((1, hidden_size), device=device, dtype=dtype), requires_grad=False)

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None):
        # mod_params = self.adaLN_modulation(c)
        if not self.pre_only:
            if not self.scale_mod_only:
                if not self.disable_adaLN:
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
                else:
                    shift_msa, scale_msa = self.shift, self.scale
                    gate_msa = torch.ones_like(scale_msa)
                    shift_mlp, scale_mlp = self.shift, self.scale
                    gate_mlp = torch.ones_like(scale_mlp)
            else:
                if not self.disable_adaLN:
                    shift_msa = None
                    shift_mlp = None
                    scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
                else:
                    shift_msa = None
                    shift_mlp = None
                    scale_msa, gate_msa = self.shift, self.scale
                    scale_mlp = self.shift
                    gate_mlp = torch.ones_like(scale_mlp)
            intermediates_post = (gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                if not self.disable_adaLN:
                    shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
                else:
                    shift_msa, scale_msa = self.shift, self.scale
            else:
                if not self.disable_adaLN:
                    shift_msa = None
                    scale_msa = self.adaLN_modulation(c)
                else:
                    shift_msa = None
                    scale_msa = self.shift
            intermediates_post = None
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x_mod, freqs_cis=freqs_cis)
        return (q, k, v), (x, intermediates_post) # Pass original x

    def post_attention(self, attn_output: torch.Tensor, x: torch.Tensor, intermediates_post: Tuple) -> torch.Tensor:
        assert not self.pre_only
        gate_msa, shift_mlp, scale_mlp, gate_mlp = intermediates_post
        attn_proj = self.attn.post_attention(attn_output)
        x = x + gate_msa.unsqueeze(1) * attn_proj
        x_mod = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, attn_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert not self.pre_only
        (q, k, v), (x_orig, intermediates_post) = self.pre_attention(x, c, freqs_cis=freqs_cis)
        attn_output = attention(q, k, v, self.attn.num_heads, attn_masks)
        x = self.post_attention(attn_output, x_orig, intermediates_post)
        return x


# Modified block_mixing to accept and pass freqs_cis for both context and x
def block_mixing(
        context, x,
        context_block, x_block, c,
        freqs_cis_context, # RoPE frequencies for context block
        freqs_cis_x,        # RoPE frequencies for x block
        attn_masks=None
    ):
    assert context is not None, "block_mixing called with None context"

    # 1. Pre-attention (Norm, Modulate, QKV, optional RoPE)
    # Pass respective RoPE frequencies (can be None)
    context_qkv, (context_orig, context_intermediates_post) = context_block.pre_attention(context, c, freqs_cis=freqs_cis_context)
    x_qkv, (x_orig, x_intermediates_post) = x_block.pre_attention(x, c, freqs_cis=freqs_cis_x)

    # 2. Concatenate Q, K, V
    q = torch.cat((context_qkv[0], x_qkv[0]), dim=1)
    k = torch.cat((context_qkv[1], x_qkv[1]), dim=1)
    v = torch.cat((context_qkv[2], x_qkv[2]), dim=1)

    # 3. Joint Attention
    attn_output = attention(q, k, v, x_block.attn.num_heads, attn_masks)

    # 4. Split attention output
    context_len = context_qkv[0].shape[1]
    context_attn_output = attn_output[:, :context_len]
    x_attn_output = attn_output[:, context_len:]

    # 5. Post-attention (Proj, Residual, MLP)
    if not context_block.pre_only:
        context = context_block.post_attention(context_attn_output, context_orig, context_intermediates_post)
    else:
        context = None # Context block is pre_only (last layer), output not needed further

    # x_block always runs post_attention
    x = x_block.post_attention(x_attn_output, x_orig, x_intermediates_post)

    return context, x


class JointBlock(nn.Module):
    """Wrapper, passes separate RoPE freqs for context and x."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        qk_norm = kwargs.pop("qk_norm", None)
        # Clone kwargs to avoid modification issues if dicts are passed
        kwargs_context = kwargs.copy()
        kwargs_x = kwargs.copy()
        # Pass all remaining args/kwargs
        self.context_block = DismantledBlock(*args, pre_only=pre_only, qk_norm=qk_norm, **kwargs_context)
        self.x_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, **kwargs_x)

    def forward(self, context: torch.Tensor, x: torch.Tensor, c: torch.Tensor,
                freqs_cis_context: Optional[torch.Tensor],
                freqs_cis_x: Optional[torch.Tensor],
                attn_masks: Optional[torch.Tensor]):
        """ Calls block_mixing with separate frequencies. """
        return block_mixing(context, x,
                            context_block=self.context_block,
                            x_block=self.x_block,
                            c=c,
                            freqs_cis_context=freqs_cis_context,
                            freqs_cis_x=freqs_cis_x,
                            attn_masks=attn_masks)


# --- (FinalLayer remains the same) ---
class FinalLayer(nn.Module):
    # ... (implementation)
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, total_out_channels: Optional[int] = None, dtype=None, device=None, disable_adaLN: bool = False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        linear_out_dim = total_out_channels if total_out_channels is not None else (patch_size * patch_size * out_channels)
        self.linear = nn.Linear(hidden_size, linear_out_dim, bias=True, dtype=dtype, device=device)
        self.disable_adaLN = disable_adaLN
        if not disable_adaLN:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))
        else:
            self.shift, self.scale = nn.Parameter(torch.zeros((1, hidden_size), device=device, dtype=dtype), requires_grad=False), nn.Parameter(torch.zeros((1, hidden_size), device=device, dtype=dtype), requires_grad=False)
            
    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.disable_adaLN:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        else:
            shift, scale = self.shift, self.scale 
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
    def get_last_layer_weight(self):
        return self.linear.weight



class MMDiT(nn.Module):
    """
    MMDiT with 2D RoPE for image patches (x) and a choice of positional
    encoding for context: 'learnable' absolute PE or 'cross_rope' 2D RoPE.
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        depth: int = 24,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None,
        context_embedding_dim: int = 768,
        max_context_seq_len: int = 77,
        context_pos_encoding_type: Literal['learnable', 'cross_rope', 'none'] = 'learnable', # Choice here!
        register_length: int = 0,
        attn_mode: str = "torch",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_max_size:  Optional[int] = None, # Max grid dim for image RoPE precomputation
        qk_norm: Optional[str] = None,
        qkv_bias: bool = True,
        rope_base: int = 10000, # RoPE base frequency
        dtype = None,
        device = None,
        using_cfg=False,
        output_register=False
    ):
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.float32
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.patch_size = patch_size
        if pos_embed_max_size is None:
            self.pos_embed_max_size = input_size // patch_size # pos_embed_max_size
        else:
            self.pos_embed_max_size = pos_embed_max_size
        self.using_cfg = using_cfg
        self.max_context_seq_len = max_context_seq_len
        self.context_pos_encoding_type = context_pos_encoding_type
        self.register_length = register_length
        self.rope_base = rope_base
        print(f"mmdit using cfg: {self.using_cfg}")
        self.output_register = output_register
        print(f"mmdit output register: {self.output_register}")
        # used for jepa
        self.num_img_tokens = (input_size // patch_size) ** 2
        
        hidden_size = 64 * depth
        num_heads = depth
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads (depth)"
        self.hidden_size = hidden_size

        # --- Embedders ---
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device)
        self.y_embedder = VectorEmbedder(adm_in_channels, hidden_size, dtype=dtype, device=device) if adm_in_channels is not None else None
        self.context_embedder = nn.Linear(context_embedding_dim, hidden_size, dtype=dtype, device=device)

        if self.using_cfg:
            self.context_null = nn.Parameter(torch.randn(1, max_context_seq_len, context_embedding_dim, dtype=dtype, device=device))

        # --- Positional Embeddings ---
        # Optional Register Tokens
        self.register = nn.Parameter(torch.randn(1, register_length, hidden_size, dtype=dtype, device=device)) if register_length > 0 else None

        # Context Positional Encoding
        self.context_pos_embed = None
        print(f"Using context_pos_encoding_type: {context_pos_encoding_type}")
        if self.context_pos_encoding_type == 'learnable':
            total_context_len = max_context_seq_len + register_length
            self.context_pos_embed = nn.Parameter(torch.zeros(1, total_context_len, hidden_size, dtype=dtype, device=device))
            nn.init.normal_(self.context_pos_embed, std=0.02)
        elif self.context_pos_encoding_type not in ['cross_rope', 'none']:
            raise ValueError(f"Invalid context_pos_encoding_type: {context_pos_encoding_type}")

        # Precompute Image RoPE frequencies
        rope_freqs = precompute_rope_freqs_2d(self.head_dim, self.pos_embed_max_size + 1 if self.context_pos_encoding_type == 'cross_rope' else self.pos_embed_max_size, self.pos_embed_max_size, base=self.rope_base, device=device, dtype=dtype)
        self.register_buffer("rope_freqs_cis_img_max", rope_freqs, persistent=False)

        # --- Transformer Blocks ---
        self.joint_blocks = nn.ModuleList(
            [JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=(i == depth - 1) if not self.output_register else False, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only, swiglu=swiglu, qk_norm=qk_norm, dtype=dtype, device=device, disable_adaLN=self.output_register) for i in range(depth)]
        )

        # --- Final Layer ---
        self.embed_dim = hidden_size
        if self.output_register:
            self.final_layer = FinalLayer(hidden_size, 1, self.out_channels, dtype=dtype, device=device, disable_adaLN=True)
        else:
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=dtype, device=device)


    def select_rope_freqs_img(self, hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Selects precomputed image RoPE frequencies for the actual grid size."""
        h, w = hw
        p = self.patch_size
        gh, gw = h // p, w // p
        assert gh <= self.pos_embed_max_size and gw <= self.pos_embed_max_size
        top, left = (self.pos_embed_max_size - gh) // 2, (self.pos_embed_max_size - gw) // 2
        max_freqs = self.rope_freqs_cis_img_max.view(self.pos_embed_max_size + 1 if self.context_pos_encoding_type == 'cross_rope' else self.pos_embed_max_size, self.pos_embed_max_size, -1)
        selected_freqs = max_freqs[top : top + gh, left : left + gw, :]
        selected_freqs = selected_freqs.reshape(gh * gw, -1)
        return selected_freqs.to(device)


    def calculate_rope_freqs_context(self, img_gw: int, context_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Calculates RoPE frequencies for context tokens using cross-modality positions."""
        # Context positions: h=0, w starts after image grid width (img_gw)
        context_h_pos = torch.ones(context_len, device=device, dtype=torch.float32) * img_gw # Use float for calculations
        context_w_pos = torch.arange(0, context_len, device=device, dtype=torch.float32)
        context_positions = torch.stack([context_h_pos, context_w_pos], dim=-1) # (context_len, 2)

        # Scale the base frequency to account for the larger range
        # The ratio of the max context position to the max image position
        position_scale = context_len / img_gw  
        context_rope_base = self.rope_base * position_scale

        # Calculate RoPE frequencies for these positions
        freqs_cis_context = get_rope_freqs_for_positions(
            context_positions,
            self.head_dim,
            base=context_rope_base,
            device=device,
            dtype=dtype # Use model's target dtype
        )
        return freqs_cis_context # (context_len, head_dim // 2)


    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(
            self, x: torch.Tensor, c_mod: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            freqs_cis_x: Optional[torch.Tensor] = None,
            freqs_cis_context: Optional[torch.Tensor] = None, # Pass context RoPE freqs
            detach: Optional[bool] = False,
            return_img_tokens: bool = False,
            attn_masks: Optional[torch.Tensor] = None, # Pass context RoPE freqs
        ) -> Tuple[torch.Tensor, list]:

        # Note: Positional embeddings/RoPE freqs are assumed to be handled *before* this point

        zs = [] # Placeholder

        for i, block in enumerate(self.joint_blocks):
            if context is None:
                 # This case should ideally be handled upstream or model adapted if context can be absent
                 raise ValueError("Context is None in forward_core_with_concat")
            
            # if self.output_register:
            #     print(f"encoder block {i}, context dtype {context.dtype}, x dtype {x.dtype}")
            # else:
            #     print(f"decoder block {i}, context dtype {context.dtype}, x dtype {x.dtype}")

            # Pass separate RoPE frequencies to the JointBlock
            context, x = block(
                context=context, x=x, c=c_mod,
                freqs_cis_context=freqs_cis_context,
                freqs_cis_x=freqs_cis_x,
                attn_masks=attn_masks,
            )
        
        # used for jepa
        if return_img_tokens:
            return x 

        if self.output_register:
            if self.register_length > 0:
                out = context[:, :self.register_length, :]
            else:
                out = x 
            return out
        else:
            x = self.final_layer(x, c_mod)  # (N, T, patch_size ** 2 * out_channels)
            return x, []

        # x = self.final_layer(x, c_mod)
        # return x, zs

    def forward(
            self, x: Union[torch.Tensor, list[torch.Tensor]], t: Optional[torch.Tensor] = None,
            context_features: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            masks_enc: Optional[torch.Tensor] = None,
            return_img_tokens: bool = False,
            detach: Optional[bool] = False
        ) -> Tuple[torch.Tensor, list]:
        

        

        # 1. Patch Embeddings & Image RoPE Frequencies
        if isinstance(x, list):
            # current_dtype = x[-1].dtype
            # current_device = x[-1].device

            # x_max_length = max(
            #     x_.shape[-2] * x_.shape[-1] // self.patch_size // self.patch_size for x_ in x
            # )
            # context_length = self.register_length if self.output_register else context_features.size(1)

            # x_tokens = []
            # freqs_cis_x = []
            # attn_masks = []
            # hw_list = []
            # bsz_list = []
            # seq_len_list = []
            # bsz_s = 0

            # for x_ in x:
            #     bsz = x_.size(0)
            #     bsz_list.append((bsz_s, bsz_s+bsz))
            #     bsz_s += bsz
            #     hw = x_.shape[-2:]
            #     hw_list.append(hw)

            #     # Tokenize
            #     x_tokens_ = self.x_embedder(x_)  # (B, T_img, D)
            #     seq_len = x_tokens_.size(1)

            #     # Get RoPE frequencies
            #     freqs_cis_x_ = self.select_rope_freqs_img(hw, device=current_device).to(current_dtype)
            #     freqs_cis_x_ = freqs_cis_x_.unsqueeze(0).repeat(bsz, 1, 1)  # (B, T_img, D_head//2)

            #     # Pad to x_max_length
            #     pad_len = x_max_length - seq_len
            #     if pad_len > 0:
            #         x_tokens_ = F.pad(x_tokens_, (0, 0, 0, pad_len))  # pad sequence dim
            #         freqs_cis_x_ = F.pad(freqs_cis_x_, (0, 0, 0, pad_len))

            #     # Create 1D token mask (True = valid token)
            #     token_mask = torch.zeros((bsz, x_max_length + context_length), dtype=torch.bool, device=current_device)
            #     token_mask[:, :seq_len + context_length] = 1  # (B, T)

            #     # Expand to 2D attention mask (B, 1, T, T)
            #     attn_mask = token_mask.unsqueeze(1)  # (B, 1, T)
            #     attn_mask = attn_mask & attn_mask.transpose(2, 1)

            #     seq_len_list.append(seq_len)
            #     x_tokens.append(x_tokens_)
            #     freqs_cis_x.append(freqs_cis_x_)
            #     attn_masks.append(attn_mask)

            # # Concatenate everything
            # x_tokens = torch.cat(x_tokens, dim=0)        # (B_total, T, D)
            # freqs_cis_x = torch.cat(freqs_cis_x, dim=0)  # (B_total, T, D_head//2)
            # attn_masks = torch.cat(attn_masks, dim=0).unsqueeze(1)     # (B_total, 1, T, T)

            # img_gw = hw_list[-1][1] // self.patch_size

            # # Merge time, context_features, and labels if needed
            # if t is not None:
            #     t = torch.cat(t, dim=0)
            # if context_features is not None and isinstance(context_features, list):
            #     context_features = torch.cat(context_features, dim=0)
            # if y is not None:
            #     y = torch.cat(y, dim=0)

            dtype   = x[-1].dtype
            device  = x[-1].device
            current_dtype = dtype
            current_device = device
            
            D       = self.hidden_size 
            Dh2     = self.head_dim // 2

            bszs        = [xi.size(0) for xi in x]
            B_total     = sum(bszs)
            T_max       = max(xi.shape[-2] * xi.shape[-1] // self.patch_size**2 for xi in x)
            ctx_len     = self.register_length if self.output_register else context_features.size(1)

            x_tokens    = torch.zeros(B_total, T_max, D,       device=device, dtype=dtype)
            freqs_cis_x = torch.zeros(B_total, T_max, Dh2,     device=device, dtype=dtype)
            token_masks = torch.zeros(B_total, T_max + ctx_len,device=device, dtype=torch.bool)

            # ===== bookkeeping lists you asked to keep =====
            hw_list:      List[Tuple[int, int]] = []
            bsz_list:     List[Tuple[int, int]] = []
            seq_len_list: List[int]             = []
            # ===============================================

            rope_cache: Dict[Tuple[int, int], torch.Tensor] = {}

            b0 = 0
            for xi in x:
                b1    = b0 + xi.size(0)
                hw    = xi.shape[-2:]
                T_i   = hw[0] * hw[1] // (self.patch_size**2)

                # embed & fill
                tok_i                               = self.x_embedder(xi)       # (b, T_i, D)
                x_tokens[b0:b1, :T_i]               = tok_i

                if hw not in rope_cache:
                    rope_cache[hw] = self.select_rope_freqs_img(hw, device=device)
                rope = rope_cache[hw].to(dtype).unsqueeze(0)
                freqs_cis_x[b0:b1, :T_i]            = rope.expand(xi.size(0), -1, -1)

                # 1-D valid-token mask
                token_masks[b0:b1, :T_i + ctx_len]  = True

                # ---------- fill the bookkeeping lists ----------
                hw_list.append(hw)                  # (H, W) for this *batch*
                bsz_list.append((b0, b1))           # slice in the big concat
                seq_len_list.append(T_i)            # image token length
                # -------------------------------------------------

                b0 = b1
                img_gw = hw[1] // self.patch_size

            attn_masks = token_masks.unsqueeze(1) & token_masks.unsqueeze(2)   # (B,1,L,L)
            attn_masks = attn_masks.unsqueeze(1)
            # print(attn_masks[-1])
            # print(attn_masks[10])
            # print(attn_masks[0])
            # save_attention_masks(attn_masks)

            # concat optionals that arrived as lists
            if isinstance(t, list):
                t = torch.cat(t, 0)
            if isinstance(context_features, list):
                context_features = torch.cat(context_features, 0)
            if isinstance(y, list):
                y = torch.cat(y, 0)
        
            batch_size = x_tokens.size(0)
            # print(x_tokens.shape)
            # print(freqs_cis_x.shape)
            # print(context_features.shape)
            # print(attn_masks.shape)
            
            
        else:
            hw = x.shape[-2:]
            batch_size = x.shape[0]
            current_dtype = x.dtype
            current_device = x.device
            x_tokens = self.x_embedder(x) # (N, T_img, D)
            freqs_cis_x = self.select_rope_freqs_img(hw, device=current_device).to(current_dtype) # (T_img, D_head//2)
            img_gw = hw[1] // self.patch_size # Get image grid width for context RoPE offset
            attn_masks = None

        # 2. Timestep & Optional Conditioning Embeddings
        if t is not None:
            c_mod = self.t_embedder(t.to(current_device), dtype=current_dtype)
        else:
            c_mod = None 
            
        if y is not None and self.y_embedder is not None:
            c_mod = c_mod + self.y_embedder(y.to(current_device, dtype=current_dtype))

        # 3. Context Processing (Embedding + Positional Info)
        freqs_cis_context = None # Default to None (used if learnable or none)
        context_final = None     # Will hold the final context tensor passed to core

        if context_features is not None or self.register_length > 0:
            # Embed context features if provided
            if context_features is not None:
                context_tokens = self.context_embedder(context_features.to(current_device, dtype=current_dtype)) if context_features is not None else None # (N, L_ctx, D)
            else:
                context_tokens =  torch.Tensor([]).type_as(x)

            # Prepend register tokens if used
            if self.register is not None:
                register_tokens = repeat(self.register, "1 ... -> b ...", b=batch_size)
                if context_tokens is not None:
                    context_final = torch.cat((register_tokens, context_tokens), dim=1)
                else:
                    context_final = register_tokens
            else:
                context_final = context_tokens # Use only context_tokens if no register

            if context_final is not None:
                 context_len = context_final.shape[1] # Total length (register + context)

                 # Apply CFG dropout if training
                 if self.training and self.using_cfg:
                     cfg_mask = (torch.rand((batch_size,), device=current_device) > 0.1).view(batch_size, 1, 1)
                     # Apply mask only to the original context part if register exists? Or whole thing? Apply to whole thing for simplicity.
                     context_final = torch.where(cfg_mask, context_final, self.context_embedder(self.context_null).repeat(batch_size, 1, 1))

                 # Apply Positional Encoding based on type
                 if self.context_pos_encoding_type == 'learnable':
                     if self.context_pos_embed is None:
                         raise ValueError("Learnable context_pos_embed not initialized for type 'learnable'")
                     if context_len > self.context_pos_embed.shape[1]:
                          raise ValueError(f"Total context sequence length ({context_len}) exceeds maximum learnable PE length ({self.context_pos_embed.shape[1]})")
                     # Add learnable PE
                     context_final = context_final + self.context_pos_embed[:, :context_len, :]

                 elif self.context_pos_encoding_type == 'cross_rope':
                     # Calculate RoPE frequencies dynamically for the context sequence
                     freqs_cis_context = self.calculate_rope_freqs_context(
                         img_gw=img_gw,
                         context_len=context_len,
                         device=current_device,
                         dtype=current_dtype # Ensure correct dtype
                     ) # (context_len, D_head // 2)
                     # Note: RoPE is applied *inside* the attention mechanism via freqs_cis

                 # elif self.context_pos_encoding_type == 'none':
                 #     pass # No positional encoding added/applied
        if masks_enc is not None and self.training:
            x_tokens = apply_masks(x_tokens, masks_enc)
            freqs_cis_x = apply_masks(freqs_cis_x.unsqueeze(0).repeat(x_tokens.shape[0], 1, 1), masks_enc)
            

        # 4. Core Transformer Computation
        if self.output_register:
            context = self.forward_core_with_concat(
                x=x_tokens,
                c_mod=c_mod,
                context=context_final, # Pass the processed context tensor
                freqs_cis_x=freqs_cis_x,
                freqs_cis_context=freqs_cis_context, # Pass context RoPE freqs (or None)
                detach=detach,
                return_img_tokens=return_img_tokens,
            )
            return context
        else:
            x_out, zs = self.forward_core_with_concat(
                x=x_tokens,
                c_mod=c_mod,
                context=context_final, # Pass the processed context tensor
                freqs_cis_x=freqs_cis_x,
                freqs_cis_context=freqs_cis_context, # Pass context RoPE freqs (or None)
                detach=detach,
                attn_masks=attn_masks
            )
            
            # 5. Unpatchify
            if isinstance(x, list):
                output_image_list = []
                for i, (bsz_s, bsz_e) in enumerate(bsz_list):
                    output_image = self.unpatchify(x_out[bsz_s:bsz_e, :seq_len_list[i]], hw=hw_list[i])
                    output_image_list.append(output_image)
                return output_image_list, zs
            else:
                output_image = self.unpatchify(x_out, hw=hw)
                return output_image, zs

# --- Model Instantiation Functions (add new arg) ---
def mmdit_d12(in_channels, input_size, patch_size, context_embedding_dim, **kwargs):
    return MMDiT(in_channels=in_channels, input_size=input_size, patch_size=patch_size, context_embedding_dim=context_embedding_dim, depth=12, rmsnorm=False, qk_norm='ln', swiglu=True, **kwargs)

def mmdit_d24(in_channels, input_size, patch_size, context_embedding_dim, **kwargs):
    return MMDiT(in_channels=in_channels, input_size=input_size, patch_size=patch_size, context_embedding_dim=context_embedding_dim, depth=24, rmsnorm=False, qk_norm='ln', swiglu=True, **kwargs)

MMDiT_models = {'mmdit_d12': mmdit_d12, 'mmdit_d24': mmdit_d24}

import os
import matplotlib.pyplot as plt
def save_attention_masks(
    attn_masks: torch.Tensor,
    out_dir: str = "./attn_masks",
    white_for_true: bool = True,
    dpi: int = 200,
):
    """
    attn_masks : (B, 1, L, L)  bool / 0-1 tensor
    Saves PNGs: attn_mask_000.png …
    """
    if attn_masks.ndim != 4 or attn_masks.size(1) != 1:
        raise ValueError("expect (B,1,L,L), got", attn_masks.shape)

    os.makedirs(out_dir, exist_ok=True)
    masks_np = attn_masks.squeeze(1).cpu().numpy().astype("float32")  # (B, L, L)

    cmap = "Greys" if white_for_true else "Greys_r"   # True→white by default

    for i, m in enumerate(masks_np):
        plt.figure(dpi=dpi)
        plt.imshow(m, cmap=cmap, vmin=0, vmax=1, origin="upper",
                   interpolation="nearest")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"attn_mask_{i:03d}.png"),
                    bbox_inches="tight", pad_inches=0)
        plt.close()
    print(f"Saved {len(masks_np)} PNGs to {os.path.abspath(out_dir)}")

# Example Usage (Illustrative)
if __name__ == '__main__':
    import traceback
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 # Example using float16

    # Config
    batch_size = 2; img_size = 64; patch_size = 4; in_channels = 3
    context_dim = 32; context_len = 128; register_len = 0
    max_grid = img_size // patch_size

    # Dummy data
    dummy_x = torch.randn(batch_size, in_channels, img_size, img_size, device=device, dtype=dtype)
    dummy_t = torch.randint(0, 1000, (batch_size,), device=device)
    dummy_context = torch.randn(batch_size, context_len, context_dim, device=device, dtype=dtype)

    print("--- Testing Context PE: learnable ---")
    try:
        model_learnable = mmdit_d12(
            in_channels=in_channels, input_size=img_size, patch_size=patch_size,
            context_embedding_dim=context_dim, max_context_seq_len=context_len,
            context_pos_encoding_type='learnable', # Use learnable PE
            register_length=register_len, pos_embed_max_size=max_grid,
            qkv_bias=True, qk_norm=None, dtype=dtype, device=device, using_cfg=False
        ).to(device)
        output, _ = model_learnable(dummy_x, dummy_t, context_features=dummy_context)
        print(f"Learnable PE Output shape: {output.shape}")
        assert model_learnable.context_pos_embed is not None
        print("Learnable PE test successful!")
    except Exception as e: print(f"Error: {e}"); traceback.print_exc()

    print("\n--- Testing Context PE: cross_rope ---")
    try:
        model_cross_rope = mmdit_d12(
            in_channels=in_channels, input_size=img_size, patch_size=patch_size,
            context_embedding_dim=context_dim, max_context_seq_len=context_len,
            context_pos_encoding_type='cross_rope', # Use cross-modality RoPE
            register_length=register_len, pos_embed_max_size=max_grid,
            qkv_bias=True, qk_norm=None, dtype=dtype, device=device, using_cfg=False
        ).to(device)
        output, _ = model_cross_rope(dummy_x, dummy_t, context_features=dummy_context)
        print(f"Cross RoPE Output shape: {output.shape}")
        assert model_cross_rope.context_pos_embed is None # Should not exist
        print("Cross RoPE test successful!")
    except Exception as e: print(f"Error: {e}"); traceback.print_exc()

    print("\n--- Testing Context PE: none ---")
    try:
        model_none = mmdit_d12(
            in_channels=in_channels, input_size=img_size, patch_size=patch_size,
            context_embedding_dim=context_dim, max_context_seq_len=context_len,
            context_pos_encoding_type='none', # Use no explicit PE for context
            register_length=register_len, # Registers still exist but get no explicit PE here
            pos_embed_max_size=max_grid,
            qkv_bias=True, qk_norm=None, dtype=dtype, device=device, using_cfg=False
        ).to(device)
        output, _ = model_none(dummy_x, dummy_t, context_features=dummy_context)
        print(f"No Context PE Output shape: {output.shape}")
        assert model_none.context_pos_embed is None
        print("No Context PE test successful!")
    except Exception as e: print(f"Error: {e}"); traceback.print_exc()