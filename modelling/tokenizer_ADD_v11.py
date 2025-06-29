# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
import math
from einops import rearrange
import sys
sys.path.append('/home/haoc/storage/rt')
from modelling.modules import Encoder, Decoder, TimmViTEncoder, TimmViTDecoder
from modelling.quantizers.vq import VectorQuantizer
from modelling.quantizers.kl import DiagonalGaussianDistribution
from modelling.quantizers.softvq import SoftVectorQuantizer
from modelling.modules.jepa_vit.vision_transformer import vit_predictor

from modelling.sit_decoder import SiT_models
from losses.sit_loss import SILoss
from losses.ms_sit_loss_ADD import MultiScaleSILoss
from modelling.samplers_ADD import euler_maruyama_sampler, multiscale_euler_sampler


from modelling.mmdit_decoder import MMDiT_models as MMDiT_models_v0
from modelling.mmdit_decoder_rope_ADD import MMDiT_models 

from timm import create_model

from masks.utils import apply_masks


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.shape))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


@dataclass
class ModelArgs:
    image_size: int = 256
    base_image_size: int = 256
    
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    vq_loss_ratio: float = 1.0 # for soft vq
    kl_loss_weight: float = 0.000001
    tau: float = 0.1
    num_codebooks: int = 1
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    enc_type: str = 'cnn'
    dec_type: str = 'cnn'
    encoder_model: str = 'llamagen_encoder'
    decoder_model: str = 'llamagen_decoder'
    num_latent_tokens: int = 256
    to_pixel: str = 'linear'
    
    # for pre-trained models
    enc_tuning_method: str = 'full'
    dec_tuning_method: str = 'full'
    enc_pretrained: bool = True
    dec_pretrained: bool = False 
    
    # for vit 
    enc_patch_size: int = 16
    dec_patch_size: int = 16
    enc_drop_path_rate: float = 0.0
    dec_drop_path_rate: float = 0.0

    # encoder token drop
    # NOTE: not used
    enc_token_drop: float = 0.1
    enc_token_drop_max: float = 0.1
    
    # deocder cls token
    enc_cls_token: bool = True
    dec_cls_token: bool = True
    
    # rope
    use_ape: bool = True 
    use_rope: bool = False
    rope_mixed: bool = False
    rope_theta: float = 10.0
    
    # repa for vit
    repa: bool = False
    repa_patch_size: int = 16
    repa_model: str = 'vit_base_patch16_224'
    repa_proj_dim: int = 2048
    repa_loss_weight: float = 0.1
    repa_align: str = 'global'
    
    # jepa 
    predictor_depth: int = 12
    predictor_embed_dim: int = 384
    
    vq_mean: float = 0.0
    vq_std: float = 1.0
    
    # diffusion decoder
    diff_dec_use_adaptive_ln_latent: bool = False
    perceptual_type: str = 'vgg_lpips'
    perceptual_loss_weight: float = 0.1
    using_cfg: bool = False
    
    # for mmdit
    latent_pos_encoding_type: str = 'none'
    pos_embed_max_size: int = None
    multiscale_decoder: bool = False
    num_multiscale_stages: int = 3
    multiscale_perceptual_start: int = 0
    sigmoid_weighting: bool = False
    flow_weighting: str = 'uniform'
    

class VQModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align
        if config.repa and config.enc_type == 'vit':
            self.repa_model = create_model(config.repa_model, pretrained=True, img_size=config.image_size, patch_size=config.repa_patch_size)
            for param in self.repa_model.parameters():
                param.requires_grad = False
            self.repa_model.eval()
            repa_z_dim = self.repa_model.embed_dim
            self.repa_z_dim = repa_z_dim
            self.projection = build_mlp(config.codebook_embed_dim, config.repa_proj_dim, repa_z_dim)
            from modelling.lpips.lpips_timm import Normalize, Denormalize
            self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            repa_z_dim = None
        
        
        if config.enc_type == 'cnn':
            if config.encoder_model == 'llamagen_encoder':
                self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        elif config.enc_type == 'vit':
            self.encoder = TimmViTEncoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.encoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size, 'drop_path_rate': config.enc_drop_path_rate},
                pretrained=config.enc_pretrained,
                tuning_method=config.enc_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                token_drop=config.enc_token_drop,
                token_drop_max=config.enc_token_drop_max,
                base_img_size=config.base_image_size,
                cls_token=config.enc_cls_token,
            )
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)
            
        
        if config.dec_type == 'cnn':
            if config.decoder_model == 'llamagen_decoder':
                self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        elif config.dec_type == 'vit':
            self.decoder = TimmViTDecoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.decoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': config.dec_drop_path_rate, 'latent_dim': config.codebook_embed_dim},
                pretrained=config.dec_pretrained,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel=config.to_pixel,
                base_img_size=config.base_image_size
            )
            self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
        # check movq
        if 'movq' in config.decoder_model:
            self.use_movq = True 
        else:
            self.use_movq = False
        
        
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        
        if self.repa and self.training:
            # get z from repa_encoder
            rescale_x = self.scale(self.de_scale(x))
            z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

            # taking average over spatial dimension
            if self.repa_align == 'global':
                z = z.mean(dim=1)
                z_hat = quant.mean(dim=1)
                # calculate repa loss
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d':
                z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d_shuffle':
                # shuffle the length dimension of z and avg
                indices = torch.randperm(z.shape[1])
                z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1) , quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'repeat':
                z_hat = self.projection(quant)
                b, l, d = z_hat.shape
                z_hat = z_hat.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
            

            z = F.normalize(z, dim=-1)
            z_hat = F.normalize(z_hat, dim=-1)
            proj_loss = mean_flat(-(z * z_hat).sum(dim=-1))
            proj_loss = proj_loss.mean()
            proj_loss *= self.repa_loss_weight
            
            emb_loss += (proj_loss,)
        
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        b, _, h, w = input.shape
        quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, x=input, h=h, w=w)
        return dec, diff, info


class SoftVQModel(VQModel, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = SoftVectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                            config.entropy_loss_ratio, 
                                            config.tau,                                   
                                            config.num_codebooks,
                                            config.codebook_l2_norm, config.codebook_show_usage)


class KLModel(VQModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.kl_loss_weight = config.kl_loss_weight
        self.quantize = None
        
        if config.enc_type == 'cnn':
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim * 2, 1)
        elif config.enc_type == 'vit':
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim * 2)
        

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h)
        h_posterior = DiagonalGaussianDistribution(h)
        return h_posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_code(self, posterior, shape=None):
        z = posterior.sample()
        dec = self.decode(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # compute kl loss
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        diff = (kl_loss * self.kl_loss_weight, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        return dec, diff, None


class AEModel(VQModel):
    def __init__(self, config: ModelArgs,
                tags=["arxiv:xxx", "image-generation", "1d-tokenizer", "128 tokens", "MAETok"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = None 


    def encode(self, x):
        
        h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info

    def decode(self, ):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec
    

class AEJEPAModel(VQModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.quantize = None 
        self.predictor = vit_predictor(
            num_patches=self.encoder.num_img_tokens,
            num_heads=self.encoder.model.num_heads,
            depth=config.predictor_depth,
            predictor_embed_dim=config.predictor_embed_dim,
            # linear predict embed dim
            embed_dim=config.codebook_embed_dim,
            num_latent_tokens=config.num_latent_tokens,
        )
        self.encoder.model.dynamic_img_size = True

    def encode(self, x, masks_enc=None, return_img_tokens=False):
        
        h = self.encoder(x, masks_enc, return_img_tokens)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None, masks_enc=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w, masks_enc)
        else:
            dec = self.decoder(quant, None, h, w, masks_enc)
        return dec
    

    def forward(self, input, masks_enc=None, masks_pred=None):
        b, _, h, w = input.shape
        quant, diff, info = self.encode(input, masks_enc)
        
        self.quant = quant
        dec = self.decode(quant, x=input, h=h, w=w, masks_enc=masks_enc)
        
        # jepa related
        if self.training:
            # predictor
            # quant = apply_masks(quant, masks_enc)
            pred_z = self.predictor(quant, masks_enc, masks_pred)
            
            info = {'pred_z': pred_z}
        return dec, diff, info


class AEDiffModel(AEModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        # block_kwargs = {"fused_attn": False, "qk_norm": False, "flash_attn": True}
        # self.decoder = SiT_models[config.decoder_model](
        #         image_size=config.image_size,
        #         patch_size=config.dec_patch_size,
        #         in_channels=3,
        #         latent_channels=config.codebook_embed_dim,
        #         num_latent_tokens=config.num_latent_tokens if config.num_latent_tokens > 0 else self.encoder.num_img_tokens,
        #         use_adaptive_ln_latent=config.diff_dec_use_adaptive_ln_latent,
        #         **block_kwargs,
        #     )
        
        # self.decoder = DDT(
        #     in_channels=3,
        #     patch_size=config.dec_patch_size,
        #     latent_channels=config.codebook_embed_dim,
        # )
        
        # support more encoder types
        if config.enc_type == 'mmdit_v0':
            self.encoder = MMDiT_models_v0[config.decoder_model](
                in_channels=3,
                input_size=config.image_size,
                patch_size=config.enc_patch_size,
                using_cfg=False,
                output_register=True,
                register_length=config.num_latent_tokens,
                out_channels=config.codebook_embed_dim,
                # qk_norm='rms',
            )
            self.quant_conv = self.encoder.final_layer
        elif config.enc_type == 'mmdit':
            self.encoder = MMDiT_models[config.decoder_model](
                in_channels=3,
                input_size=config.image_size,
                patch_size=config.enc_patch_size,
                using_cfg=False,
                output_register=True,
                context_embedding_dim=config.codebook_embed_dim,
                register_length=config.num_latent_tokens,
                out_channels=config.codebook_embed_dim,
                context_pos_encoding_type=config.latent_pos_encoding_type,
                pos_embed_max_size=config.pos_embed_max_size
                # qk_norm='rms',
            )
            self.quant_conv = self.encoder.final_layer
        
        # support more decoder types
        if config.dec_type == 'sit':
            block_kwargs = {"fused_attn": False, "qk_norm": False, "flash_attn": True}
            self.decoder = SiT_models[config.decoder_model](
                    image_size=config.image_size,
                    patch_size=config.dec_patch_size,
                    in_channels=3,
                    latent_channels=config.codebook_embed_dim,
                    num_latent_tokens=config.num_latent_tokens if config.num_latent_tokens > 0 else self.encoder.num_img_tokens,
                    use_adaptive_ln_latent=config.diff_dec_use_adaptive_ln_latent,
                    **block_kwargs,
                )
        elif config.dec_type == 'mmdit_v0':
            self.decoder = MMDiT_models_v0[config.decoder_model](
                in_channels=3,
                input_size=config.image_size,
                patch_size=config.dec_patch_size,
                using_cfg=config.using_cfg,
            )
        elif config.dec_type == 'mmdit':
            self.decoder = MMDiT_models[config.decoder_model](
                in_channels=3,
                input_size=config.image_size,
                patch_size=config.dec_patch_size,
                using_cfg=config.using_cfg,
                context_pos_encoding_type=config.latent_pos_encoding_type,
                context_embedding_dim=config.codebook_embed_dim, 
                max_context_seq_len=config.num_latent_tokens if config.num_latent_tokens > 0 else self.encoder.num_img_tokens,
                pos_embed_max_size=config.pos_embed_max_size
            )
        
        # self.decoder = MMDiT(
        #     input_size=config.image_size,
        #     patch_size=config.dec_patch_size,
        #     in_channels=3,
        #     depth=12,
        #     rmsnorm=True,
        #     swiglu=True,
        # )
        self.multiscale_decoder = config.multiscale_decoder
        self.num_multiscale_stages = config.num_multiscale_stages
        if self.multiscale_decoder:
            self.sit_loss = MultiScaleSILoss(
                    prediction='v',
                    path_type='linear', 
                    weighting=config.flow_weighting,
                    perceptual_type=config.perceptual_type,
                    perceptual_loss_weight=config.perceptual_loss_weight,
                    num_stages=config.num_multiscale_stages,
                    perceptual_start_stage=config.multiscale_perceptual_start,
                    sigmoid_weighting=config.sigmoid_weighting,
            )
        else:
            self.sit_loss = SILoss(
                    prediction='v',
                    path_type='linear', 
                    encoders=[],
                    weighting=config.flow_weighting,
                    perceptual_type=config.perceptual_type,
                    perceptual_loss_weight=config.perceptual_loss_weight,
                    sigmoid_weighting=config.sigmoid_weighting,
                )
            
        print("Tokenizer encoder: ", config.encoder_model)
        print("Tokenizer encoder type: ", config.enc_type)
        print("Diffusion decoder: ", config.decoder_model)
        print("Diffusion decoder type: ", config.dec_type)
        print("Diffusion decoder perceptual type: ", config.perceptual_type)
        print("Diffusion decoder perceptual loss weight: ", config.perceptual_loss_weight)
        print("Diffusion decoder using cfg: ", config.using_cfg)
        print("Diffusion decoder latent pos encoding type: ", config.latent_pos_encoding_type)
        

    def encode(self, x):
        
        h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info


    def decode_train(self, quant, x, masks_enc=None):
        x_end_student_list, x_student = self.sit_loss.forward_student(self.decoder, x, {'context_features':quant, 'masks_enc':masks_enc}, zs=[])
        x_end_teacher_list, start_stage = self.sit_loss.forward_teacher(self.decoder, x_student, {'context_features':quant, 'masks_enc':masks_enc}, zs=[])
        return x_end_student_list, x_end_teacher_list, start_stage
    
    @torch.no_grad
    def decode(self, quant, num_steps=1, cfg_scale=1.0, guidance_low=0., guidance_high=1., clip=False):
        if self.multiscale_decoder:
            init_factor = 2 ** (self.num_multiscale_stages - 1)
            image_size = self.config.image_size // init_factor
            xT = torch.randn(quant.size(0), 3, image_size, image_size, device=quant.device, dtype=quant.dtype)
            # shift = 1.0
            
            for stage_idx in range(self.num_multiscale_stages):
                # set the number of inference steps
                timestep_start = self.sit_loss.timesteps_per_stage[stage_idx][0].item()
                timestep_end = self.sit_loss.timesteps_per_stage[stage_idx][-1].item()
                timesteps = np.linspace(timestep_start, timestep_end, num_steps)
                timesteps = torch.from_numpy(timesteps).to(device=quant.device, dtype=quant.dtype) / 1000.0
                
                t_start = self.sit_loss.t_per_stage[stage_idx][0].item()
                t_end = self.sit_loss.t_per_stage[stage_idx][-1].item()
                t = np.linspace(t_start, t_end, num_steps)
                t = np.append(t, 0.0)
                t = torch.from_numpy(t).to(device=quant.device, dtype=quant.dtype)
                
                # # TODO: check this
                # t = t / (shift  + (1 - shift) * t)
                
                # # linearly map t to T: T = k * t + b
                # k = (stage_timestep_end - stage_timestep_start) / (t_end - t_start)
                # b = stage_timestep_start - t_start * k
                # timesteps = k * t + b

                if stage_idx > 0:
                    image_size = image_size * 2
                    xT = F.interpolate(xT, size=(image_size, image_size), mode='nearest').to(quant.dtype)
                    
                    orig_start_t = 1 - self.sit_loss.orig_start_t[stage_idx]
                    gamma = self.sit_loss.gamma
                    alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - orig_start_t) + orig_start_t)
                    beta = alpha * (1 - orig_start_t) / math.sqrt(gamma)

                    # bs, ch, height, width = latents.shape
                    noise = self.sample_block_noise(*xT.shape)
                    noise = noise.to(device=xT.device, dtype=xT.dtype)
                    xT = alpha * xT + beta * noise

                # print(f"stage_idx {stage_idx}")
                # print(f"t {t}")
                # print(f"timesteps {timesteps}")
                # print(f"xT {xT.shape}")
                
                samples = multiscale_euler_sampler(
                    self.decoder,
                    xT, 
                    quant,
                    t,
                    timesteps,
                    cfg_scale=cfg_scale,
                    guidance_low=guidance_low,
                    guidance_high=guidance_high,
                    path_type='linear',
                    heun=False,
                )
                xT = samples
        else:
            xT = torch.randn(quant.size(0), 3, self.config.image_size, self.config.image_size, device=quant.device)
            samples = euler_maruyama_sampler(
                self.decoder, 
                xT, 
                quant,
                num_steps=num_steps, 
                cfg_scale=cfg_scale,
                guidance_low=guidance_low,
                guidance_high=guidance_high,
                path_type='linear',
                heun=False,
                clip=clip,
                )
        return samples.to(torch.float32)


    def decode_teacher(self, input, input_student, num_steps=1, cfg_scale=1.0, guidance_low=0., guidance_high=1., clip=False):
        b, _, h, w = input.shape
        quant, diff, info = self.encode(input)
        x_end_teacher_list = list()
        # print("self.multiscale_decoder: ", self.multiscale_decoder)
        if self.multiscale_decoder:
            u = torch.rand(1, device=quant.device).expand(quant.size(0))
            start_stage = self.num_multiscale_stages - 1  # u >= end_t[0] 时默认从最早的 stage 开始
            for i in range(1, self.num_multiscale_stages):
                # print("self.sit_loss.timesteps_per_stage[i - 1][0]: ", self.sit_loss.timesteps_per_stage[i - 1][0])
                # print("self.sit_loss.timesteps_per_stage[i][0]: ", self.sit_loss.timesteps_per_stage[i][0])
                # print("u[0] * 1000: ", u[0] * 1000)
                if self.sit_loss.timesteps_per_stage[i - 1][0].item() >= u[0] * 1000 > self.sit_loss.timesteps_per_stage[i][0].item():
                    start_stage = i - 1
                    break
            
            init_factor = 2 ** (self.num_multiscale_stages - start_stage - 1)
            image_size = self.config.image_size // init_factor
            xT = input_student
            # print(f"x_end start {x_end.shape}")
            for d in range(1, self.num_multiscale_stages - start_stage):
                # print(f"x_end down {d}")
                xT = F.interpolate(xT, size=(h // (2 ** d), w // (2 ** d)), mode="bilinear")
            noises = torch.randn_like(xT)
            # xT = u.view(-1,1,1,1) * xT + (1 - u).view(-1,1,1,1) * noises
            xT = (1-u).view(-1,1,1,1) * xT + u.view(-1,1,1,1) * noises

            for stage_idx in range(start_stage, self.num_multiscale_stages):
                # set the number of inference steps
                timestep_start = self.sit_loss.timesteps_per_stage[stage_idx][0].item()
                timestep_end = self.sit_loss.timesteps_per_stage[stage_idx][-1].item()
                timesteps = np.linspace(timestep_start, timestep_end, num_steps)
                timesteps = torch.from_numpy(timesteps).to(device=quant.device, dtype=quant.dtype) / 1000.0
                
                t_start = self.sit_loss.t_per_stage[stage_idx][0].item()
                t_end = self.sit_loss.t_per_stage[stage_idx][-1].item()
                t = np.linspace(t_start, t_end, num_steps)
                t = np.append(t, 0.0)
                t = torch.from_numpy(t).to(device=quant.device, dtype=quant.dtype)
                
                if stage_idx > start_stage:
                    image_size = image_size * 2
                    xT = F.interpolate(xT, size=(image_size, image_size), mode='nearest').to(quant.dtype)
                    
                    orig_start_t = 1 - self.sit_loss.orig_start_t[stage_idx]
                    gamma = self.sit_loss.gamma
                    alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - orig_start_t) + orig_start_t)
                    beta = alpha * (1 - orig_start_t) / math.sqrt(gamma)

                    # bs, ch, height, width = latents.shape
                    noise = self.sample_block_noise(*xT.shape)
                    noise = noise.to(device=xT.device, dtype=xT.dtype)
                    xT = alpha * xT + beta * noise
                
                samples = multiscale_euler_sampler(
                    self.decoder,
                    xT, 
                    quant,
                    t,
                    timesteps,
                    cfg_scale=cfg_scale,
                    guidance_low=guidance_low,
                    guidance_high=guidance_high,
                    path_type='linear',
                    heun=False,
                    is_teacher=True
                )
                x_end_teacher_list.append(samples)
                xT = samples
        else:
            xT = torch.randn(quant.size(0), 3, self.config.image_size, self.config.image_size, device=quant.device)
            samples = euler_maruyama_sampler(
                self.decoder, 
                xT, 
                quant,
                num_steps=num_steps, 
                cfg_scale=cfg_scale,
                guidance_low=guidance_low,
                guidance_high=guidance_high,
                path_type='linear',
                heun=False,
                clip=clip,
                )
        return x_end_teacher_list, start_stage, u, diff, info


    def decode_student(self, input, num_steps=1, cfg_scale=1.0, guidance_low=0., guidance_high=1., clip=False):
        b, _, h, w = input.shape
        quant, diff, info = self.encode(input)
        x_end_student_list = list()

        if self.multiscale_decoder:
            init_factor = 2 ** (self.num_multiscale_stages - 1)
            image_size = self.config.image_size // init_factor

            # t in [1, 0.75, 0.5, 0.25] # start t
            # forward input add noise # stage start xT 
            # 3 - T, backward
            # teacher xT - 1

            xT = torch.randn(quant.size(0), 3, image_size, image_size, device=quant.device, dtype=quant.dtype)
            # shift = 1.0
            
            for stage_idx in range(self.num_multiscale_stages):
                # set the number of inference steps
                timestep_start = self.sit_loss.timesteps_per_stage[stage_idx][0].item()
                timestep_end = self.sit_loss.timesteps_per_stage[stage_idx][-1].item()
                timesteps = np.linspace(timestep_start, timestep_end, num_steps)
                timesteps = torch.from_numpy(timesteps).to(device=quant.device, dtype=quant.dtype) / 1000.0
                print("timesteps: ", timesteps)
                
                t_start = self.sit_loss.t_per_stage[stage_idx][0].item()
                t_end = self.sit_loss.t_per_stage[stage_idx][-1].item()
                t = np.linspace(t_start, t_end, num_steps)
                t = np.append(t, 0.0)
                t = torch.from_numpy(t).to(device=quant.device, dtype=quant.dtype)
                # print("t: ", t)
                
                if stage_idx > 0:
                    image_size = image_size * 2
                    xT = F.interpolate(xT, size=(image_size, image_size), mode='nearest').to(quant.dtype)
                    
                    orig_start_t = 1 - self.sit_loss.orig_start_t[stage_idx]
                    gamma = self.sit_loss.gamma
                    alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - orig_start_t) + orig_start_t)
                    beta = alpha * (1 - orig_start_t) / math.sqrt(gamma)

                    # bs, ch, height, width = latents.shape
                    noise = self.sample_block_noise(*xT.shape)
                    noise = noise.to(device=xT.device, dtype=xT.dtype)
                    xT = alpha * xT + beta * noise
                
                samples = multiscale_euler_sampler(
                    self.decoder,
                    xT, 
                    quant,
                    t,
                    timesteps,
                    cfg_scale=cfg_scale,
                    guidance_low=guidance_low,
                    guidance_high=guidance_high,
                    path_type='linear',
                    heun=False,
                    is_teacher=False
                )
                # _dtype = xT.dtype    
                # samples = xT # .to(torch.float64)
                # device = samples.device
                # for i, timestep in enumerate(timesteps):
                #     x_cur = samples
                #     model_input = x_cur
                #     y_cur = quant

                #     kwargs = dict(context_features=y_cur)
                #     time_input = torch.ones(model_input.size(0)).to(device=device) * timestep
                #     d_cur = self.decoder(
                #         model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                #         )[0] #.to(torch.float64) 
                #     t_cur = t[i]                      
                #     t_next = t[i + 1]
                #     samples = x_cur + (t_next - t_cur) * d_cur

                x_end_student_list.append(samples)
                xT = samples
        else:
            xT = torch.randn(quant.size(0), 3, self.config.image_size, self.config.image_size, device=quant.device)
            samples = euler_maruyama_sampler(
                self.decoder, 
                xT, 
                quant,
                num_steps=num_steps, 
                cfg_scale=cfg_scale,
                guidance_low=guidance_low,
                guidance_high=guidance_high,
                path_type='linear',
                heun=False,
                clip=clip,
                )
        return x_end_student_list, samples.to(torch.float32), diff, info


    def forward(self, input):
        b, _, h, w = input.shape
        quant, diff, info = self.encode(input)
        
        self.quant = quant
        x_end_student_list, x_end_teacher_list, start_stage  = self.decode_train(quant, input)
        # diff  = (denois_loss, lpips_loss, proj_loss)
        
        return x_end_student_list, x_end_teacher_list, start_stage, diff, info
    
    def forward_teacher(self, input, x_student):
        b, _, h, w = input.shape
        quant, diff, info = self.encode(input)
        
        self.quant = quant
        x_end_teacher_list, start_stage, u = self.sit_loss.forward_teacher(self.decoder, x_student, {'context_features':quant, 'masks_enc':None}, zs=[])
        # diff  = (denois_loss, lpips_loss, proj_loss)
        
        return x_end_teacher_list, start_stage, u, diff, info
    
    def forward_student(self, input):
        b, _, h, w = input.shape
        quant, diff, info = self.encode(input)
        
        self.quant = quant
        x_end_student_list, x_student = self.sit_loss.forward_student(self.decoder, input, {'context_features':quant, 'masks_enc':None}, zs=[])
        # diff  = (denois_loss, lpips_loss, proj_loss)
        
        return x_end_student_list, x_student, diff, info

    def sample_block_noise(self, bs, ch, height, width, eps=1e-6):
        gamma = self.sit_loss.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 - gamma) + torch.ones(4, 4) * gamma + eps * torch.eye(4))
        block_number = bs * ch * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c h w) (p q) -> b c (h p) (w q)',b=bs,c=ch,h=height//2,w=width//2,p=2,q=2)
        return noise
    

class VQDiffModel(AEDiffModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)


    def encode(self, x):
        
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        # emb loss: vq loss, commit_loss, entropy_loss, code_usage
        return quant, emb_loss, info
    
    def forward(self, input):
        b, _, h, w = input.shape
        quant, diff, info = self.encode(input)
        
        self.quant = quant
        dec_loss, denois_loss, lpips_loss, proj_loss  = self.decode_train(quant, input)
        # emb loss: vq loss, commit_loss, entropy_loss, code_usage
        dec_loss += (diff[0] + diff[1] + diff[2])
        diff  = (denois_loss, lpips_loss, proj_loss) + diff
        
        
        return dec_loss, diff, info

    

class AEJEPADiffModel(AEModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.quantize = None 

        # support more encoder types
        if config.enc_type == 'mmdit_v0':
            self.encoder = MMDiT_models_v0[config.decoder_model](
                in_channels=3,
                input_size=config.image_size,
                patch_size=config.enc_patch_size,
                using_cfg=False,
                output_register=True,
                register_length=config.num_latent_tokens,
                out_channels=config.codebook_embed_dim,
                # qk_norm='rms',
            )
            self.quant_conv = self.encoder.final_layer
        elif config.enc_type == 'mmdit':
            self.encoder = MMDiT_models[config.decoder_model](
                in_channels=3,
                input_size=config.image_size,
                patch_size=config.enc_patch_size,
                using_cfg=False,
                output_register=True,
                context_embedding_dim=config.codebook_embed_dim,
                register_length=config.num_latent_tokens,
                out_channels=config.codebook_embed_dim,
                context_pos_encoding_type=config.latent_pos_encoding_type,
                # qk_norm='rms',
            )
            self.quant_conv = self.encoder.final_layer
        
        # support more decoder types
        if config.dec_type == 'sit':
            block_kwargs = {"fused_attn": False, "qk_norm": False, "flash_attn": True}
            self.decoder = SiT_models[config.decoder_model](
                    image_size=config.image_size,
                    patch_size=config.dec_patch_size,
                    in_channels=3,
                    latent_channels=config.codebook_embed_dim,
                    num_latent_tokens=config.num_latent_tokens if config.num_latent_tokens > 0 else self.encoder.num_img_tokens,
                    use_adaptive_ln_latent=config.diff_dec_use_adaptive_ln_latent,
                    **block_kwargs,
                )
        elif config.dec_type == 'mmdit_v0':
            self.decoder = MMDiT_models_v0[config.decoder_model](
                in_channels=3,
                input_size=config.image_size,
                patch_size=config.dec_patch_size,
                using_cfg=config.using_cfg,
            )
        elif config.dec_type == 'mmdit':
            self.decoder = MMDiT_models[config.decoder_model](
                in_channels=3,
                input_size=config.image_size,
                patch_size=config.dec_patch_size,
                using_cfg=config.using_cfg,
                context_pos_encoding_type=config.latent_pos_encoding_type,
                context_embedding_dim=config.codebook_embed_dim, 
                max_context_seq_len=config.num_latent_tokens if config.num_latent_tokens > 0 else self.encoder.num_img_tokens,
            )
            
        self.sit_loss = SILoss(
            prediction='v',
            path_type='linear', 
            encoders=[],
            weighting='lognormal',
            perceptual_loss_weight=0.1,
        )

        self.predictor = vit_predictor(
            num_patches=self.encoder.num_img_tokens,
            num_heads=12, # self.encoder.model.num_heads,
            depth=config.predictor_depth,
            predictor_embed_dim=config.predictor_embed_dim,
            # linear predict embed dim
            embed_dim=config.codebook_embed_dim,
            num_latent_tokens=config.num_latent_tokens,
        )
        # self.encoder.model.dynamic_img_size = True

    def encode(self, x, masks_enc=None, return_img_tokens=False):
        
        h = self.encoder(x, masks_enc=masks_enc, return_img_tokens=return_img_tokens)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        return quant, emb_loss, info

    def decode_train(self, quant, x, masks_enc=None):
        dec_loss, denois_loss, lpips_loss, proj_loss = self.sit_loss(self.decoder, x, {'context_features':quant, 'masks_enc':masks_enc}, zs=[])
        return dec_loss, denois_loss, lpips_loss, proj_loss 
    
    @torch.no_grad
    def decode(self, quant, num_steps=50, cfg_scale=1.0, guidance_low=0., guidance_high=1.):
        xT = torch.randn(quant.size(0), 3, self.config.image_size, self.config.image_size, device=quant.device)
        samples = euler_maruyama_sampler(
            self.decoder, 
            xT, 
            quant,
            num_steps=num_steps, 
            cfg_scale=cfg_scale,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            path_type='linear',
            heun=False,
            ).to(torch.float32)
        return samples

    def forward(self, input, masks_enc=None, masks_pred=None):
        b, _, h, w = input.shape
        quant, diff, info = self.encode(input, masks_enc)
        
        
        self.quant = quant
        if self.config.num_latent_tokens > 0:
            dec_loss, denois_loss, lpips_loss, proj_loss  = self.decode_train(quant, input)
        else:
            dec_loss, denois_loss, lpips_loss, proj_loss  = self.decode_train(quant, input, masks_enc)
    
        # jepa related
        if self.training:
            # predictor
            pred_z = self.predictor(quant, masks_enc, masks_pred)
            
            info = {'pred_z': pred_z}
        
        diff  = (denois_loss, lpips_loss, proj_loss)
            
        return dec_loss, diff, info



#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def KL_8(**kwargs):
    return KLModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def KL_16(**kwargs):
    return KLModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AE_16(**kwargs):
    return AEModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AEJEPA_16(**kwargs):
    return AEJEPAModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AEDiff_16(**kwargs):
    return AEDiffModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def VQDiff_16(**kwargs):
    return VQDiffModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AEJEPADiff_16(**kwargs):
    return AEJEPADiffModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def SoftVQ(**kwargs):
    return SoftVQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


VQ_models = {
    'AE-16': AE_16,
    'AE-Diff-16': AEDiff_16,
    'VQ-Diff-16': VQDiff_16,
    'AE-JEPA-16': AEJEPA_16,
    'AE-JEPA-Diff-16': AEJEPADiff_16,
    'VQ-16': VQ_16, 'VQ-8': VQ_8,
    'KL-16': KL_16, 'KL-8': KL_8,
    'SoftVQ': SoftVQ,
    }


if __name__ == '__main__':
    
    # model = VQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', repa=True, repa_model='vit_base_patch14_dinov2.lvd142m', repa_align='avg_1d_shuffle', enc_img_res=True, enc_img_align='avg_1d', dec_img_res=True)    
    # model = SoftVQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384, topk=16)
    # model = AE_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384)
    model = AEDiff_16(codebook_embed_dim=32, enc_type='mmdit', dec_type='mmdit', encoder_model='mmdit_d12', decoder_model='mmdit_d12', num_latent_tokens=16, multiscale_decoder=True, pos_embed_max_size=32)
    model = model.cuda()
    model.train()
    # model = KL_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    # model = GMM_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    x = torch.randn(16, 3, 512, 512).cuda()
    x = x.to(torch.bfloat16)
    model = model.to(torch.bfloat16)
    y, _, info = model(x)
    
    # test sampling
    quant, _, _ = model.encode(x)
    rec_x = model.decode(quant, num_steps=50)

    # token_mask = torch.tensor([
    #     [1, 1, 1, 0],
    #     [1, 1, 0, 0]
    # ], dtype=torch.bool)  # (B=2, T=4)

    # attn_mask = token_mask.unsqueeze(1) & token_mask.unsqueeze(2)  # (B, 1, T, T)

    # print(attn_mask)  # torch.Size([2, 1, 4, 4])