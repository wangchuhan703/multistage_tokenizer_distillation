import yaml

import torch
from modelling.tokenizer import VQ_models



def build_tokenizer(vq_config,
                    vq_ckpt):
    
    with open(vq_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config_name = vq_config.split('/')[-2]
    exp_index = int(config_name.split('-')[0][3:])

    vae = VQ_models[config['vq_model']](
        image_size=config['image_size'],
        codebook_size=config['codebook_size'],
        codebook_embed_dim=config['codebook_embed_dim'],
        codebook_l2_norm=config['codebook_l2_norm'],
        commit_loss_beta=config['commit_loss_beta'],
        entropy_loss_ratio=config['entropy_loss_ratio'],
        vq_loss_ratio=config['vq_loss_ratio'],
        kl_loss_weight=config['kl_loss_weight'],
        dropout_p=config['dropout_p'],
        enc_type=config['enc_type'],
        encoder_model=config['encoder_model'],
        dec_type=config['dec_type'],
        decoder_model=config['decoder_model'],
        num_latent_tokens=config['num_latent_tokens'],
        enc_tuning_method=config['encoder_tuning_method'],
        dec_tuning_method=config['decoder_tuning_method'],
        enc_pretrained=config['encoder_pretrained'],
        dec_pretrained=config['decoder_pretrained'],
        enc_patch_size=config['encoder_patch_size'],
        dec_patch_size=config['decoder_patch_size'],
        enc_cls_token=config['encoder_cls_token'],
        dec_cls_token=config['decoder_cls_token'],
        tau=config.get('tau', 1.0),
        repa=config.get('repa', False),
        repa_model=config.get('repa_model', None),
        repa_patch_size=config.get('repa_patch_size', None),
        repa_proj_dim=config.get('repa_proj_dim', None),
        repa_loss_weight=config.get('repa_loss_weight', 0.0),
        repa_align=config.get('repa_align', False),
        num_codebooks=config.get('num_codebooks', 1),
        perceptual_type=config.get('perceptual_type', "vgg_lpips"),
        perceptual_loss_weight=config.get('perceptual_loss_weight', 0.0),
        using_cfg=config.get('using_cfg', False),
        latent_pos_encoding_type=config.get('latent_pos_embed_type', 'cross-rope'),
        pos_embed_max_size=config.get('pos_embed_max_size', 16),
        multiscale_decoder=config.get('multiscale_decoder', False),
        num_multiscale_stages=config.get('num_multiscale_stages', 1)
    )

    # vq_model.to(device)
    # vq_model.eval()
    checkpoint = torch.load(vq_ckpt, map_location="cpu", weights_only=False)
    model_weight = checkpoint['model']
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    keys = vae.load_state_dict(model_weight, strict=False)
    print(keys)
    vq_1d = True
    vq_mean = None
    vq_std = None
    dit_input_size = config['num_latent_tokens']
    # vq_1d = False
    # if config_name == 'exp003-aejepa-16':
    #     vq_mean, vq_std = 0.0, 1.0
    #     dit_input_size = 16
    # else:
    #     vq_mean, vq_std = 0.0, 1.0
    #     dit_input_size = 16
    
    
    return vae, config['vq_model'], config['codebook_embed_dim'], dit_input_size, vq_mean, vq_std, vq_1d