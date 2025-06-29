# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

def apply_masks_with_mask_token(x, masks, mask_token):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    B, N, D = x.shape
    all_x = []
    for m in masks:
        # create a binary mask of shape [B, N] initialized with False
        binary_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        
        # set True at the indices where patches are kept
        binary_mask.scatter_(1, m, True)
        
        # expand the binary mask to match feature dimension
        binary_mask = binary_mask.unsqueeze(-1).expand(-1, -1, D)
        
        # get masked x
        masked_x = torch.where(binary_mask, x, mask_token)
        
        all_x += [masked_x]
    return torch.cat(all_x, dim=0)


def reverse_masks(masked_tokens, masks, original_shape, mask_token):
    """
    Reconstruct the original tensor with masked positions filled by mask_token.
    
    :param masked_tokens: Tensor of shape [B * len(masks), num_keep, D] 
                          containing tokens selected by the masks.
    :param masks: List of tensors, each of shape [B, num_keep], containing the indices of tokens kept.
    :param original_shape: Tuple (B, N, D) representing the shape of the original tensor.
    :param mask_token: Tensor of shape [D] to fill positions not selected by the mask.
    :return: Tensor of shape [B * len(masks), N, D] where positions not in the mask are filled with mask_token.
    """
    B, N, D = original_shape
    outputs = []
    # Split masked_tokens into a list where each element corresponds to one mask.
    tokens_split = torch.chunk(masked_tokens, len(masks), dim=0)
    
    for tokens, m in zip(tokens_split, masks):
        # Create an output tensor filled with mask_token.
        # mask_token is of shape [D], so we expand it to [B, N, D].
        out = mask_token.to(tokens.dtype).expand(B, N, D).clone()
        
        # The mask `m` is of shape [B, num_keep]. We need to expand it to [B, num_keep, D]
        # so that we can scatter the corresponding tokens.
        index = m.unsqueeze(-1).expand_as(tokens)
        
        # Scatter the tokens from the masked_tokens into their original positions.
        out.scatter_(dim=1, index=index, src=tokens)
        outputs.append(out)
        
    # Concatenate the outputs for each mask along the batch dimension.
    return torch.cat(outputs, dim=0)

def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x