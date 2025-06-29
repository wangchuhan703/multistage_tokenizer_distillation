import numpy as np
import torch
import math
import cv2
from multiprocessing import Value
from logging import getLogger
_GLOBAL_SEED = 0
logger = getLogger()

class MaskCollator(object):

    def __init__(
        self,
        input_size=(64, 64),
        patch_size=4,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        shadow_lower=np.array([0, 0, 50]),
        shadow_upper=np.array([180, 50, 150]),
        nenc=1,
        npred=1,
        min_keep=2,
        allow_overlap=False
    ):
        self.patch_size = patch_size
        self.height, self.width = input_size[0]//patch_size, input_size[1]//patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self.shadow_lower = shadow_lower
        self.shadow_upper = shadow_upper
        self._itr_counter = Value('i', -1)

    def _find_shadow_region(self, img_tensor):
        transform_mean = [0.5, 0.5, 0.5]
        transform_std = [0.5, 0.5, 0.5]
        img = (img_tensor * torch.tensor(transform_std).view(3,1,1) 
              + torch.tensor(transform_mean).view(3,1,1)).clamp(0,1)
        img_np = img.permute(1,2,0).numpy() * 255
        img_np = img_np.astype(np.uint8)
        
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        shadow_mask = cv2.inRange(hsv, self.shadow_lower, self.shadow_upper)
        
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        shadow_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(shadow_contour)
        return (x, y, w, h)


    def _gen_shadow_mask(self, img_tensor):
        """生成影子相关预测掩码"""
        shadow_region = self._find_shadow_region(img_tensor)
        if shadow_region is None:
            return None
            
        x, y, w, h = shadow_region
        x_min = x // self.patch_size
        x_max = (x + w) // self.patch_size
        y_min = y // self.patch_size
        y_max = (y + h) // self.patch_size
        


        new_h = self.height - y_min #min(h_patch * 2, self.height - y_min)
        new_w = self.width - x_min #min(w_patch * 2, self.width - x_min)
        
        return (y_min, x_min, new_h, new_w)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height: h -= 1
        while w >= self.width: w -= 1
        return (h, w)

    def _sample_block_mask(self, b_size, shadow_param= None, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            if shadow_param is not None:
                top,left,h,w = shadow_param
            # -- Sample block top-left corner
            else:
                top = torch.randint(0, self.height - h, (1,))
                left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            a = len(torch.nonzero(mask.flatten()))
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]. Top-Left-h-w-len(mask)-a-height-min_keep:{top}-{left}-{h}-{w}-{len(mask)}-{a}-{self.height}-{self.min_keep}"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        
        for img_idx in range(B):
            img_tensor = collated_batch[0][img_idx]
            shadow_mask = self._gen_shadow_mask(img_tensor)
            # print("shadow_mask:",shadow_mask)
            # shadow_mask = None
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size, shadow_param=shadow_mask, acceptable_regions=None)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')
            # print('Encoder')
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
    
