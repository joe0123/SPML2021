import numpy as np
from PIL import Image
from PIL import ImageFilter as IF
import torch
from art.defences.preprocessor import JpegCompression, SpatialSmoothingPyTorch, ThermometerEncoding

class PreDefense():
    def __init__(self, defense_name):
        if defense_name is None:
            self.defense = None
        elif defense_name == "jpeg":
            self.defense = JpegCompression(quality=90, clip_values=(0, 1), channels_first=True)
        elif defense_name == "spatial":
            self.defense = SpatialSmoothingPyTorch(window_size=2, clip_values=(0, 1), channels_first=True)
        elif defense_name == "blur":
            self.defense = GaussianBlur(r=0.6, clip_values=(0, 1), channels_first=True)
        else:
            raise NotImplementedError
    
    def __call__(self, x):
        if self.defense is None:
            return x
        else:
            return torch.from_numpy(self.defense(x.numpy())[0])

class GaussianBlur():
    def __init__(self, r=1, clip_values=(0, 1), channels_first=False):
        self.filter = IF.GaussianBlur(r)
        self.clip_values = clip_values
        self.channels_first = channels_first

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        if self.clip_values[1] == 1:
            imgs_ = (imgs * 255.).astype(np.uint8)
        else:
            imgs_ = imgs.astype(np.uint8)

        if self.channels_first:
            imgs_ = imgs_.transpose((0, 2, 3, 1))
        
        blurred_imgs = []
        for img in imgs_:
            img = Image.fromarray(img)
            blurred_img = img.filter(self.filter)
            blurred_imgs.append(np.array(blurred_img))
        blurred_imgs = np.stack(blurred_imgs, axis=0).astype(imgs.dtype)
        
        if self.clip_values[1] == 1:
            blurred_imgs = blurred_imgs / 255.

        if self.channels_first:
            blurred_imgs = blurred_imgs.transpose((0, 3, 1, 2))
        
        return blurred_imgs, None
