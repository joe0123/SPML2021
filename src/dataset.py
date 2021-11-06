import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import numpy as np
from torchvision.transforms import transforms

class JpegCompression():
    def __init__(self, quality=80):
        self._quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        with BytesIO() as f:
            img.save(f, format='JPEG', quality=self._quality)
            img = Image.open(f).convert('RGB')
        return img

class CIFAR100(Dataset):
    def __init__(self, data_dir):
        self.images = []
        self.labels = []
        self.names = []
        for f_name in sorted(glob.glob("{}/*".format(data_dir))):
            self.images.append(Image.open(f_name))
            name = os.path.basename(f_name)
            self.labels.append(int(name.split('_')[0]))
            self.names.append(name)
        
        self.transform = transforms.ToTensor()
        #self.transform = transforms.Compose([JpegCompression(),
        #                                    transforms.ToTensor()])
    
    def __getitem__(self, idx):
        pil_image = np.array(self.images[idx])
        image = self.transform(self.images[idx])
        label = self.labels[idx]
        name = self.names[idx]
        return image, label, pil_image, name

    def __len__(self):
        return len(self.images)
