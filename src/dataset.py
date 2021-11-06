import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import numpy as np
from torchvision.transforms import transforms


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

    def __getitem__(self, idx):
        pil_image = np.array(self.images[idx])
        image = self.transform(self.images[idx])    # From PIL to Tensor (0-1 scale)
        label = self.labels[idx]
        name = self.names[idx]
        return image, label, pil_image, name

    def __len__(self):
        return len(self.images)
