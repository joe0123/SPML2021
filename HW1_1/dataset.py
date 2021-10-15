import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms


class CIFAR100(Dataset):
    def __init__(self, args):
        self.images = []
        self.labels = []
        self.names = []
        for f_name in sorted(glob.glob("{}/*".format(args.data_dir))):
            self.images.append(Image.open(f_name))
            self.labels.append(int(os.path.basename(f_name).split('_')[0]))
            self.names.append(f_name)
        
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
    
    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = self.labels[idx]
        name = self.names[idx]
        return image, label, name

    def __len__(self):
        return len(self.images)
