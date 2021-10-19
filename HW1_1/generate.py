import os
import argparse
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model
from PIL import Image

from dataset import CIFAR100

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/cifar-100_eval")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--model_names", nargs='+',
                        default=["resnet56_cifar100", "resnet110_cifar100", 
                                "resnet164bn_cifar100", "densenet100_k24_cifar100"])
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    logger.info("Loading data...")
    dataloader = DataLoader(dataset=CIFAR100(args), \
                            batch_size=args.batch_size, \
                            shuffle=False, \
                            num_workers=4)

    logger.info("Loading proxy models...")
    models = []
    for model_name in args.model_names:
        model = get_model(model_name, pretrained=True).to(args.device)
        model.eval()
        models.append(model)
    
    means = torch.Tensor(dataloader.dataset.mean).reshape(1, 3, 1, 1).to(args.device)
    stds = torch.Tensor(dataloader.dataset.std).reshape(1, 3, 1, 1).to(args.device)
    epsilons = args.epsilon / (255 * stds)
    for images, labels, ori_images, names in dataloader:
        modifiers = torch.zeros_like(images, requires_grad=True, device=args.device)
        optimizer = Adam([modifiers], lr=args.lr)
        images = images.to(args.device)
        labels_oh = F.one_hot(labels, num_classes=100).to(args.device)
        
        for it in range(args.max_iter):
            adv_images = images + modifiers.clamp(-epsilons, epsilons)
            preds = torch.zeros_like(labels_oh).float()
            for model in models:
                preds += model(adv_images).softmax(-1)
            preds = preds / len(models)
            #preds = model(adv_images).softmax(-1)
            losses = -torch.log(1 - labels_oh * preds).sum(-1)

            optimizer.zero_grad()
            losses.backward(torch.ones_like(losses), retain_graph=True)
            optimizer.step()
        
        adv_images = images + modifiers.clamp(-epsilons, epsilons)
        for model in models:
            print((model(adv_images).argmax(-1).cpu() == labels).float().mean())
        print('')
        
        ori_images = ori_images.numpy()
        noises = (modifiers.detach() * stds * 255).clamp(-args.epsilon, args.epsilon)
        noises = noises.cpu().numpy().transpose((0, 2, 3, 1)) # (b, C, H, W) -> (b, H, W, C)
        adv_images = np.floor(ori_images + noises).clip(0, 255)     
        for adv_image, name in zip(adv_images, names):
            im = Image.fromarray(adv_image.astype(np.uint8))
            im.save(os.path.join(args.out_dir, name))
