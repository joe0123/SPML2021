import os
import argparse
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from dataset import CIFAR100
from model import Ensemble
from algorithms import FGSM, PGD, Optimization
from defense import PreDefense


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
    parser.add_argument("--out_dir", type=str, default="./adv_images")
    parser.add_argument("--algor", type=str, required=True, choices=["fgsm", "ifgsm", "pgd", "opt"])
    parser.add_argument("--model_names", nargs='+',
                        default=["vgg16", "googlenet", 
                                "resnet20_cifar100", "resnet1001_cifar100",
                                "preresnet20_cifar100", "preresnet1001_cifar100",
                                "seresnet20_cifar100", "seresnet272bn_cifar100",
                                "densenet40_k12_cifar100", "densenet250_k24_bc_cifar100",
                                "pyramidnet110_a84_cifar100", "pyramidnet272_a200_bn_cifar100",
                                "resnext29_32x4d_cifar100", "wrn28_10_cifar100", 
                                "nin_cifar100", "ror3_164_cifar100"])
    parser.add_argument("--defense", type=str, choices=["jpeg", "spatial", "blur"])
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.out_dir)
    set_seed(args.seed)

    logger.info("Loading data...")
    dataloader = DataLoader(dataset=CIFAR100(args.data_dir), \
                            batch_size=args.batch_size, \
                            shuffle=False, \
                            num_workers=4)

    logger.info("Loading proxy models...")
    ensemble_model = Ensemble(args).to(args.device)
    ensemble_model.eval()
    
    logger.info("Generating adversarial images...")
    epsilon = args.epsilon / 255.
    defense = PreDefense(args.defense)
    for i, (images, labels, ori_images, names) in enumerate(dataloader):
        #print(i, flush=True)
        images = images.to(args.device)
        labels = labels.to(args.device)
        if args.algor == "fgsm":
            modifiers = FGSM(ensemble_model, images, labels, nn.CrossEntropyLoss(), \
                            epsilon, lr=1, max_iter=1, defense=defense)
        elif args.algor == "ifgsm":
            modifiers = FGSM(ensemble_model, images, labels, nn.CrossEntropyLoss(), \
                            epsilon, lr=args.lr, max_iter=args.max_iter, defense=defense)
        elif args.algor == "pgd":
            modifiers = PGD(ensemble_model, images, labels, nn.CrossEntropyLoss(), \
                            epsilon, lr=args.lr, max_iter=args.max_iter, defense=defense)
        elif args.algor == "opt":
            modifiers = Optimization(ensemble_model, images, labels, \
                            epsilon, lr=args.lr, max_iter=args.max_iter, defense=defense)
        else:
            raise NotImplementedError
        
                
        ori_images = ori_images.numpy()
        noises = (modifiers.detach() * 255.).clamp(-args.epsilon, args.epsilon)
        noises = noises.cpu().numpy().transpose((0, 2, 3, 1)) # (b, C, H, W) -> (b, H, W, C)
        adv_images = np.floor(ori_images + noises).clip(0, 255)     
        for adv_image, name in zip(adv_images, names):
            im = Image.fromarray(adv_image.astype(np.uint8))
            im.save(os.path.join(args.out_dir, name))
