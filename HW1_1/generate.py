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

from dataset import CIFAR100

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
    parser.add_argument("--max_iter", type=int, default=100)
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

    lrs = args.epsilon / (255 * torch.Tensor(dataloader.dataset.std)) / args.max_iter
    all_modifiers = []
    correct = 0
    for images, labels, names in dataloader:
        modifiers = [torch.zeros_like(images[:, ci, :, :], requires_grad=True, device=args.device) 
                    for ci in range(3)]
        optimizer = Adam([{"params": m, "lr": lr} for m, lr in zip(modifiers, lrs)])
        images = images.to(args.device)
        labels = labels.to(args.device)
        labels_oh = F.one_hot(labels, num_classes=100).to(args.device)
        
        for it in range(args.max_iter):
            modified_images = images.clone()
            for ci in range(3):
                modified_images[:, ci, :, :] += modifiers[ci]
            preds = model(modified_images)
            losses = -torch.log(1 - labels_oh * preds.softmax(-1)).sum(-1)

            optimizer.zero_grad()
            losses.backward(torch.ones_like(losses), retain_graph=True)
            optimizer.step()
        #print(modifiers[0].max())
        #print(modifiers[1].max())
        #print(modifiers[2].max())
        
        modified_images = images.clone()
        for ci in range(3):
            modified_images[:, ci, :, :] += modifiers[ci]
        preds = model(modified_images)
        correct += (preds.argmax(-1) == labels).sum().item()
    print(correct / len(dataloader.dataset))
