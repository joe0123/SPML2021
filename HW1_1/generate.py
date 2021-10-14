import os
import argparse
import logging
import numpy as np
import random
import torch
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
    parser.add_argument("--batch_size", type=int, default=64)
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
        models.append(model)

