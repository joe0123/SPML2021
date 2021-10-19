import os
import argparse
import logging
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model

from dataset import CIFAR100

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
    parser.add_argument("--data_dir", type=str, default="./out")
    parser.add_argument("--model_names", nargs='+',
                        default=["resnet56_cifar100", "resnet110_cifar100", 
                                "resnet164bn_cifar100", "densenet100_k24_cifar100",
                                "resnet20_cifar100", "seresnet164bn_cifar100",
                                "preresnet272bn_cifar100", "diaresnet56_cifar100"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    logger.info("Loading data...")
    dataloader = DataLoader(dataset=CIFAR100(args), \
                            batch_size=args.batch_size, \
                            shuffle=False, \
                            num_workers=4)

    for model_name in args.model_names:
        logger.info("Loading proxy model {}...".format(model_name))
        model = get_model(model_name, pretrained=True).to(args.device)
        model.eval()
        
        logger.info("Evaluating...")
        correct_count = 0
        for images, labels, _, _ in dataloader:
            images = images.to(args.device)
            preds = model(images)
            correct_count += (preds.argmax(-1).cpu() == labels).sum().item()
        acc = np.round(correct_count / len(dataloader.dataset), 5)
        print("Accuracy for {}: {}".format(model_name, acc))
