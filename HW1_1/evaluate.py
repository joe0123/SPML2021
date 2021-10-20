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

def validate(ori_dataset, adv_dataset, epsilon):
    if len(ori_dataset) != len(adv_dataset):
        print("Result: FAIL! Please generate images with proper counts.")
        exit()
    
    ori_map = dict()
    for i in range(len(ori_dataset)):
        _, _, ori_image, ori_name = ori_dataset[i]
        ori_map[ori_name] = ori_image
    
    for i in range(len(adv_dataset)):
        _, _, adv_image, adv_name = adv_dataset[i]
        ori_image = np.array(ori_map[adv_name])
        adv_image = np.array(adv_image)
        if np.abs(ori_image - adv_image).max() > epsilon:
            print("Result: FAIL! Please generate images with proper epsilon.")
            exit()
    
    print("Result: PASS!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_dir", type=str, default="./adv_images")
    parser.add_argument("--adv_dir", type=str, default="./adv_images")
    parser.add_argument("--model_names", nargs='+',
                        default=["resnet20_cifar100", "resnet56_cifar100", 
                                "resnet110_cifar100", "resnet164bn_cifar100"])
    parser.add_argument("--epsilon", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    logger.info("Loading data...")
    ori_dataset = CIFAR100(args.ori_dir)
    adv_dataset = CIFAR100(args.adv_dir)
    adv_dataloader = DataLoader(dataset=adv_dataset, \
                                batch_size=args.batch_size, \
                                shuffle=False, \
                                num_workers=4)

    logger.info("Validating data with epsilon {}...".format(args.epsilon))
    validate(ori_dataset, adv_dataset, args.epsilon)
    
    for model_name in args.model_names:
        logger.info("Loading proxy model {}...".format(model_name))
        model = get_model(model_name, pretrained=True).to(args.device)
        model.eval()
        
        logger.info("Evaluating...")
        correct_count = 0
        for images, labels, _, _ in adv_dataloader:
            images = images.to(args.device)
            preds = model(images)
            correct_count += (preds.argmax(-1).cpu() == labels).sum().item()
        acc = np.round(correct_count / len(adv_dataloader.dataset), 5)
        print("Accuracy for {}: {}".format(model_name, acc))
