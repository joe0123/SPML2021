import os
import argparse
import logging
import warnings
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model

from dataset import CIFAR100
from model import get_cifar_model, CIFAR100Model, pre_defense


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

warnings.filterwarnings("ignore")

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
        ori_image = np.array(ori_map[adv_name]).astype(np.int8)
        adv_image = np.array(adv_image).astype(np.int8)
        if np.abs(ori_image - adv_image).max() > epsilon:
            print("Result: FAIL! Please generate images with proper epsilon.")
            exit()
    
    print("Result: PASS!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_dir", type=str, default="../data/cifar-100_eval/")
    parser.add_argument("--adv_dir", type=str, default="./adv_images")
    parser.add_argument("--model_names", nargs='+',
                        default=["vgg16", "googlenet", "mobilenetv2", "seresnet50",
                                "resnet20_cifar100", "resnet56_cifar100", 
                                "resnet110_cifar100", "resnet164bn_cifar100", 
                                "resnet272bn_cifar100", "resnet1001_cifar100",
                                "preresnet20_cifar100", "preresnet56_cifar100", 
                                "preresnet110_cifar100", "preresnet164bn_cifar100", 
                                "preresnet272bn_cifar100", "preresnet1001_cifar100",
                                "seresnet20_cifar100", "seresnet56_cifar100",
                                "seresnet110_cifar100", "seresnet164bn_cifar100",
                                "seresnet272bn_cifar100",
                                "densenet40_k12_cifar100", "densenet40_k12_bc_cifar100",
                                "densenet100_k12_cifar100", "densenet100_k24_cifar100",
                                "densenet250_k24_bc_cifar100",
                                "pyramidnet110_a48_cifar100", "pyramidnet110_a84_cifar100",
                                "pyramidnet236_a220_bn_cifar100", "pyramidnet272_a200_bn_cifar100",
                                "wrn28_10_cifar100", "wrn40_8_cifar100",
                                "nin_cifar100", "ror3_164_cifar100",
                                "sepreresnet20_cifar100", "sepreresnet56_cifar100", 
                                "sepreresnet110_cifar100", "sepreresnet164bn_cifar100", 
                                "sepreresnet272bn_cifar100", "sepreresnet542bn_cifar100", 
                                "resnext29_32x4d_cifar100", "resnext272_2x32d_cifar100",
                                "rir_cifar100", "xdensenet40_2_k36_bc_cifar100",
                                "shakeshakeresnet26_2x32d_cifar100", "diaresnet110_cifar100",
                                ])
    parser.add_argument("--defense", type=str, choices=["jpeg", "spatial"])
    parser.add_argument("--epsilon", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
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
        if model_name.endswith("cifar100"):
            model = get_model(model_name, pretrained=True)
        elif model_name.startswith("cifar100"):
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
        else:
            model = get_cifar_model(model_name)
        model = CIFAR100Model(model).to(args.device)
        model.eval()

        logger.info("Evaluating...")
        correct_count = 0
        for images, labels, _, _ in adv_dataloader:
            images = pre_defense(images, args.defense)
            images = images.to(args.device)
            preds = model(images)
            correct_count += (preds.argmax(-1).cpu() == labels).sum().item()
        acc = np.round(correct_count / len(adv_dataloader.dataset), 5)
        print("Accuracy for {}: {}".format(model_name, acc))
