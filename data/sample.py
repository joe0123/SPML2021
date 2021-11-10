import os
from collections import defaultdict
from torchvision.datasets import CIFAR100
import numpy as np
np.random.seed(14)

train_dataset = CIFAR100(root="./cifar-100", train=True, download=True)
train_dict = defaultdict(list)
for img, label in train_dataset:
    train_dict[label].append(img)
os.makedirs("cifar-100_train")
for label, imgs in train_dict.items():
    imgs = np.random.choice(imgs, size=5, replace=False)
    for i, img in enumerate(imgs):
        img.save(os.path.join("cifar-100_train", "{}_{}.png".format(label, i)))


test_dataset = CIFAR100(root="./cifar-100", train=False, download=True)
test_dict = defaultdict(list)
for img, label in test_dataset:
    test_dict[label].append(img)
os.makedirs("cifar-100_test")
for label, imgs in test_dict.items():
    imgs = np.random.choice(imgs, size=5, replace=False)
    for i, img in enumerate(imgs):
        img.save(os.path.join("cifar-100_test", "{}_{}.png".format(label, i)))

