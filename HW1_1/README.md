# SPML2021 HW1-1 Black Box Attack

## Preparations

### Environment

Please create the environment by the following commands, where conda is needed.

```
cd envs/
bash create.sh
```

### Proxy Models

There are two sources of pretrained proxy models:

1. [pytorchcv](https://github.com/osmr/imgclsmob/tree/master/pytorch): Installation of Python package is needed. Nothing else has to do if the aforementioned conda environment is created sucessfully.
2. [Pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100): Codes are already forked in the directory `pretrain_cifar/`. You can train and test models for CIFAR100 according to the instruction inside. For generating and evaluating my adversarial images, please download pretrained weights directly from []() and put the checkpoints into directory `cifar_ckpts/`. Like this: 
```
src/
│   README.md
└───pretrain_cifar/
    └─── ...
└───cifar_ckpts/
│   │   vgg16.pth
│   │   googlenet.pth
│   │   mobilenetv2.pth
│   |   seresnet50.pth
|   generate.py
|   evaluate.py
|   ...
```
