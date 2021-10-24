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
2. [Pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100): Codes are already forked in the directory *pretrain_cifar/*. You can train and test models for CIFAR100 according to the instructions inside. For generating and evaluating my adversarial images, please download pretrained weights directly from [google drive](https://drive.google.com/file/d/1MOJce8uvf-eTTzVzEW49tmsM_u4ODxJb/view?usp=sharing) and put the checkpoints into directory *cifar_ckpts/*. Please refer to the following structure:
```
src/
│   README.md
└───pretrain_cifar/
│   └─── ...
└───cifar_ckpts/
│   │   vgg16.pth
│   │   googlenet.pth
│   │   mobilenetv2.pth
│   │   seresnet50.pth
│   generate.py
│   evaluate.py
│   ...
```

## Usages

### Generate adversarial images

Please run the following command to generate adversarial images. 

Note that the position of original CIFAR100 images and attack algorithm must be assigned.

Also, you can assign proxy models to `--model_names` as you want. For [pytorchcv](https://github.com/osmr/imgclsmob/tree/master/pytorch), please find those with suffix `_cifar100` from [model provider list](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py). For [Pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100), please train any models you want from *pretrain_cifar/*, and rename the checkpoints to *cifar_ckpts/[model_type].pth*, such as *cifar_ckpts/mobilenetv2.pth* for mobilenetv2. Please refer to the structure mentioned in Preparation.

```
python generate.py --data_dir [original CIFAR100 root] --algor [fgsm / ifgsm / opt / pgd]
```

### Evaluate adversarial images

Please run the following command to evaluate adversarial images after generating the adversarial images.

Note that the position of original CIFAR100 images must be assigned, and you might need to assign argument `--adv_dir` for the position of adversarial CIFAR100 images if you change `--out_dir` when generating images.

```
python evaluate.py --ori_dir [original CIFAR100 root]
```

## Reproduction of submitted results

Please run the following commands to reproduce my submitted results.

```
proxy_models=(
    "vgg16"
    "googlenet"
    "resnet20_cifar100"
    "resnet1001_cifar100"
    "preresnet20_cifar100"
    "preresnet1001_cifar100"
    "seresnet20_cifar100"
    "seresnet272bn_cifar100"
    "densenet40_k12_cifar100"
    "densenet250_k24_bc_cifar100"
    "pyramidnet110_a84_cifar100"
    "pyramidnet272_a200_bn_cifar100"
    "resnext29_32x4d_cifar100"
    "wrn28_10_cifar100"
    "nin_cifar100"
    "ror3_164_cifar100"
)
python generate.py --data_dir [original CIFAR100 root] --algor ifgsm --model_names ${proxy_models[@]}
```


