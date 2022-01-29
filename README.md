# Security and Privacy of Machine Learning 2021
Homework of [Security and Privacy of Machine Learning 2021 at NTU](https://www.csie.ntu.edu.tw/~stchen/teaching/spml21fall/) (Lectured by Shang-Tse, Chen)

> Black-box and Grey-box attacks on CIFAR-100 datasets

## Preparations

### Environment

Please create the environment by the following commands, where conda is needed.

```
cd envs/
bash create.sh
conda activate spml
```

### Proxy Models

There are three sources of pretrained proxy models:

1. [pytorchcv](https://github.com/osmr/imgclsmob/tree/master/pytorch): Installation of Python package is needed. Nothing else has to do if the aforementioned conda environment is created sucessfully.
2. [Pytorch-CIFAR-Models](https://github.com/chenyaofo/pytorch-cifar-models): Nothing else has to do if pytorch is installed sucessfully.
3. [Pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100): Codes are already forked in the directory *pretrain_cifar/*. You can train and test models for CIFAR100 according to the instructions inside. For generating and evaluating my adversarial images, please download pretrained weights directly from [google drive](https://drive.google.com/file/d/1MOJce8uvf-eTTzVzEW49tmsM_u4ODxJb/view?usp=sharing) and put the checkpoints into directory *cifar_ckpts/*. Please refer to the following structure:
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

Note that the position of original CIFAR100 images and the attack algorithm must be assigned.

Also, you can assign proxy models to `--model_names` as you want. For [pytorchcv](https://github.com/osmr/imgclsmob/tree/master/pytorch), please find those with suffix `_cifar100` from [model provider list](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py). For [Pytorch-CIFAR-Models](https://github.com/chenyaofo/pytorch-cifar-models), please add prefix `cifar100_` to models trained with CIFAR-100. For [Pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100), please train any models you want from *pretrain_cifar/*, and rename the checkpoints to *cifar_ckpts/[model_type].pth*, such as *cifar_ckpts/mobilenetv2.pth* for mobilenetv2. Please refer to the structure mentioned in Preparation.

Lastly, you can attack models against some popular defenses, such as JPEG Compression, Spatial Smoothing and Gaussian Blur. Please assign one of them to `--defense` as you want.

```
python generate.py --data_dir [original CIFAR100 root] --algor [fgsm/ifgsm/opt/pgd]
```

### Evaluate adversarial images

Please run the following command to evaluate adversarial images after generating adversarial images.

Note that the position of original CIFAR100 images must be assigned, and you might need to assign argument `--adv_dir` for the position of adversarial CIFAR100 images if you change `--out_dir` when generating images.

Further, you can evaluate images with some popular defenses, such as JPEG Compression, Spatial Smoothing and Gaussian Blur. Similar to generation, please assign one of them to `--defense` as you want.

```
python evaluate.py --ori_dir [original CIFAR100 root]
```

## Reproduction of submitted results

Please run the following commands to reproduce my submitted results for Black-box attack (Phase 1).

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

Please run the following commands to reproduce my submitted results for Grey-box attack (Phase 2).

```
proxy_models=(
    "vgg16"
    "googlenet"
    "seresnet50"
    "xception"
    "inceptionv4"
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
    "cifar100_resnet32"
    "cifar100_vgg16_bn"
    "cifar100_mobilenetv2_x1_0"
    "cifar100_shufflenetv2_x1_5"
    "cifar100_repvgg_a1"
)
python generate.py --data_dir [original CIFAR100 root] --algor ifgsm --model_names ${proxy_models[@]} --defense blur
```

## Final Evaluation

### Black-box Attack

| Models     | wrn16 | preresnet20 | rir (PGD) | densenet (FGSM) | ror3 (PGD) |
| :--------: | :------: | :------: | :------: | :------: | :------: |
| Accuracy  | 0.016 | 0.016 | 0.396 | 0.162 | 0.29 |

### Grey-box Attack

> One query system is available before deadline.

Target model is Ensemble of resnet56 (FGSM), nin (FGSM), resnet110 (PGD)

| Data     | Public | Private |
| :--------: | :------: | :------: |
| Accuracy  | 0.140 | <= 0.200 |

