import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from pytorchcv.model_provider import get_model


class DefenseModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cifar100_mean, cifar100_std)])

    def forward(self, x):
        x = torch.stack([self.transform(x_) for x_ in x], dim=0).to(x.device)
        return self.model(x)


class Ensemble(nn.Module):
    def __init__(self, args):
        super().__init__()
        for model_name in args.model_names:
            if model_name.endswith("cifar100"):
                model = get_model(model_name, pretrained=True)
            else:
                model = get_cifar_model(model_name)
            self.add_module(model_name, DefenseModel(model))

    def forward(self, x, reduction="mean"):   # Return logits
        logits = []
        for model in self.children():
            logits.append(model(x))
        
        logits = torch.stack(logits, dim=0)
        if reduction == "mean":
            logits = logits.mean(0)
        elif reduction == "sum":
            logits = logits.sum(0)
        elif reduction != "none":
            raise NotImplementedError
        
        return logits

def get_cifar_model(model_name, ckpt_dir="./cifar_ckpts"):
    model = get_network(model_name)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "{}.pth".format(model_name))))
    
    return model

def get_network(net_name):
    """ Return given network"""

    if net_name == "vgg16":
        from pretrain_cifar.models.vgg import vgg16_bn
        net = vgg16_bn()
    elif net_name == "vgg13":
        from pretrain_cifar.models.vgg import vgg13_bn
        net = vgg13_bn()
    elif net_name == "vgg11":
        from pretrain_cifar.models.vgg import vgg11_bn
        net = vgg11_bn()
    elif net_name == "vgg19":
        from pretrain_cifar.models.vgg import vgg19_bn
        net = vgg19_bn()
    elif net_name == "densenet121":
        from pretrain_cifar.models.densenet import densenet121
        net = densenet121()
    elif net_name == "densenet161":
        from pretrain_cifar.models.densenet import densenet161
        net = densenet161()
    elif net_name == "densenet169":
        from pretrain_cifar.models.densenet import densenet169
        net = densenet169()
    elif net_name == "densenet201":
        from pretrain_cifar.models.densenet import densenet201
        net = densenet201()
    elif net_name == "googlenet":
        from pretrain_cifar.models.googlenet import googlenet
        net = googlenet()
    elif net_name == "inceptionv3":
        from pretrain_cifar.models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif net_name == "inceptionv4":
        from pretrain_cifar.models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif net_name == "inceptionresnetv2":
        from pretrain_cifar.models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif net_name == "xception":
        from pretrain_cifar.models.xception import xception
        net = xception()
    elif net_name == "resnet18":
        from pretrain_cifar.models.resnet import resnet18
        net = resnet18()
    elif net_name == "resnet34":
        from pretrain_cifar.models.resnet import resnet34
        net = resnet34()
    elif net_name == "resnet50":
        from pretrain_cifar.models.resnet import resnet50
        net = resnet50()
    elif net_name == "resnet101":
        from pretrain_cifar.models.resnet import resnet101
        net = resnet101()
    elif net_name == "resnet152":
        from pretrain_cifar.models.resnet import resnet152
        net = resnet152()
    elif net_name == "preactresnet18":
        from pretrain_cifar.models.preactresnet import preactresnet18
        net = preactresnet18()
    elif net_name == "preactresnet34":
        from pretrain_cifar.models.preactresnet import preactresnet34
        net = preactresnet34()
    elif net_name == "preactresnet50":
        from pretrain_cifar.models.preactresnet import preactresnet50
        net = preactresnet50()
    elif net_name == "preactresnet101":
        from pretrain_cifar.models.preactresnet import preactresnet101
        net = preactresnet101()
    elif net_name == "preactresnet152":
        from pretrain_cifar.models.preactresnet import preactresnet152
        net = preactresnet152()
    elif net_name == "resnext50":
        from pretrain_cifar.models.resnext import resnext50
        net = resnext50()
    elif net_name == "resnext101":
        from pretrain_cifar.models.resnext import resnext101
        net = resnext101()
    elif net_name == "resnext152":
        from pretrain_cifar.models.resnext import resnext152
        net = resnext152()
    elif net_name == "shufflenet":
        from pretrain_cifar.models.shufflenet import shufflenet
        net = shufflenet()
    elif net_name == "shufflenetv2":
        from pretrain_cifar.models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif net_name == "squeezenet":
        from pretrain_cifar.models.squeezenet import squeezenet
        net = squeezenet()
    elif net_name == "mobilenet":
        from pretrain_cifar.models.mobilenet import mobilenet
        net = mobilenet()
    elif net_name == "mobilenetv2":
        from pretrain_cifar.models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif net_name == "nasnet":
        from pretrain_cifar.models.nasnet import nasnet
        net = nasnet()
    elif net_name == "attention56":
        from pretrain_cifar.models.attention import attention56
        net = attention56()
    elif net_name == "attention92":
        from pretrain_cifar.models.attention import attention92
        net = attention92()
    elif net_name == "seresnet18":
        from pretrain_cifar.models.senet import seresnet18
        net = seresnet18()
    elif net_name == "seresnet34":
        from pretrain_cifar.models.senet import seresnet34
        net = seresnet34()
    elif net_name == "seresnet50":
        from pretrain_cifar.models.senet import seresnet50
        net = seresnet50()
    elif net_name == "seresnet101":
        from pretrain_cifar.models.senet import seresnet101
        net = seresnet101()
    elif net_name == "seresnet152":
        from pretrain_cifar.models.senet import seresnet152
        net = seresnet152()
    elif net_name == "wideresnet":
        from pretrain_cifar.models.wideresidual import wideresnet
        net = wideresnet()
    elif net_name == "stochasticdepth18":
        from pretrain_cifar.models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif net_name == "stochasticdepth34":
        from pretrain_cifar.models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif net_name == "stochasticdepth50":
        from pretrain_cifar.models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif net_name == "stochasticdepth101":
        from pretrain_cifar.models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()
    else:
        raise NotImplementedError

    return net


