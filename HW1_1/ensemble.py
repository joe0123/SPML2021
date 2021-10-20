import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model

class Ensemble(nn.Module):
    def __init__(self, args):
        super().__init__()
        for model_name in args.model_names:
            model = get_model(model_name, pretrained=True)
            self.add_module(model_name, model)

    def forward(self, x, reduction="mean"):   # return mean probability
        probs = []
        for model in self.children():
            probs.append(model(x).softmax(-1))
        
        probs = torch.stack(probs, dim=0)
        if reduction == "mean":
            probs = probs.mean(0)
        elif reduction == "sum":
            probs = probs.sum(0)
        elif reduction != "none":
            raise NotImplementedError
        
        return probs

