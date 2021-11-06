import torch
from torch.optim import Adam
import torch.nn.functional as F


def FGSM(model, x, y, loss_fn, epsilon, lr=1, max_iter=100):
    modifiers = torch.zeros_like(x, device=x.device)
    
    for it in range(max_iter):
        adv_x = x + modifiers
        adv_x.requires_grad = True
        loss = loss_fn(model(adv_x), y)
        loss.backward()
    
        modifiers += lr * epsilon * adv_x.grad.detach().sign()
        modifiers = modifiers.clamp(-epsilon, epsilon)
    
    return modifiers

def PGD(model, x, y, loss_fn, epsilon, lr=1, max_iter=100):
    modifiers = torch.zeros_like(x, device=x.device)
    
    for it in range(max_iter):
        adv_x = x + modifiers
        adv_x.requires_grad = True
        loss = loss_fn(model(adv_x), y)
        loss.backward()
    
        modifiers += lr * adv_x.grad.detach()
        modifiers = modifiers.clamp(-epsilon, epsilon)
    
    return modifiers

def Optimization(model, x, y, epsilon, lr=1e-1, max_iter=100, num_classes=100):
    modifiers = torch.zeros_like(x, requires_grad=True, device=x.device)
    optimizer = Adam([modifiers], lr=lr)
    y_oh = F.one_hot(y, num_classes=num_classes).to(y.device)

    for it in range(max_iter):
        adv_x = x + modifiers.clamp(-epsilon, epsilon)
        pred = model(adv_x).softmax(-1)
        losses = -torch.log(1 - y_oh * pred).sum(-1).mean()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return modifiers.clamp(-epsilon, epsilon)
