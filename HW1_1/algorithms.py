import torch
from torch.optim import Adam
import torch.nn.functional as F

def FGSM(model, x, y, loss_fn, epsilons):
    x = x.detach()
    x.requires_grad = True
    
    loss = loss_fn(model(x), y)
    loss.backward()
    
    modifiers = epsilons * x.grad.detach().sign()
    
    return modifiers.clamp(-epsilons, epsilons)


def PGD(model, x, y, loss_fn, lr, max_iter, epsilons):
    modifiers = torch.zeros_like(x, device=x.device)
    
    max_iter = 100
    lr = 1
    for it in range(max_iter):
        adv_x = x + modifiers
        adv_x.requires_grad = True
        loss = loss_fn(model(adv_x), y)
        loss.backward()
    
        modifiers += lr * adv_x.grad.detach()
        modifiers = modifiers.clamp(-epsilons, epsilons)
    
    return modifiers.clamp(-epsilons, epsilons)


def Optimization(model, x, y, lr, max_iter, epsilons, num_classes=100):
    modifiers = torch.zeros_like(x, requires_grad=True, device=x.device)
    optimizer = Adam([modifiers], lr=lr)
    y_oh = F.one_hot(y, num_classes=num_classes).to(y.device)

    for it in range(max_iter):
        adv_x = x + modifiers.clamp(-epsilons, epsilons)
        pred = model(adv_x)
        losses = -torch.log(1 - y_oh * pred).sum(-1)

        optimizer.zero_grad()
        losses.backward(torch.ones_like(losses), retain_graph=True)
        optimizer.step()

    return modifiers.clamp(-epsilons, epsilons)
