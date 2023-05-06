import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def create_optimizers(model, config):
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(params=[
        {'params': model.parameters(), 'lr': 0.1*config.lr},
        ], lr=config.lr, betas=(0.9, 0.999), eps=1e-08)
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(params=[
        {'params': model.parameters(), 'lr': 0.1*config.lr},
        ], lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)
    return optimizer, scheduler
