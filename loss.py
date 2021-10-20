import torch
import torch.nn as nn
import lpips

def TV_loss(x):
    ell = torch.pow(torch.abs(x[:, :, 1:, :] - x[:, :, 0:-1, :]), 2).mean()
    ell += torch.pow(torch.abs(x[:, :, :, 1:] - x[:, :, :, 0:-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:, :, 1:, 1:] - x[:, :, :-1, :-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:, :, 1:, :-1] - x[:, :, :-1, 1:]), 2).mean()
    ell /= 4.
    return ell

def BCE_loss(device):
    return nn.BCELoss().to(device)

def L1_loss(device):
    return nn.L1Loss().to(device)

def LPIPS_loss(device):
    return lpips.LPIPS(net='alex').to(device)