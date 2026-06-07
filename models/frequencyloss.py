import math
import torch
import numpy as np
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F


def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


class AmplitudeLoss(nn.Module):
    def __init__(self):
        super(AmplitudeLoss, self).__init__()
    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        amp = torch.abs(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        amp1 = torch.abs(fre1)
        # return l1_loss(amp, amp1, reduction='mean')
        return F.l1_loss(amp, amp1, reduction='mean')



class PhaseLoss(nn.Module):
    def __init__(self):
        super(PhaseLoss, self).__init__()
    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        pha = torch.angle(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        pha1 = torch.angle(fre1)
        # return l1_loss(pha, pha1, reduction='mean')
        return F.l1_loss(pha, pha1, reduction='mean')
        
        
class Frequency_loss(nn.Module):
    def __init__(self, opt):
        super(Frequency_loss, self).__init__()
        self.phase = PhaseLoss()
        self.amplitude = AmplitudeLoss()
    def forward(self, out, gt, lam_p, lam_a):
        loss = lam_p * self.phase(out, gt) + lam_a * self.amplitude(out, gt)
        return loss