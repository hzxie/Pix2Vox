#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# References:
# - https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
# - https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
# - https://github.com/andreaazzini/retinanet.pytorch/blob/master/loss.py
# - https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, x, y):
        pt     = x * y + (1 - x) * (1 - y)
        pt     = pt.clamp(1e-7, 1. - 1e-7)
        log_pt = -torch.log(pt)
        loss   = (1 - pt).pow(self.gamma) * log_pt
        return loss.mean()
