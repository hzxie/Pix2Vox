#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch

class Generator(torch.nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg
        
        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(cfg.CONST.Z_SIZE, cfg.CONST.N_VOX * 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX * 8),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(cfg.CONST.N_VOX * 8, cfg.CONST.N_VOX * 4, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX * 4),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(cfg.CONST.N_VOX * 4, cfg.CONST.N_VOX * 2, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX * 2),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(cfg.CONST.N_VOX * 2, cfg.CONST.N_VOX, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(cfg.CONST.N_VOX, 1, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x, y):
        out = x.view(-1, self.cfg.CONST.Z_SIZE, 1, 1, 1)
        #print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([100, 512, 4, 4, 4])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([100, 128, 16, 16, 16])
        out = self.layer4(out)
        #print(out.size())  # torch.Size([100, 64, 32, 32, 32])
        out = self.layer5(out)
        #print(out.size())  # torch.Size([100, 1, 64, 64, 64])

        return out
