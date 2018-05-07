#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch

class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, cfg.CONST.N_VOX, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(cfg.CONST.N_VOX, cfg.CONST.N_VOX * 2, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX * 2),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(cfg.CONST.N_VOX * 2, cfg.CONST.N_VOX * 4, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX * 4),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(cfg.CONST.N_VOX * 4, cfg.CONST.N_VOX * 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX * 8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

    def forward(self, gen_voxels, raw_features):
        return gen_voxels
