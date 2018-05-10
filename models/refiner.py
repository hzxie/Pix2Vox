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
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=cfg.NETWORK.DROPOUT_RATE)
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=11),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=cfg.NETWORK.DROPOUT_RATE)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, kernel_size=3, dilation=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(3),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, kernel_size=3, dilation=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=2),
            torch.nn.BatchNorm3d(3),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, kernel_size=3, dilation=4, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=4),
            torch.nn.BatchNorm3d(3),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, kernel_size=3, dilation=8, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=8),
            torch.nn.BatchNorm3d(3),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, kernel_size=3, dilation=16, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=16),
            torch.nn.BatchNorm3d(3),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 3, kernel_size=3, dilation=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(3),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 1, kernel_size=3, dilation=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, gen_voxels, raw_features):
        raw_features    = self.layer0(raw_features)
        # print(raw_features.size())    # torch.Size([batch_size, 256, 26, 26])
        raw_features    = self.layer1(raw_features)
        # print(raw_features.size())    # torch.Size([batch_size, 256, 16, 16])
        raw_features    = raw_features.view((-1, 2, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX))
        # print(raw_features.size())    # torch.Size([batch_size, 2, 32, 32, 32])

        gen_voxels      = gen_voxels.view((-1, 1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX))
        voxel_features  = torch.cat((gen_voxels, raw_features), 1)
        # print(voxel_features.size())  # torch.Size([batch_size, 3, 32, 32, 32])

        voxel_features  = self.layer2(voxel_features)
        # print(voxel_features.size())  # torch.Size([batch_size, 3, 32, 32, 32])
        voxel_features  = self.layer3(voxel_features)
        # print(voxel_features.size())  # torch.Size([batch_size, 3, 32, 32, 32])
        voxel_features  = self.layer4(voxel_features)
        # print(voxel_features.size())  # torch.Size([batch_size, 3, 32, 32, 32])
        voxel_features  = self.layer5(voxel_features)
        # print(voxel_features.size())  # torch.Size([batch_size, 3, 32, 32, 32])
        voxel_features  = self.layer6(voxel_features)
        # print(voxel_features.size())  # torch.Size([batch_size, 3, 32, 32, 32])
        voxel_features  = self.layer7(voxel_features)
        # print(voxel_features.size())  # torch.Size([batch_size, 3, 32, 32, 32])
        voxel_features  = self.layer8(voxel_features)

        return voxel_features.view((-1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX))
