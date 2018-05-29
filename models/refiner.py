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
            torch.nn.Conv3d(1, cfg.CONST.N_VOX, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(cfg.CONST.N_VOX, cfg.CONST.N_VOX * 2, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX * 2),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(cfg.CONST.N_VOX * 2, cfg.CONST.N_VOX * 4, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX * 4),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(8192, 2048),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192),
            torch.nn.ReLU()
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(cfg.CONST.N_VOX * 4, cfg.CONST.N_VOX * 2, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX * 2),
            torch.nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(cfg.CONST.N_VOX * 2, cfg.CONST.N_VOX, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(cfg.CONST.N_VOX),
            torch.nn.ReLU()
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(cfg.CONST.N_VOX, 1, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, coarse_voxels):
        voxels_32_l      = coarse_voxels.view((-1, 1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX))
        # print(voxels_32_l.size())       # torch.Size([batch_size, 1, 32, 32, 32])
        voxels_16_l      = self.layer1(voxels_32_l)
        # print(voxels_16_l.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        voxels_8_l       = self.layer2(voxels_16_l)
        # print(voxels_8_l.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        voxels_4_l       = self.layer3(voxels_8_l)
        # print(voxels_4_l.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        flatten_features = self.layer4(voxels_4_l.view(-1, 8192))
        # print(flatten_features.size())  # torch.Size([batch_size, 2048])
        flatten_features = self.layer5(flatten_features)
        # print(flatten_features.size())  # torch.Size([batch_size, 8192])
        voxels_4_r       = voxels_4_l + flatten_features.view(-1, self.cfg.CONST.N_VOX * 4, 4, 4, 4)
        # print(voxels_4_r.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        voxels_8_r       = voxels_8_l + self.layer6(voxels_4_r)
        # print(voxels_8_r.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        voxels_16_r      = voxels_16_l + self.layer7(voxels_8_r)
        # print(voxels_16_r.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        voxels_32_r      = (voxels_32_l + self.layer8(voxels_16_r)) * 0.5
        # print(voxels_32_r.size())       # torch.Size([batch_size, 1, 32, 32, 32])

        return voxels_32_r.view((-1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX))
