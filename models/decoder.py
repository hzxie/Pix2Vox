#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch

class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1024, 256, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 16, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, image_features):
        gen_voxels = image_features.view(-1, 1024, 2, 2, 2)
        # print(gen_voxels.size())  # torch.Size([batch_size, 1024, 2, 2, 2])
        gen_voxels = self.layer1(gen_voxels)
        # print(gen_voxels.size())  # torch.Size([batch_size, 256, 4, 4, 4])
        gen_voxels = self.layer2(gen_voxels)
        # print(gen_voxels.size())  # torch.Size([batch_size, 64, 8, 8, 8])
        gen_voxels = self.layer3(gen_voxels)
        # print(gen_voxels.size())  # torch.Size([batch_size, 16, 16, 16, 16])
        gen_voxels = self.layer4(gen_voxels)
        # print(gen_voxels.size())  # torch.Size([batch_size, 1, 32, 32, 32])

        return torch.squeeze(gen_voxels, 1)