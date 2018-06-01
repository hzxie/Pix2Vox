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
            torch.nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_voxels     = []

        for features in image_features:
            gen_voxel = features.view(-1, 2048, 2, 2, 2)
            # print(gen_voxel.size())   # torch.Size([batch_size, 2048, 2, 2, 2])
            gen_voxel = self.layer1(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 512, 4, 4, 4])
            gen_voxel = self.layer2(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 128, 8, 8, 8])
            gen_voxel = self.layer3(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_voxel = self.layer4(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            gen_voxel = self.layer5(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            gen_voxels.append(torch.squeeze(gen_voxel, 1))

        gen_voxels = torch.stack(gen_voxels).permute(1, 0, 2, 3, 4).contiguous()
        # print(gen_voxels.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        return torch.mean(gen_voxels, dim=1)
