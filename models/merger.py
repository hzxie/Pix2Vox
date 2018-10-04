# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Merger(torch.nn.Module):
    def __init__(self, cfg):
        super(Merger, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 4, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(4),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(4, 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(2),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

    def forward(self, raw_features, coarse_voxels):
        raw_features = torch.split(raw_features, 1, dim=1)
        voxel_weights = []

        for i in range(self.cfg.CONST.N_VIEWS_RENDERING):
            raw_feature = torch.squeeze(raw_features[i], dim=1)
            # print(raw_feature.size())       # torch.Size([batch_size, 9, 32, 32, 32])

            voxel_weight = self.layer1(raw_feature)
            # print(voxel_weight.size())      # torch.Size([batch_size, 16, 32, 32, 32])
            voxel_weight = self.layer2(voxel_weight)
            # print(voxel_weight.size())      # torch.Size([batch_size, 8, 32, 32, 32])
            voxel_weight = self.layer3(voxel_weight)
            # print(voxel_weight.size())      # torch.Size([batch_size, 4, 32, 32, 32])
            voxel_weight = self.layer4(voxel_weight)
            # print(voxel_weight.size())      # torch.Size([batch_size, 2, 32, 32, 32])
            voxel_weight = self.layer5(voxel_weight)
            # print(voxel_weight.size())      # torch.Size([batch_size, 1, 32, 32, 32])

            voxel_weight = torch.squeeze(voxel_weight, dim=1)
            # print(voxel_weight.size())      # torch.Size([batch_size, 32, 32, 32])
            voxel_weights.append(voxel_weight)

        voxel_weights = torch.stack(voxel_weights).permute(1, 0, 2, 3, 4).contiguous()
        voxel_weights = torch.softmax(voxel_weights, dim=1)
        # print(voxel_weights.size())         # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(coarse_voxels.size())         # torch.Size([batch_size, n_views, 32, 32, 32])
        coarse_voxels = coarse_voxels * voxel_weights
        coarse_voxels = torch.sum(coarse_voxels, dim=1)

        return torch.clamp(coarse_voxels, min=0, max=1)
