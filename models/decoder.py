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

    def forward(self, x):
        # print(x.size())    # torch.Size([batch_size, z_size])
        out = x.view(-1, self.cfg.CONST.Z_SIZE, 1, 1, 1)
        # print(out.size())  # torch.Size([batch_size, z_size, 1, 1, 1])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([batch_size, 256, 2, 2, 2])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([batch_size, 128, 4, 4, 4])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([batch_size, 64, 8, 8, 8])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([batch_size, 32, 16, 16, 16])
        out = self.layer5(out)
        # print(out.size())  # torch.Size([batch_size, 1, 32, 32, 32])

        return torch.squeeze(out, 1)
