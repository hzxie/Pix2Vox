#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models

class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Dropout(p=cfg.NETWORK.DROPOUT_RATE),
            torch.nn.Conv2d(self.cfg.CONST.N_VIEWS_RENDERING * 512, 512, kernel_size=1),
            torch.nn.ELU(inplace=True)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=cfg.NETWORK.DROPOUT_RATE)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=5)
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images   = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images   = torch.split(rendering_images, 1, dim=0)
        image_features     = []

        for view_idx, img in enumerate(rendering_images):
            features = self.vgg(img.view(-1, self.cfg.CONST.IMG_C, self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W))
            image_features.append(features)

        image_features = torch.cat(image_features, 1)
        # print(image_features.size())  # torch.Size([batch_size, n_views * 256, 28, 28])
        image_features = self.layer1(image_features)
        raw_features   = image_features
        # print(image_features.size())  # torch.Size([batch_size, 512, 28, 28])
        image_features = self.layer2(image_features)
        # print(image_features.size())  # torch.Size([batch_size, 256, 26, 26])
        image_features = self.layer3(image_features)
        # print(image_features.size())  # torch.Size([batch_size, 128, 4, 4])

        return image_features.view(-1, 2048), raw_features
