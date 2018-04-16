#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

import utils.data_loaders
import utils.data_transforms

from datetime import datetime as dt
from tensorboardX import SummaryWriter

DATASET_LOADER_MAPPING = {
    'ShapeNet': utils.data_loaders.ShapeNetDataLoader
}

def train_net(cfg):
    dataset_loader   = DATASET_LOADER_MAPPING[cfg.DIR.DATASET](cfg)
    n_views          = np.random.randint(cfg.CONST.N_VIEWS) + 1 if cfg.TRAIN.RANDOM_NUM_VIEWS else cfg.CONST.N_VIEWS
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CropCenter(cfg.CONST.IMG_H, cfg.CONST.IMG_W, cfg.CONST.IMG_C),
        utils.data_transforms.AddRandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
    ])
    val_transforms   = utils.data_transforms.Compose([
        utils.data_transforms.CropCenter(cfg.CONST.IMG_H, cfg.CONST.IMG_W, cfg.CONST.IMG_C),
        utils.data_transforms.AddRandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
    ])

    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(cfg.TRAIN.DATASET_PORTION, n_views, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, pin_memory=True, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(cfg.TEST.DATASET_PORTION, n_views, val_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, pin_memory=True, shuffle=True)

    for idx, (rendering_images, voxel) in enumerate(train_data_loader):
        print(rendering_images.shape)
        print(voxel.shape)
