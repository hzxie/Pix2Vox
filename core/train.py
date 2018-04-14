#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import numpy as np
import torch.utils.data

import utils.data_loaders

from datetime import datetime as dt
from tensorboardX import SummaryWriter

DATASET_LOADER_MAPPING = {
    'ShapeNet': utils.data_loaders.ShapeNetDataLoader
}

def train_net(cfg):
    dataset_getter = DATASET_LOADER_MAPPING[cfg.DIR.DATASET](cfg)
    n_views        = np.random.randint(cfg.CONST.N_VIEWS) + 1 if cfg.TRAIN.RANDOM_NUM_VIEWS else cfg.CONST.N_VIEWS
    transforms     = None   # TODO

    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_getter.get_dataset(cfg.TRAIN.DATASET_PORTION, n_views),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, pin_memory=True, shuffle=False)
    val_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_getter.get_dataset(cfg.TEST.DATASET_PORTION, n_views),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, pin_memory=True, shuffle=False)

