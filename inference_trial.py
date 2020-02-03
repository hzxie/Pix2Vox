#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from datetime import datetime as dt
from pprint import pprint

from config import cfg
from core.train import train_net
#from core.test import test_net
#from core.inference import inference_net
from core.demo import test_net
from torch.utils.tensorboard import SummaryWriter

PATH = 'pretrained_models/Pix2Vox-A-ShapeNet.pth'
torch.backends.cudnn.benchmark = True

checkpoint = (torch.load(PATH))

print('Use config:')
pprint(cfg)

cfg.CONST.WEIGHTS = './pretrained_models/Pix2Vox-A-ShapeNet-118epoch.pth'

writer = SummaryWriter('./output/tensorboard')
test_net(cfg,output_dir='./output')
