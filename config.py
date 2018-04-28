#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C     = edict()
cfg     = __C

#
# Common
#
__C.CONST                               = edict()
__C.CONST.DEVICE                        = '0'
__C.CONST.RNG_SEED                      = 0
__C.CONST.IMG_W                         = 200       # Image width after cropping
__C.CONST.IMG_H                         = 200       # Image height after cropping
__C.CONST.IMG_C                         = 4         # Image channels before (/after) cropping
__C.CONST.N_VOX                         = 32
__C.CONST.N_VIEWS                       = 1
__C.CONST.BATCH_SIZE                    = 100
__C.CONST.Z_SIZE                        = 200

#
# Directories
#
__C.DIR     = edict()
__C.DIR.DATASET                         = 'ShapeNet'
__C.DIR.DATASET_TAXONOMY_FILE_PATH      = './datasets/ShapeNet.json'
__C.DIR.DATASET_QUERY_PATH              = '/run/media/Data/Temporary/Datasets/ShapeNet/ShapeNetRendering'
__C.DIR.VOXEL_PATH                      = '/run/media/Data/Temporary/Datasets/ShapeNet/ShapeNetVox32/%s/%s.mat'
__C.DIR.RENDERING_PATH                  = '/run/media/Data/Temporary/Datasets/ShapeNet/ShapeNetRendering/%s/%s/render_%s.png'
__C.DIR.OUT_PATH                        = './output'

#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.LEAKY_VALUE                 = .2
__C.NETWORK.TCONV_USE_BIAS              = False

#
# Training
#
__C.TRAIN                               = edict()
__C.TRAIN.RESUME_TRAIN                  = False
__C.TRAIN.INITIAL_EPOCH                 = 0         # when the training resumes, set the epoch number
__C.TRAIN.DATASET_PORTION               = [0, .8]
## Data worker
__C.TRAIN.NUM_WORKER                    = 1         # number of data workers
__C.TRAIN.NUM_EPOCHES                   = 1000      # maximum number of epoches
__C.TRAIN.NUM_RENDERING                 = 20
__C.TRAIN.RANDOM_NUM_VIEWS              = False     # feed in random # views if n_views > 1
## Data augmentation
__C.TRAIN.RANDOM_CROP                   = True
__C.TRAIN.PAD_X                         = 10
__C.TRAIN.PAD_Y                         = 10
__C.TRAIN.FLIP                          = True
__C.TRAIN.RANDOM_BG_COLOR_RANGE         = [[225, 255], [225, 255], [225, 255]]
## Learning
__C.TRAIN.POLICY                        = 'adam'    # available options: sgd, adam
__C.TRAIN.GENERATOR_LEARNING_RATE       = .0025     # for SGD use 0.1, for ADAM, use 1e-5
__C.TRAIN.DISCRIMINATOR_LEARNING_RATE   = .001      # for SGD use 0.1, for ADAM, use 1e-5
__C.TRAIN.GENERATOR_LR_MILESTONES       = []
__C.TRAIN.DISCRIMINATOR_LR_MILESTONES   = []
__C.TRAIN.DISCRIMINATOR_ACC_THRESHOLD   = .8
__C.TRAIN.BETAS                         = (.5, .5)
__C.TRAIN.MOMENTUM                      = .9
__C.TRAIN.VISUALIZATION_FREQ            = 50        # visualization reconstruction voxels every visualization_freq batch
__C.TRAIN.SAVE_FREQ                     = 10        # weights will be overwritten every save_freq epoch

#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.DATASET_PORTION                = [.8, 1]
__C.TEST.RANDOM_BG_COLOR_RANGE          = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                   = [.4]