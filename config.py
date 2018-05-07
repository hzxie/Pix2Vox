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
__C.CONST.IMG_W                         = 224       # Image width for input
__C.CONST.IMG_H                         = 224       # Image height for input
__C.CONST.IMG_C                         = 3         # Image channels for input
__C.CONST.N_VOX                         = 32
__C.CONST.N_VIEWS                       = 12
__C.CONST.BATCH_SIZE                    = 8
__C.CONST.Z_SIZE                        = 128

#
# Directories
#
__C.DIR                                 = edict()
__C.DIR.DATASET_TAXONOMY_FILE_PATH      = './datasets/ShapeNet.json'
__C.DIR.DATASET_QUERY_PATH              = '/run/media/Data/Temporary/Datasets/ShapeNet/ShapeNetRendering'
__C.DIR.VOXEL_PATH                      = '/run/media/Data/Temporary/Datasets/ShapeNet/ShapeNetVox32/%s/%s.mat'
__C.DIR.RENDERING_PATH                  = '/run/media/Data/Temporary/Datasets/ShapeNet/ShapeNetRendering/%s/%s/render_%s.png'
__C.DIR.OUT_PATH                        = './output'

#
# Dataset
#
__C.DATASET                             = edict()
__C.DATASET.DATASET_NAME                = 'ShapeNet'
__C.DATASET.MEAN                        = [32.2859, 27.1176, 24.4343, 50.5743]
__C.DATASET.STD                         = [0.1903, 0.1708, 0.1634, 0.2678]

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
__C.TRAIN.DATASET_PORTION               = [0, .8]
## Data worker
__C.TRAIN.NUM_WORKER                    = 1         # number of data workers
__C.TRAIN.NUM_EPOCHES                   = 2000      # maximum number of epoches
__C.TRAIN.NUM_RENDERING                 = 20
__C.TRAIN.RANDOM_NUM_VIEWS              = False     # feed in random # views if n_views > 1
## Data augmentation
__C.TRAIN.CROP_IMG_W                    = 200
__C.TRAIN.CROP_IMG_H                    = 200
__C.TRAIN.CROP_IMG_C                    = 4
__C.TRAIN.ROTATE_DEGREE_RANGE           = (-15, 15) # range of degrees to select from
__C.TRAIN.TRANSLATE_RANGE               = None      # tuple of maximum absolute fraction for horizontal and vertical translations
__C.TRAIN.SCALE_RANGE                   = None      # tuple of scaling factor interval
__C.TRAIN.BRIGHTNESS                    = .25
__C.TRAIN.CONTRAST                      = .25
__C.TRAIN.SATURATION                    = .25
__C.TRAIN.HUE                           = .25
__C.TRAIN.RANDOM_BG_COLOR_RANGE         = [[225, 255], [225, 255], [225, 255]]
## Learning
__C.TRAIN.POLICY                        = 'adam'    # available options: sgd, adam
__C.TRAIN.ENCODER_LEARNING_RATE         = .001
__C.TRAIN.DECODER_LEARNING_RATE         = .0025
__C.TRAIN.REFINER_LEARNING_RATE         = .0025
__C.TRAIN.ENCODER_LR_MILESTONES         = []
__C.TRAIN.DECODER_LR_MILESTONES         = []
__C.TRAIN.REFINER_LR_MILESTONES         = []
__C.TRAIN.BETAS                         = (.5, .5)
__C.TRAIN.MOMENTUM                      = .9
__C.TRAIN.VISUALIZATION_FREQ            = 100       # visualization reconstruction voxels every visualization_freq batch
__C.TRAIN.SAVE_FREQ                     = 10        # weights will be overwritten every save_freq epoch

#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.DATASET_PORTION                = [.8, 1]
__C.TEST.RANDOM_BG_COLOR_RANGE          = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                   = [.2, .25, .3, .4, .5]