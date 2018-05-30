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
__C.CONST.N_VIEWS                       = 20
__C.CONST.N_VIEWS_RENDERING             = 1
__C.CONST.BATCH_SIZE                    = 24
__C.CONST.CROP_IMG_W                    = 200       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                    = 200       # Dummy property for Pascal 3D
# For ShapeNet
__C.CONST.CROP_IMG_C                    = 4
# For Pascal3D
# __C.CONST.CROP_IMG_C                  = 3

#
# Directories
#
__C.DIR                                 = edict()
__C.DIR.OUT_PATH                        = './output'
# For ShapeNet
__C.DIR.DATASET_TAXONOMY_FILE_PATH      = './datasets/ShapeNet.json'
__C.DIR.VOXEL_PATH                      = '/home/hzxie/Datasets/ShapeNet/ShapeNetVox32/%s/%s.mat'
__C.DIR.RENDERING_PATH                  = '/home/hzxie/Datasets/ShapeNet/ShapeNetRendering/%s/%s/render_%s.png'
# For Pascal 3D
# __C.DIR.DATASET_TAXONOMY_FILE_PATH    = './datasets/Pascal3D.json'
# __C.DIR.VOXEL_PATH                    = '/home/hzxie/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
# __C.DIR.ANNOTATION_PATH               = '/home/hzxie/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
# __C.DIR.RENDERING_PATH                = '/home/hzxie/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'

#
# Dataset
#
__C.DATASET                             = edict()
### For ShapeNet
__C.DATASET.DATASET_NAME                = 'ShapeNet'
__C.DATASET.MEAN                        = [26.2284, 22.7098, 20.8072, 0]
__C.DATASET.STD                         = [0.0676, 0.0618, 0.0598, 1]
### For Pascal 3D
# __C.DATASET.DATASET_NAME              = 'Pascal3D'
# __C.DATASET.MEAN                      = [121.7832, 118.1967, 113.1437]
# __C.DATASET.STD                       = [0.4232, 0.4206, 0.4345]

#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.LEAKY_VALUE                 = .2
__C.NETWORK.DROPOUT_RATE                = .2
__C.NETWORK.TCONV_USE_BIAS              = False
__C.NETWORK.USE_REFINER                 = True

#
# Training
#
__C.TRAIN                               = edict()
__C.TRAIN.RESUME_TRAIN                  = False
__C.TRAIN.NUM_WORKER                    = 4             # number of data workers
### For ShapeNet
__C.TRAIN.NUM_EPOCHES                   = 300           # maximum number of epoches
### For Pascal 3D
# __C.TRAIN.NUM_EPOCHES                 = 100           # maximum number of epoches
__C.TRAIN.RANDOM_NUM_VIEWS              = False         # feed in random #views if n_views > 1
__C.TRAIN.ROTATE_DEGREE_RANGE           = (-15, 15)     # range of degrees to select from
__C.TRAIN.TRANSLATE_RANGE               = (.1, .1)      # tuple of maximum absolute fraction for horizontal and vertical translations
__C.TRAIN.SCALE_RANGE                   = (.75, 1.5)    # tuple of scaling factor interval
__C.TRAIN.BRIGHTNESS                    = .25
__C.TRAIN.CONTRAST                      = .25
__C.TRAIN.SATURATION                    = .25
__C.TRAIN.HUE                           = .25
__C.TRAIN.RANDOM_BG_COLOR_RANGE         = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                        = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER       = 0
__C.TRAIN.ENCODER_LEARNING_RATE         = .001
__C.TRAIN.DECODER_LEARNING_RATE         = .001
__C.TRAIN.REFINER_LEARNING_RATE         = .005
### For ShapeNet
__C.TRAIN.ENCODER_LR_MILESTONES         = [150]
__C.TRAIN.DECODER_LR_MILESTONES         = [150]
__C.TRAIN.REFINER_LR_MILESTONES         = [150]
### For Pascal 3D
# __C.TRAIN.ENCODER_LR_MILESTONES       = [50]
# __C.TRAIN.DECODER_LR_MILESTONES       = [50]
# __C.TRAIN.REFINER_LR_MILESTONES       = [50]
__C.TRAIN.BETAS                         = (.5, .5)
__C.TRAIN.MOMENTUM                      = .9
__C.TRAIN.VISUALIZATION_FREQ            = 10000         # visualization reconstruction voxels every visualization_freq batch
__C.TRAIN.SAVE_FREQ                     = 10            # weights will be overwritten every save_freq epoch

#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE          = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                   = [.2, .3, .4, .5]
