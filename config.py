#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C     = edict()
cfg     = __C

#
# Dataset Config
#
DATASET_CONFIG                          = {
    'ShapeNet': {
        'CONST.N_VIEWS':                20,
        'CONST.BATCH_SIZE':             64,
        'CONST.CROP_IMG_C':             4,
        'DATASET.TAXONOMY_FILE_PATH':   './datasets/ShapeNet.json',
        'DATASET.ANNOTATION_PATH':      'Dummy Property. DO NOT USE IT',
        'DATASET.RENDERING_PATH':       '/home/hzxie/Datasets/ShapeNet/ShapeNetRendering/%s/%s/render_%s.png',
        'DATASET.VOXEL_PATH':           '/home/hzxie/Datasets/ShapeNet/ShapeNetVox32/%s/%s.mat',
        'DATASET.MEAN':                 [26.2284, 22.7098, 20.8072, 0],
        'DATASET.STD':                  [0.0676, 0.0618, 0.0598, 1],
        'TRAIN.NUM_EPOCHES':            250,
        'TRAIN.ENCODER_LR_MILESTONES':  [150],
        'TRAIN.DECODER_LR_MILESTONES':  [150],
        'TRAIN.REFINER_LR_MILESTONES':  [150],
        'TRAIN.MERGER_LR_MILESTONES':   [150]
    },
    'Pascal3D': {
        'CONST.N_VIEWS':                0,
        'CONST.BATCH_SIZE':             24,
        'CONST.CROP_IMG_C':             3,
        'DATASET.TAXONOMY_FILE_PATH':   './datasets/Pascal3D.json',
        'DATASET.ANNOTATION_PATH':      '/home/hzxie/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat',
        'DATASET.RENDERING_PATH':       '/home/hzxie/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG',
        'DATASET.VOXEL_PATH':           '/home/hzxie/Datasets/PASCAL3D/CAD/%s/%02d.binvox',
        'DATASET.MEAN':                 [121.7832, 118.1967, 113.1437],
        'DATASET.STD':                  [0.4232, 0.4206, 0.4345],
        'TRAIN.NUM_EPOCHES':            100,
        'TRAIN.ENCODER_LR_MILESTONES':  [50],
        'TRAIN.DECODER_LR_MILESTONES':  [50],
        'TRAIN.REFINER_LR_MILESTONES':  [50],
        'TRAIN.MERGER_LR_MILESTONES':   [50]
    },
    'ModelNet40': {
        'CONST.N_VIEWS':                12,
        'CONST.BATCH_SIZE':             64,
        'CONST.CROP_IMG_C':             4,
        'DATASET.TAXONOMY_FILE_PATH':   './datasets/ModelNet40.json',
        'DATASET.ANNOTATION_PATH':      'Dummy Property. DO NOT USE IT',
        'DATASET.RENDERING_PATH':       '/home/hzxie/Datasets/ModelNet40/%s/%s/%s_%03d.png',
        'DATASET.VOXEL_PATH':           '/home/hzxie/Datasets/ModelNet40/%s/%s/%s.binvox',
        'DATASET.MEAN':                 [223.1698, 223.1698, 223.1698],
        'DATASET.STD':                  [0.2642, 0.2642, 0.2642],
        'TRAIN.NUM_EPOCHES':            250,
        'TRAIN.ENCODER_LR_MILESTONES':  [150],
        'TRAIN.DECODER_LR_MILESTONES':  [150],
        'TRAIN.REFINER_LR_MILESTONES':  [150],
        'TRAIN.MERGER_LR_MILESTONES':   [150],
    }
}

#
# Dataset
#
__C.DATASET                             = edict()
__C.DATASET.DATASET_NAME                = 'ShapeNet'
# __C.DATASET.DATASET_NAME              = 'Pascal3D'
# __C.DATASET.DATASET_NAME              = 'ModelNet40'
__C.DATASET.TAXONOMY_FILE_PATH          = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['DATASET.TAXONOMY_FILE_PATH']
__C.DATASET.RENDERING_PATH              = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['DATASET.RENDERING_PATH']
__C.DATASET.VOXEL_PATH                  = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['DATASET.VOXEL_PATH']
__C.DATASET.ANNOTATION_PATH             = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['DATASET.ANNOTATION_PATH']
__C.DATASET.MEAN                        = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['DATASET.MEAN']
__C.DATASET.STD                         = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['DATASET.STD']

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
__C.CONST.BATCH_SIZE                    = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['CONST.BATCH_SIZE']
__C.CONST.N_VIEWS                       = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['CONST.N_VIEWS']
__C.CONST.N_VIEWS_RENDERING             = 1         # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_W                    = 210       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                    = 210       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_C                    = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['CONST.CROP_IMG_C']

#
# Directories
#
__C.DIR                                 = edict()
__C.DIR.OUT_PATH                        = './output'

#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.LEAKY_VALUE                 = .2
__C.NETWORK.DROPOUT_RATE                = .2
__C.NETWORK.TCONV_USE_BIAS              = False
__C.NETWORK.USE_REFINER                 = True
__C.NETWORK.USE_MERGER                  = True

#
# Training
#
__C.TRAIN                               = edict()
__C.TRAIN.RESUME_TRAIN                  = False
__C.TRAIN.NUM_WORKER                    = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                   = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['TRAIN.NUM_EPOCHES']
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
__C.TRAIN.EPOCH_START_USE_MERGER        = 0
__C.TRAIN.ENCODER_LEARNING_RATE         = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE         = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE         = 1e-2
__C.TRAIN.MERGER_LEARNING_RATE          = 1e-3
__C.TRAIN.ENCODER_LR_MILESTONES         = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['TRAIN.ENCODER_LR_MILESTONES']
__C.TRAIN.DECODER_LR_MILESTONES         = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['TRAIN.DECODER_LR_MILESTONES']
__C.TRAIN.REFINER_LR_MILESTONES         = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['TRAIN.REFINER_LR_MILESTONES']
__C.TRAIN.MERGER_LR_MILESTONES          = DATASET_CONFIG[cfg.DATASET.DATASET_NAME]['TRAIN.MERGER_LR_MILESTONES']
__C.TRAIN.BETAS                         = (.9, .999)
__C.TRAIN.MOMENTUM                      = .9
__C.TRAIN.GAMMA                         = .5
__C.TRAIN.VISUALIZATION_FREQ            = 10000         # visualization reconstruction voxels every visualization_freq batch
__C.TRAIN.SAVE_FREQ                     = 10            # weights will be overwritten every save_freq epoch

#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE          = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                   = [.2, .3, .4, .5]
