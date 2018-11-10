# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.CROP_IMG_C            = 4
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/home/hzxie/Datasets/ShapeNet/ShapeNetRendering/%s/%s/render_%s.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/home/hzxie/Datasets/ShapeNet/ShapeNetVox32/%s/%s.mat'
__C.DATASETS.PASCAL3D                       = edict()
__C.DATASETS.PASCAL3D.CROP_IMG_C            = 3
__C.DATASETS.PASCAL3D.TAXONOMY_FILE_PATH    = './datasets/Pascal3D.json'
__C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/home/hzxie/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
__C.DATASETS.PASCAL3D.RENDERING_PATH        = '/home/hzxie/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
__C.DATASETS.PASCAL3D.VOXEL_PATH            = '/home/hzxie/Datasets/PASCAL3D/CAD/%s/%02d.binvox'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.DATASET_NAME                    = 'ShapeNet'
# __C.DATASET.DATASET_NAME                  = 'Pascal3D'
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [255, 255, 255]

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.IMG_C                             = 3         # Image channels for input
__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 64
__C.CONST.N_VIEWS                           = 20        # Dummy property for Pascal 3D
__C.CONST.N_VIEWS_RENDERING                 = 1         # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_W                        = 210       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 210       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_C                        = __C.DATASETS[__C.DATASET.DATASET_NAME.upper()].CROP_IMG_C

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.DROPOUT_RATE                    = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = True

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 250
__C.TRAIN.ROTATE_DEGREE_RANGE               = (-15, 15)     # range of degrees to select from
__C.TRAIN.TRANSLATE_RANGE                   = (.1, .1)      # tuple of maximum absolute fraction for horizontal and vertical translations
__C.TRAIN.SCALE_RANGE                       = (.75, 1.5)    # tuple of scaling factor interval
__C.TRAIN.BRIGHTNESS                        = .25
__C.TRAIN.CONTRAST                          = .25
__C.TRAIN.SATURATION                        = .25
__C.TRAIN.HUE                               = .25
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE             = 1e-2
__C.TRAIN.MERGER_LEARNING_RATE              = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.VISUALIZATION_FREQ                = 10000         # visualization reconstruction voxels every visualization_freq batch
__C.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
