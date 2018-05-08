#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

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

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner

def test_net(cfg, epoch_idx=-1, output_dir=None, test_data_loader=None, test_writer=None, encoder=None, decoder=None, refiner=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE  = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.TRAIN.CROP_IMG_H, cfg.TRAIN.CROP_IMG_W
        test_transforms  = utils.data_transforms.Compose([
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader    = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME](cfg)
        n_rendering_views = np.random.randint(cfg.TRAIN.NUM_RENDERING) + 1 if cfg.TRAIN.RANDOM_NUM_VIEWS else cfg.TRAIN.NUM_RENDERING
        test_data_loader  = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(cfg.TEST.DATASET_PORTION, cfg.CONST.N_VIEWS, n_rendering_views, test_transforms),
            batch_size=1,
            num_workers=1, pin_memory=True, shuffle=False)

    # Summary writer for TensorBoard
    need_to_close_writer = False
    if output_dir is None:
        need_to_close_writer = True
        output_dir  = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
        log_dir     = output_dir % 'logs'
        test_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    
    # Set up networks
    if decoder is None or encoder is None:
        encoder     = Encoder(cfg)
        decoder     = Decoder(cfg)
        refiner     = Refiner(cfg)

        if torch.cuda.is_available():
            encoder.cuda()
            decoder.cuda()
            refiner.cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        refiner.load_state_dict(checkpoint['refiner_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou  = dict()
    test_encoder_loss = []
    test_refiner_loss = []
    for sample_idx, (taxonomy_name, sample_name, rendering_images, ground_truth_voxel) in enumerate(test_data_loader):
        taxonomy_name = taxonomy_name[0]
        sample_name   = sample_name[0]

        # Switch models to training mode
        encoder.eval();
        decoder.eval();
        refiner.eval();

        with torch.no_grad():
            # Get data from data loader
            rendering_images    = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_voxel  = utils.network_utils.var_or_cuda(ground_truth_voxel)

            # Test the decoder
            image_features, raw_features    = encoder(rendering_images)
            generated_voxel                 = decoder(image_features)
            encoder_loss                    = bce_loss(generated_voxel, ground_truth_voxel) * 10

            generated_voxel                 = refiner(generated_voxel, raw_features)
            refiner_loss                    = bce_loss(generated_voxel, ground_truth_voxel) * 10

            # Append loss and accuracy to average metrics
            test_encoder_loss.append(encoder_loss.item())
            test_refiner_loss.append(refiner_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _voxel       = torch.ge(generated_voxel, th).float()
                intersection = torch.sum(_voxel.mul(ground_truth_voxel)).float()
                union        = torch.sum(torch.ge(_voxel.add(ground_truth_voxel), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if not taxonomy_name in test_iou:
                test_iou[taxonomy_name] = {
                    'n_samples': 0,
                    'iou': []
                }
            test_iou[taxonomy_name]['n_samples'] += 1
            test_iou[taxonomy_name]['iou'].append(sample_iou)

            # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' % \
                (dt.now(), sample_idx + 1, n_samples, taxonomy_name, sample_name, encoder_loss.item(), \
                    refiner_loss.item(), ['%.4f' % si for si in sample_iou]))

    # Output testing results
    mean_iou = []
    for taxonomy_name in test_iou:
        test_iou[taxonomy_name]['iou'] = np.mean(test_iou[taxonomy_name]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_name]['iou'] * test_iou[taxonomy_name]['n_samples'])
    mean_iou = np.mean(mean_iou, axis=0) / n_samples

    # Print header
    print('====== TEST RESULTS ======')
    print('Taxonomy', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_name in test_iou:
        print(taxonomy_name, end='\t')
        for ti in test_iou[taxonomy_name]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if not epoch_idx == -1:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', np.mean(test_encoder_loss), epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', np.mean(test_refiner_loss), epoch_idx)
        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    # Close SummaryWriter for TensorBoard
    if need_to_close_writer:
        test_writer.close()

    return max_iou
