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

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    IMG_SIZE  = cfg.CONST.IMG_H, cfg.CONST.IMG_W, cfg.CONST.IMG_C
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W, cfg.CONST.CROP_IMG_C
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION, cfg.TRAIN.HUE),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms  = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ToTensor(),
    ])
    
    # Set up data loader
    dataset_loader    = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME](cfg)
    n_rendering_views = np.random.randint(cfg.CONST.N_VIEWS_RENDERING) + 1 if cfg.TRAIN.RANDOM_NUM_VIEWS else cfg.CONST.N_VIEWS_RENDERING
    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS, n_rendering_views, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, pin_memory=True, shuffle=True)
    val_data_loader   = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS, n_rendering_views, val_transforms),
        batch_size=1,
        num_workers=1, pin_memory=True, shuffle=False)

    # Set up networks
    encoder      = Encoder(cfg)
    decoder      = Decoder(cfg)
    refiner      = Refiner(cfg)
    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    refiner.apply(utils.network_utils.init_weights)

    # Set up solver
    decoder_solver = None
    encoder_solver = None
    refiner_solver = None
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        refiner_solver = torch.optim.SGD(refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver, milestones=cfg.TRAIN.ENCODER_LR_MILESTONES, gamma=0.1)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver, milestones=cfg.TRAIN.DECODER_LR_MILESTONES, gamma=0.1)
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver, milestones=cfg.TRAIN.REFINER_LR_MILESTONES, gamma=0.1)

    if torch.cuda.is_available():
        torch.nn.DataParallel(encoder).cuda()
        torch.nn.DataParallel(decoder).cuda()
        torch.nn.DataParallel(refiner).cuda()

    # Set up loss functions
    bce_loss   = torch.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch     = 0
    best_iou       = -1
    best_epoch     = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou   = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder_solver.load_state_dict(checkpoint['encoder_solver_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder_solver.load_state_dict(checkpoint['decoder_solver_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
            refiner_solver.load_state_dict(checkpoint['refiner_solver_state_dict'])
        
        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' \
                 % (dt.now(), init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir   = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    log_dir      = output_dir % 'logs'
    img_dir      = output_dir % 'images'
    ckpt_dir     = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer   = SummaryWriter(os.path.join(log_dir, 'test'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()
        
        # Batch average meterics
        batch_time        = utils.network_utils.AverageMeter()
        data_time         = utils.network_utils.AverageMeter()
        encoder_losses    = utils.network_utils.AverageMeter()
        refiner_losses    = utils.network_utils.AverageMeter()

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()

        batch_end_time = time()
        n_batches      = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_voxels) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            n_samples = len(ground_truth_voxels)
            # Ignore imcomplete batches at the end of each epoch
            if not n_samples == cfg.CONST.BATCH_SIZE:
                continue

            # switch models to training mode
            encoder.train();
            decoder.train();
            # refiner.train();

            # Get data from data loader
            rendering_images     = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_voxels  = utils.network_utils.var_or_cuda(ground_truth_voxels)

            # Train the encoder, decoder and refiner
            image_features       = encoder(rendering_images)
            generated_voxels     = decoder(image_features)
            encoder_loss         = bce_loss(generated_voxels, ground_truth_voxels) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_voxels = refiner(generated_voxels)
                refiner_loss     = bce_loss(generated_voxels, ground_truth_voxels) * 10
            else:
                refiner_loss     = encoder_loss
            
            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            refiner.zero_grad()
            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()
            else:
                encoder_loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            
            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)
            # Append rendering images of voxels to TensorBoard
            if n_itr > 0 and n_itr % cfg.TRAIN.VISUALIZATION_FREQ == 0:
                gtv          = ground_truth_voxels.cpu().data[:8].numpy()
                voxel_views  = utils.binvox_visualization.get_voxel_views(gtv, os.path.join(img_dir, 'train'), n_itr)
                train_writer.add_image('Ground Truth Voxels', voxel_views, n_itr)
                gv           = generated_voxels.cpu().data[:8].numpy()
                voxel_views  = utils.binvox_visualization.get_voxel_views(gv, os.path.join(img_dir, 'train'), n_itr)
                train_writer.add_image('Reconstructed Voxels', voxel_views, n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            print('[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' % \
                (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, \
                    batch_time.val, data_time.val, encoder_loss.item(), refiner_loss.item()))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' % 
            (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, \
                encoder_losses.avg, refiner_losses.avg))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, refiner)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            
            utils.network_utils.save_checkpoints(cfg, \
                    os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                    epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver, \
                    refiner, refiner_solver, best_iou, best_epoch)
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            
            best_iou   = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, \
                    os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                    epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver, \
                    refiner, refiner_solver, best_iou, best_epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()

