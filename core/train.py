#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import os
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from models.discriminator import Discriminator
from models.generator import Generator

DATASET_LOADER_MAPPING = {
    'ShapeNet': utils.data_loaders.ShapeNetDataLoader,
}

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CropCenter(cfg.CONST.IMG_H, cfg.CONST.IMG_W, cfg.CONST.IMG_C),
        utils.data_transforms.AddRandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
    ])
    val_transforms   = utils.data_transforms.Compose([
        utils.data_transforms.CropCenter(cfg.CONST.IMG_H, cfg.CONST.IMG_W, cfg.CONST.IMG_C),
        utils.data_transforms.AddRandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
    ])
    
    # Set up data loader
    dataset_loader    = DATASET_LOADER_MAPPING[cfg.DIR.DATASET](cfg)
    n_views           = np.random.randint(cfg.CONST.N_VIEWS) + 1 if cfg.TRAIN.RANDOM_NUM_VIEWS else cfg.CONST.N_VIEWS
    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(cfg.TRAIN.DATASET_PORTION, n_views, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, pin_memory=True, shuffle=True)
    val_data_loader   = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(cfg.TEST.DATASET_PORTION, n_views, val_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, pin_memory=True, shuffle=True)

    # Summary writer for TensorBoard
    log_dir      = os.path.join(cfg.DIR.OUT_PATH, 'logs', dt.now().isoformat())
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer   = SummaryWriter(os.path.join(log_dir, 'test'))

    # Set up networks
    generator            = Generator(cfg)
    discriminator        = Discriminator(cfg)
    # Set up solver
    generator_solver     = None
    discriminator_solver = None
    if cfg.TRAIN.POLICY == 'adam':
        generator_solver     = torch.optim.Adam(generator.parameters(), lr=cfg.TRAIN.GENERATOR_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        discriminator_solver = torch.optim.Adam(discriminator.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        generator_solver     = torch.optim.SGD(generator.parameters(), lr=cfg.TRAIN.GENERATOR_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        discriminator_solver = torch.optim.SGD(discriminator.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    generator_lr_scheduler     = torch.optim.lr_scheduler.MultiStepLR(generator_solver, milestones=cfg.TRAIN.GENERATOR_LR_MILESTONES, gamma=0.1)
    discriminator_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_solver, milestones=cfg.TRAIN.DISCRIMINATOR_LR_MILESTONES, gamma=0.1)

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Use CUDA if it is available
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        bce_loss.cuda()

    # Load pretrained model if exists
    network_params = None
    if 'WEIGHTS' in cfg.CONST:
        # TODO
        network_params = torch.load(cfg.CONST.WEIGHTS)

    # Training loop
    for epoch_idx in range(cfg.TRAIN.INITIAL_EPOCH, cfg.TRAIN.NUM_EPOCHES):
        n_batches = len(train_data_loader)
        # Average meterics
        batch_generator_loss        = []
        batch_discriminator_loss    = []
        batch_discriminator_acuracy = []
        
        # Tick / tock
        epoch_start_time = time()

        for batch_idx, (rendering_images, voxels) in enumerate(train_data_loader):
            # Tick / tock
            batch_start_time = time()

            # Generate Gaussian noise
            z = torch.Tensor(cfg.CONST.BATCH_SIZE, cfg.CONST.Z_SIZE).normal_(0, .33)

            # Use soft labels
            labels_real = torch.Tensor(cfg.CONST.BATCH_SIZE).uniform_(.7, 1.2)
            labels_fake = torch.Tensor(cfg.CONST.BATCH_SIZE).uniform_(0, .3)

            # Use CUDA if it is available
            if torch.cuda.is_available():
                rendering_images = rendering_images.cuda()
                voxels           = voxels.cuda()
                z                = z.cuda()
                labels_real      = labels_real.cuda()
                labels_fake      = labels_fake.cuda()

            # Train the discriminator
            generated_voxels            = generator(z, None)
            pred_labels_fake            = discriminator(generated_voxels, None)
            pred_labels_real            = discriminator(voxels, None)

            discriminator_loss_fake     = bce_loss(pred_labels_fake, labels_fake)
            discriminator_loss_real     = bce_loss(pred_labels_real, labels_real)
            discriminator_loss          = discriminator_loss_fake + discriminator_loss_real

            discriminator_acuracy_fake  = torch.le(pred_labels_fake.squeeze(), .5).float()
            discriminator_acuracy_real  = torch.ge(pred_labels_real.squeeze(), .5).float()
            discriminator_acuracy       = torch.mean(torch.cat((discriminator_acuracy_fake, discriminator_acuracy_real)), 0)

            # Balance the learning speed of discriminator and generator
            discriminator.zero_grad()
            discriminator_loss.backward()
            if discriminator_acuracy <= cfg.TRAIN.DISCRIMINATOR_ACC_THRESHOLD:
                discriminator_solver.step()

            # Train the generator
            z = torch.Tensor(cfg.CONST.BATCH_SIZE, cfg.CONST.Z_SIZE).normal_(0, .33)
            if torch.cuda.is_available():
                z = z.cuda()
            
            generated_voxels    = generator(z, None)
            pred_labels_fake    = discriminator(generated_voxels, None)
            generator_loss      = bce_loss(pred_labels_fake, labels_fake)

            discriminator.zero_grad()
            generator.zero_grad()
            generator_loss.backward()
            generator_solver.step()

            # Tick / tock
            batch_end_time = time()
            # Append loss and accuracy to average metrics
            batch_generator_loss.append(generator_loss)
            batch_discriminator_loss.append(discriminator_loss)
            batch_discriminator_acuracy.append(discriminator_acuracy)
            # Append loss and accuracy to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('%s/GLoss' % cfg.DIR.DATASET, generator_loss, n_itr)
            train_writer.add_scalar('%s/DLoss' % cfg.DIR.DATASET, discriminator_loss, n_itr)
            train_writer.add_scalar('%s/DAccuracy' % cfg.DIR.DATASET, discriminator_acuracy, n_itr)

            print('[INFO] %s [Epoch %d/%d][Batch %d/%d] Total Time = %.3f (s) DLoss = %.4f DAccuracy = %.4f GLoss = %.4f' % \
                (dt.now(), epoch_idx, cfg.TRAIN.NUM_EPOCHES, batch_idx, n_batches, batch_end_time - batch_start_time, \
                    discriminator_loss, discriminator_acuracy, generator_loss))

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] Total Time = %.3f (s) DLoss = %.4f DAccuracy = %.4f GLoss = %.4f' % 
            (dt.now(), epoch_idx, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, np.mean(batch_discriminator_loss), \
                np.mean(batch_discriminator_acuracy), np.mean(batch_generator_loss)))

        # Validate the training models
        # TODO

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()