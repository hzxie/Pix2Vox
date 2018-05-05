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
from models.discriminator import Discriminator
from models.generator import Generator
from models.image_encoder import ImageEncoder

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.CropCenter(cfg.CONST.IMG_H, cfg.CONST.IMG_W, cfg.CONST.IMG_C),
        utils.data_transforms.AddRandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ArrayToTensor3d(),
    ])
    
    # Set up data loader
    dataset_loader    = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME](cfg)
    n_views           = np.random.randint(cfg.CONST.N_VIEWS) + 1 if cfg.TRAIN.RANDOM_NUM_VIEWS else cfg.CONST.N_VIEWS
    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(cfg.TRAIN.DATASET_PORTION, n_views, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, pin_memory=True, shuffle=True)

    # Summary writer for TensorBoard
    log_dir      = os.path.join(cfg.DIR.OUT_PATH, 'logs', dt.now().isoformat())
    img_dir      = os.path.join(cfg.DIR.OUT_PATH, 'images', dt.now().isoformat())
    ckpt_dir     = os.path.join(cfg.DIR.OUT_PATH, 'checkpoints', dt.now().isoformat())
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer   = SummaryWriter(os.path.join(log_dir, 'test'))

    # Set up networks
    generator            = Generator(cfg)
    discriminator        = Discriminator(cfg)
    image_encoder        = ImageEncoder(cfg)

    # Initialize weights of networks
    generator.apply(utils.network_utils.init_weights)
    discriminator.apply(utils.network_utils.init_weights)
    image_encoder.apply(utils.network_utils.init_weights)

    # Set up solver
    generator_solver     = None
    discriminator_solver = None
    image_encoder_solver = None
    if cfg.TRAIN.POLICY == 'adam':
        generator_solver     = torch.optim.Adam(generator.parameters(), lr=cfg.TRAIN.GENERATOR_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        discriminator_solver = torch.optim.Adam(discriminator.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        image_encoder_solver = torch.optim.Adam(image_encoder.parameters(), lr=cfg.TRAIN.IMAGE_ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        generator_solver     = torch.optim.SGD(generator.parameters(), lr=cfg.TRAIN.GENERATOR_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        discriminator_solver = torch.optim.SGD(discriminator.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        image_encoder_solver = torch.optim.SGD(image_encoder.parameters(), lr=cfg.TRAIN.IMAGE_ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    generator_lr_scheduler     = torch.optim.lr_scheduler.MultiStepLR(generator_solver, milestones=cfg.TRAIN.GENERATOR_LR_MILESTONES, gamma=0.1)
    discriminator_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_solver, milestones=cfg.TRAIN.DISCRIMINATOR_LR_MILESTONES, gamma=0.1)
    image_encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(image_encoder_solver, milestones=cfg.TRAIN.IMAGE_ENCODER_LR_MILESTONES, gamma=0.1)

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        image_encoder.cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Load pretrained model if exists
    network_params = None
    if 'WEIGHTS' in cfg.CONST:
        network_params = torch.load(cfg.CONST.WEIGHTS)

    # Training loop
    best_iou = 0
    for epoch_idx in range(cfg.TRAIN.INITIAL_EPOCH, cfg.TRAIN.NUM_EPOCHES):
        n_batches = len(train_data_loader)
        # Average meterics
        epoch_image_encoder_loss    = []
        epoch_generator_loss        = []
        epoch_discriminator_loss    = []
        epoch_discriminator_acuracy = []
        
        # Tick / tock
        epoch_start_time = time()

        for batch_idx, (taxonomy_names, sample_names, rendering_images, voxels) in enumerate(train_data_loader):
            n_samples = len(voxels)
            if not n_samples == cfg.CONST.BATCH_SIZE:
                continue

            # Tick / tock
            batch_start_time = time()

            # switch models to training mode
            generator.train();
            discriminator.train();
            image_encoder.train();

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            voxels           = utils.network_utils.var_or_cuda(voxels)

            # Use soft labels
            labels_real     = utils.network_utils.var_or_cuda(torch.Tensor(n_samples).uniform_(0.7, 1.2))
            labels_fake     = utils.network_utils.var_or_cuda(torch.Tensor(n_samples).uniform_(0, 0.3))

            # Train the discriminator
            rendering_image_features    = image_encoder(rendering_images)
            generated_voxels            = generator(rendering_image_features)
            pred_labels_real            = discriminator(voxels, rendering_image_features)
            pred_labels_fake            = discriminator(generated_voxels, rendering_image_features)

            discriminator_loss_real     = bce_loss(pred_labels_real, labels_real)
            discriminator_loss_fake     = bce_loss(pred_labels_fake, labels_fake)
            discriminator_loss          = (discriminator_loss_real + discriminator_loss_fake) * 0.5

            discriminator_acuracy_real  = torch.ge(pred_labels_real.view(cfg.CONST.BATCH_SIZE), 0.5).float()
            discriminator_acuracy_fake  = torch.le(pred_labels_fake.view(cfg.CONST.BATCH_SIZE), 0.5).float()
            discriminator_acuracy       = torch.mean(torch.cat((discriminator_acuracy_real, discriminator_acuracy_fake), 0))

            # Balance the learning speed of discriminator and generator
            if discriminator_acuracy <= cfg.TRAIN.DISCRIMINATOR_ACC_THRESHOLD:
                discriminator.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                discriminator_solver.step()

            # Train the generator and the image encoder
            image_encoder_loss          = bce_loss(generated_voxels, voxels) * 10
            generator_loss              = bce_loss(pred_labels_fake, labels_real) + image_encoder_loss

            discriminator.zero_grad()
            generator.zero_grad()
            image_encoder.zero_grad()
            
            generator_loss.backward()
            
            generator_solver.step()
            image_encoder_solver.step()

            # Tick / tock
            batch_end_time = time()
            
            # Append loss and accuracy to average metrics
            epoch_image_encoder_loss.append(image_encoder_loss.item())
            epoch_generator_loss.append(generator_loss.item())
            epoch_discriminator_loss.append(discriminator_loss.item())
            epoch_discriminator_acuracy.append(discriminator_acuracy.item())
            # Append loss and accuracy to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('Generator/GeneratorLoss', generator_loss.item(), n_itr)
            train_writer.add_scalar('Generator/ImageEncoderLoss', image_encoder_loss.item(), n_itr)
            train_writer.add_scalar('Discriminator/Loss', discriminator_loss.item(), n_itr)
            train_writer.add_scalar('Discriminator/Accuracy', discriminator_acuracy.item(), n_itr)
            # Append rendering images of voxels to TensorBoard
            if n_itr % cfg.TRAIN.VISUALIZATION_FREQ == 0:
                # TODO: add GT here ...
                gv           = generated_voxels.cpu().data[:8].numpy()
                voxel_views  = utils.binvox_visualization.get_voxel_views(gv, os.path.join(img_dir, 'train'), n_itr)
                train_writer.add_image('Reconstructed Voxels', voxel_views, n_itr)

            print('[INFO] %s [Epoch %d/%d][Batch %d/%d] Total Time = %.3f (s) DLoss = %.4f DAccuracy = %.4f GLoss = %.4f ILoss = %.4f' % \
                (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_end_time - batch_start_time, \
                    discriminator_loss, discriminator_acuracy, generator_loss, image_encoder_loss))

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] Total Time = %.3f (s) DLoss = %.4f DAccuracy = %.4f GLoss = %.4f ILoss = %.4f' % 
            (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, np.mean(epoch_discriminator_loss), \
                np.mean(epoch_discriminator_acuracy), np.mean(epoch_generator_loss), np.mean(epoch_image_encoder_loss)))

        # Validate the training models
        mean_iou = test_net(cfg, val_writer, generator, image_encoder)

        # Save weights to file
        # TODO: Save the best validation model
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 and not epoch_idx == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            torch.save({
                'epoch_idx': epoch_idx,
                'generator_state_dict': generator.state_dict(),
                'generator_solver_state_dict': generator_solver.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'discriminator_solver_state_dict': discriminator_solver.state_dict(),
            }, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % epoch_idx))

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
