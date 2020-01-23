# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

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

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

def inference_net(cfg, epoch_idx=-1, output_dir=None, test_data_loader=None, \
        test_writer=None, encoder=None, decoder=None, refiner=None, merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    # open the data sets with the given configurations

    # Here 'cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH' is just the filepath to the json file

    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

        """
        There is a file in utils folder called data_transforms.py
        within that there is a class called compose which is used to create a data transformer class
        some of the transformations done here are:
        CenterCrop, RandomBackground,Normalize,and
        ToTensor that transforms images to tensor
        """
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        """
        In the utils folder there is a file called data_loaders
        within that there is this dictionary 
        DATASET_LOADER_MAPPING = {
        'ShapeNet': ShapeNetDataLoader,
        'Pascal3D': Pascal3dDataLoader,
        'Pix3D': Pix3dDataLoader
        } # yapf: disable
        So basically you are ultimately creating the instance of the class ShapeNetDataLoader
        by passing cfg
        Now, the cfg has the following things being used by the ShapeNetDataLoader class
        1. Image Path: '/home/sidroy/software/Pix2Vox/datasets/ShapeNetRendering/%s/%s/rendering/%02d.png'
        2. Voxel Path: '/home/sidroy/software/Pix2Vox/datasets/ShapeNetVox32/%s/%s/model.binvox'
        3. Taxonomy File Path: i.e. the json file -> './datasets/ShapeNet.json'
        """
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        # Test Data Loader uses the get_dataset method to obtain the data
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST,
                                               cfg.CONST.N_VIEWS_RENDERING, test_transforms),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False)
        print("test_data_loader type {}".format(type(test_data_loader)))
        """
        Instead of the dataset loader and the test data loader where you need all the data use something else
        like simple load images where you can load multiple view images
        and based on the number of images in the folder the number of views will be updated
        """
    # Set up networks
    if decoder is None or encoder is None:
        """
        These are simply creating the encoder, decoder, refiner, and merger class
        """
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger  = Merger(cfg)

        # This has something to do with using cuda for GPU systems
        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        # This is just saying that the weights are being loaded
        # from the path 'cfg.CONST.WEIGHTS' at current time
        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    print("n_samples = {}".format(n_samples))
    test_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    # evaluation mode is a pytorch thing that will notify layers that you are in eval mode,
    # that way, batchnorm or dropout layers will work in eval mode instead of training mode.
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        print("rendering image type {}".format(rendering_images.type))
        # the dimensions of the rendering_images tensor are:
        # (batch size,views?,rgb channels,img dim x, img dim y)
        print("rendering image shape {}".format(rendering_images.shape))
        # the dimensions of the ground_truth_volume tensor are:
        # (batch size,vox dim x, vox dim y, vox dim z)
        print("ground truth volume shape {}".format(ground_truth_volume.shape))

        # this is used becasue we do not want pytorch to follow gradients
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            print("shape of rendering images after cuda {}".format(rendering_images.shape))
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            print("shape of image features {}".format(image_features.shape))
            raw_features, generated_volume = decoder(image_features)
            print("shape of decoded generated volume {}".format(generated_volume.shape))
            print("shape of raw features {}".format(raw_features.shape))

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)
            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)

                refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            else:
                refiner_loss = encoder_loss

            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if not taxonomy_id in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Append generated volumes to TensorBoard
            if output_dir and sample_idx > 4:
                img_dir = output_dir + '/images'
                # Volume Visualization
                gv = generated_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'test'),
                                                                              epoch_idx+sample_idx)

                #test_writer.add_image('TestSample#%02d/Volume Reconstructed' % sample_idx, rendering_views, epoch_idx)
                gtv = ground_truth_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'test'),
                                                                              epoch_idx+sample_idx)
                #test_writer.add_image('TestSample#%02d/Volume GroundTruth' % sample_idx, rendering_views, epoch_idx)

            # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' % \
                (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(), \
                    refiner_loss.item(), ['%.4f' % si for si in sample_iou]))

    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if not test_writer is None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    return max_iou
