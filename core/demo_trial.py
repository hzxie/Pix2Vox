# -*- coding: utf-8 -*-
#
# Developed by Sidhartha Roy

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

def test_net(cfg, epoch_idx=-1, output_dir=None, test_data_loader=None, \
        test_writer=None, encoder=None, decoder=None, refiner=None, merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        """
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])
        """
        sample_idx = 1
        rendering_images = load_image()

        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST,
                                               cfg.CONST.N_VIEWS_RENDERING, test_transforms),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        print('Epoch ID of the current model is {}'.format(epoch_idx))
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
    test_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    print("test data loader type is {}".format(type(test_data_loader)))

    with torch.no_grad():
        # Get data from data loader
        rendering_images = utils.network_utils.var_or_cuda(rendering_images)

        print("Shape of the loaded images {}".format(rendering_images.shape))

        # Test the encoder, decoder, refiner and merger
        image_features = encoder(rendering_images)
        raw_features, generated_volume = decoder(image_features)

        if cfg.NETWORK.USE_MERGER:
            generated_volume = merger(raw_features, generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)


        if cfg.NETWORK.USE_REFINER:
            generated_volume = refiner(generated_volume)

        print("vox shape {}".format(generated_volume.shape))

        gv = generated_volume.cpu().numpy()

        rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(
            './LargeDatasets/inference_images/', 'inference'),
                                                                      sample_idx)
        #if sample_idx == 3:
        #    break
    print("gv shape is {}".format(gv.shape))
    return gv, rendering_images

def load_image():
    import cv2
    filename = './LargeDatasets/DemoImage/car/car_subfolder/rendering/black_chair.png'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    #dim = (width, height)
    dim = (224, 224)
    print("Dimensions of input to segmentation code {}".format(dim))
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print("Image Shape = {}".format(img.shape))
    img = img.reshape(1, 1, 3, img.shape[0], img.shape[1])
    print("Image ReShaped = {}".format(img.shape))
    img = torch.from_numpy(img)
    img = img.float().div(255)
    return img