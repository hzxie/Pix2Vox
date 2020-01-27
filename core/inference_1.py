import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms
from torchvision import datasets, transforms

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
import tkinter

from skimage import io, transform
import cv2
import matplotlib.pyplot as plt

def inference_net(cfg,output_dir=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    IMG_SIZE = 224, 224
    CROP_SIZE = 128, 128

    # I need this for performing the data transformations
    # for example I need input images to be of certain size etc
    test_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

    dataset = datasets.ImageFolder('./LargeDatasets/inference_images/', transform=test_transforms)
    print(len(dataset))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    print(type(dataloader))

    images = np.empty(shape=(0, 3, 224, 224))
    for image, label in dataloader:
        print("shape of image {}".format(image.shape))
        #images.append(image)
        images = np.append(images, [np.asarray(image.view(3, 224,224))], axis=0)

    print("shape of images {}".format(images.shape))
    images = np.reshape(images,(1,images.shape[0],3,224,224))
    rendering_images = torch.from_numpy(images)
    print("size of rendering images {}".format(rendering_images.shape))

    """
    rendering_images = np.asanyarray(images)
    
    processed_images = np.append(processed_images, [img], axis=0)
    print("size of all images {}".format(images.shape))
    """

    images, label = next(iter(dataloader))
    print(type(images.size))
    
    fig, ax = plt.subplots(figsize=(5, 3.5), nrows=1, ncols=1)
    plt.imshow(images.view(224,224,3))
    plt.savefig('image1234.png')

    print(images.shape)
    rendering_image = images.view(1,1,3,224,224)

    #Setup Model
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)

    # This has something to do with using cuda for GPU systems
    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()

    # This is just saying that the weights are being loaded
    # from the path 'cfg.CONST.WEIGHTS' at current time
    print('[INFO] Loading weights from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    epoch_idx = checkpoint['epoch_idx']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    if cfg.NETWORK.USE_REFINER:
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
    if cfg.NETWORK.USE_MERGER:
        merger.load_state_dict(checkpoint['merger_state_dict'])

    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    with torch.no_grad():
        # Get data from data loader
        rendering_image = utils.network_utils.var_or_cuda(rendering_images)
        print("shape of rendering images after cuda {}".format(rendering_image.shape))

        # Test the encoder, decoder, refiner and merger
        image_features = encoder(rendering_image)
        print("shape of image features {}".format(image_features.shape))
        raw_features, generated_volume = decoder(image_features)
        print("shape of decoded generated volume {}".format(generated_volume.shape))
        print("shape of raw features {}".format(raw_features.shape))

        generated_volume = merger(raw_features, generated_volume)
        generated_volume = refiner(generated_volume)

        gv = generated_volume.cpu().numpy()
        print('gv shape is {}'.format(gv.shape))
        rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join('./LargeDatasets/inference_images/', 'inference'),
                                                                      1)
        gen_vol = gv.reshape(32,32,32)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(gen_vol, edgecolor='k')
        plt.savefig('image22222.png')



