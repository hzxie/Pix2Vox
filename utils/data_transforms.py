#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import torch

class Compose(object):
    """ Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.AddRandomBackground(),
    >>>     transforms.CropCenter(127, 127, 3),
    >>>  ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rendering_images, voxel):
        for t in self.transforms:
            rendering_images, voxel = t(rendering_images, voxel)
        
        return rendering_images, voxel


class ArrayToTensor3d(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    
    def __call__(self, rendering_images, voxel):
        assert(isinstance(rendering_images, np.ndarray))
        array = np.transpose(rendering_images, (0, 3, 1, 2))
        # handle numpy array
        tensor = torch.from_numpy(array)
        
        # put it from HWC to CHW format
        return tensor.float(), voxel


class CropCenter(object):
    def __init__(self, crop_height, crop_width, n_channels):
        ''' Set the height and weight after cropping
        '''
        self.crop_height   = crop_height
        self.crop_width    = crop_width
        self.n_channels    = n_channels

    def __call__(self, rendering_images, voxel):
        processed_images = np.empty(shape=(0, self.crop_height, self.crop_width, self.n_channels))

        for img_idx, img in enumerate(rendering_images):
            img_height, img_width, _ = img.shape

            if img_height <= self.crop_height or img_width <= self.crop_width:
                return rendering_images, voxel

            x_left  = int((img_width - self.crop_width) / 2.)
            x_right = int(x_left + self.crop_width)
            y_left  = int((img_height - self.crop_height) / 2.)
            y_right = int(y_left + self.crop_height)

            processed_images = np.append(processed_images, [img[y_left: y_right, x_left: x_right]], axis=0)
        
        return processed_images, voxel


class AddRandomBackground(object):
    def __init__(self, random_bg_color_range):
        self.random_bg_color_range = random_bg_color_range

    def __call__(self, rendering_images, voxel):
        if len(rendering_images) == 0:
            return rendering_images, voxel

        img_height, img_width, img_channels = rendering_images[0].shape
        if not img_channels == 4:
            return rendering_images, voxel

        processed_images = np.empty(shape=(0, img_height, img_width, img_channels - 1))
        for img_idx, img in enumerate(rendering_images):
            # If the image has the alpha channel, add the background
            r, g, b  = [np.random.randint(self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1) for i in range(3)]
            alpha    = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
            img      = img[:, :, :3]
            bg_color = np.array([[[r, g, b]]])
            img      = alpha * bg_color + (1 - alpha) * img

            processed_images = np.append(processed_images, [img], axis=0)

        return processed_images, voxel

