#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import numpy as np
import scipy.misc
import torch
import torchvision.transforms

from PIL import Image
from random import random

class Compose(object):
    """ Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.RandomBackground(),
    >>>     transforms.CenterCrop(127, 127, 3),
    >>>  ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rendering_images, voxel):
        for t in self.transforms:
            rendering_images, voxel = t(rendering_images, voxel)
        
        return rendering_images, voxel


class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self):
        pass
    
    def __call__(self, rendering_images, voxel):
        assert(isinstance(rendering_images, np.ndarray))
        array = np.transpose(rendering_images, (0, 3, 1, 2))
        # handle numpy array
        tensor = torch.from_numpy(array)
        
        # put it from HWC to CHW format
        return tensor.float(), voxel


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, rendering_images, voxel):
        assert(isinstance(rendering_images, np.ndarray))
        rendering_images -= self.mean
        rendering_images /= self.std

        return rendering_images, voxel


class CenterCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h   = img_size[0]
        self.img_size_w   = img_size[1]
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, rendering_images, voxel):
        if len(rendering_images) == 0:
            return rendering_images, voxel

        img_height, img_width, img_channels = rendering_images[0].shape
        processed_images = np.empty(shape=(0, self.img_size_h, self.img_size_w, img_channels))
        for img_idx, img in enumerate(rendering_images):
            if img_height <= self.crop_size_h or img_width <= self.crop_size_w:
                return rendering_images, voxel

            x_left  = int((img_width - self.crop_size_w) * 0.5)
            x_right = int(x_left + self.crop_size_w)
            y_left  = int((img_height - self.crop_size_h) * 0.5)
            y_right = int(y_left + self.crop_size_h)

            processed_image  = scipy.misc.imresize(img[y_left: y_right, x_left: x_right], (self.img_size_h, self.img_size_w))
            processed_images = np.append(processed_images, [processed_image], axis=0)
        
        return processed_images, voxel


class RandomCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h   = img_size[0]
        self.img_size_w   = img_size[1]
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, rendering_images, voxel):
        if len(rendering_images) == 0:
            return rendering_images, voxel

        img_height, img_width, img_channels = rendering_images[0].shape
        processed_images = np.empty(shape=(0, self.img_size_h, self.img_size_w, img_channels))
        for img_idx, img in enumerate(rendering_images):
            if img_height <= self.crop_size_h or img_width <= self.crop_size_w:
                return rendering_images, voxel

            x_left  = int((img_width - self.crop_size_w) * random())
            x_right = int(x_left + self.crop_size_w)
            y_left  = int((img_height - self.crop_size_h) * random())
            y_right = int(y_left + self.crop_size_h)

            processed_image  = scipy.misc.imresize(img[y_left: y_right, x_left: x_right], (self.img_size_h, self.img_size_w))
            processed_images = np.append(processed_images, [processed_image], axis=0)
        
        return processed_images, voxel


class RandomAffine(object):
    def __init__(self, rotate_degree_range, translation_range, scale_range):
        self._random_affine = torchvision.transforms.RandomAffine(rotate_degree_range, translation_range, scale_range)

    def __call__(self, rendering_images, voxel):
        # TODO
        
        return rendering_images, voxel


class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self._color_jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, rendering_images, voxel):
        if len(rendering_images) == 0:
            return rendering_images, voxel

        img_height, img_width, img_channels = rendering_images[0].shape
        processed_images = np.empty(shape=(0, img_height, img_width, img_channels))
        for img_idx, img in enumerate(rendering_images):
            processed_image  = np.array(self._color_jitter(Image.fromarray(np.uint8(img * 255))))
            processed_images = np.append(processed_images, [processed_image], axis=0)
        
        return rendering_images, voxel


class RandomBackground(object):
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

