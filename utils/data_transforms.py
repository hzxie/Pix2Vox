#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
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

    def __call__(self, rendering_images, voxel, bounding_box=None):
        for t in self.transforms:
            if t.__class__.__name__ == 'RandomCrop' or t.__class__.__name__ == 'CenterCrop':
                rendering_images, voxel = t(rendering_images, voxel, bounding_box)
            else:
                rendering_images, voxel = t(rendering_images, voxel)

        return rendering_images, voxel


class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, rendering_images, voxel):
        assert (isinstance(rendering_images, np.ndarray))
        array = np.transpose(rendering_images, (0, 3, 1, 2))
        # handle numpy array
        tensor = torch.from_numpy(array)

        # put it from HWC to CHW format
        return tensor.float(), voxel


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, rendering_images, voxel):
        assert (isinstance(rendering_images, np.ndarray))
        rendering_images -= self.mean
        rendering_images /= self.std

        return rendering_images, voxel


class RandomPermuteRGB(object):
    def __call__(self, rendering_images, voxel):
        assert (isinstance(rendering_images, np.ndarray))

        random_permutation = np.random.permutation(3)
        for img_idx, img in enumerate(rendering_images):
            rendering_images[img_idx] = img[..., random_permutation]

        return rendering_images, voxel


class CenterCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]
        self.crop_size_c = crop_size[2]

    def __call__(self, rendering_images, voxel, bounding_box=None):
        if len(rendering_images) == 0:
            return rendering_images, voxel

        processed_images = np.empty(shape=(0, self.img_size_h, self.img_size_w, self.crop_size_c))
        for img_idx, img in enumerate(rendering_images):
            img_height, img_width, _ = img.shape

            if not bounding_box is None:
                # Calculate the size of bounding boxes
                bbox_width = bounding_box[2] - bounding_box[0]
                bbox_height = bounding_box[3] - bounding_box[1]
                bbox_x_mid = (bounding_box[2] + bounding_box[0]) * 0.5
                bbox_y_mid = (bounding_box[3] + bounding_box[1]) * 0.5

                crop_size_w = bbox_width if bbox_width > bbox_height else bbox_height
                crop_size_h = bbox_width if bbox_width > bbox_height else bbox_height

                # Make the crop area as a square
                x_left = bbox_x_mid - crop_size_w * 0.5
                x_right = x_left + crop_size_w
                y_top = bbox_y_mid - crop_size_h * 0.5
                y_bottom = y_top + crop_size_h

                # If the crop position is out of the image, fix it
                if x_left < 0:
                    x_left = 0
                    x_right = crop_size_w
                elif x_right > img_width:
                    x_left = img_width - crop_size_w if img_width > crop_size_w else 0
                    x_right = img_width
                if y_top < 0:
                    y_top = 0
                    y_bottom = crop_size_h
                elif y_bottom > img_height:
                    y_top = img_height - crop_size_h if img_height > crop_size_h else 0
                    y_bottom = img_height
            else:
                if img_height > self.crop_size_h and img_width > self.crop_size_w:
                    x_left = (img_width - self.crop_size_w) * 0.5
                    x_right = x_left + self.crop_size_w
                    y_top = (img_height - self.crop_size_h) * 0.5
                    y_bottom = y_top + self.crop_size_h
                else:
                    x_left = 0
                    x_right = img_width
                    y_top = 0
                    y_bottom = img_height

            processed_image = cv2.resize(
                img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
            processed_images = np.append(processed_images, [processed_image], axis=0)

        return processed_images, voxel


class RandomCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]
        self.crop_size_c = crop_size[2]

    def __call__(self, rendering_images, voxel, bounding_box=None):
        if len(rendering_images) == 0:
            return rendering_images, voxel

        processed_images = np.empty(shape=(0, self.img_size_h, self.img_size_w, self.crop_size_c))
        for img_idx, img in enumerate(rendering_images):
            img_height, img_width, _ = img.shape

            if not bounding_box is None:
                # Random move bounding boxes
                for i in range(4):
                    bounding_box[i] += random() * 100 - 50
                    if bounding_box[i] < 0:
                        bounding_box[i] = 0
                    if (i == 0 or i == 2) and bounding_box[i] > img_width:
                        bounding_box[i] = img_width - 50
                    if (i == 1 or i == 3) and bounding_box[i] > img_height:
                        bounding_box[i] = img_height - 50

                if bounding_box[2] <= bounding_box[0]:
                    bounding_box[2] = bounding_box[0] + 50
                if bounding_box[3] <= bounding_box[1]:
                    bounding_box[3] = bounding_box[1] + 50

                # Calculate the size of bounding boxes
                bbox_width = bounding_box[2] - bounding_box[0]
                bbox_height = bounding_box[3] - bounding_box[1]
                bbox_x_mid = (bounding_box[2] + bounding_box[0]) * 0.5
                bbox_y_mid = (bounding_box[3] + bounding_box[1]) * 0.5

                crop_size_w = bbox_width if bbox_width > bbox_height else bbox_height
                crop_size_h = bbox_width if bbox_width > bbox_height else bbox_height

                # Make the crop area as a square
                x_left = bbox_x_mid - crop_size_w * 0.5
                x_right = x_left + crop_size_w
                y_top = bbox_y_mid - crop_size_h * 0.5
                y_bottom = y_top + crop_size_h

                # If the crop position is out of the image, fix it
                if x_left < 0:
                    x_left = 0
                    x_right = crop_size_w
                elif x_right > img_width:
                    x_left = img_width - crop_size_w if img_width > crop_size_w else 0
                    x_right = img_width
                if y_top < 0:
                    y_top = 0
                    y_bottom = crop_size_h
                elif y_bottom > img_height:
                    y_top = img_height - crop_size_h if img_height > crop_size_h else 0
                    y_bottom = img_height
            else:
                if img_height > self.crop_size_h and img_width > self.crop_size_w:
                    x_left = (img_width - self.crop_size_w) * random()
                    x_right = x_left + self.crop_size_w
                    y_top = (img_height - self.crop_size_h) * random()
                    y_bottom = y_top + self.crop_size_h
                else:
                    x_left = 0
                    x_right = img_width
                    y_top = 0
                    y_bottom = img_height

            processed_image = cv2.resize(
                img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
            processed_images = np.append(processed_images, [processed_image], axis=0)
            # Debug
            # fig = plt.figure(figsize=(8, 4))
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(img.astype(np.uint8))
            # if not bounding_box is None:
            #     rect = patches.Rectangle((bounding_box[0], bounding_box[1]), bbox_width, bbox_height, linewidth=1, edgecolor='r', facecolor='none')
            #     ax1.add_patch(rect)

            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(processed_image.astype(np.uint8))
            # plt.show()

        return processed_images, voxel


class RandomAffine(object):
    def __init__(self, rotate_degree_range, translation_range, scale_range):
        self._random_affine = torchvision.transforms.RandomAffine(rotate_degree_range, translation_range, scale_range)

    def __call__(self, rendering_images, voxel):
        if len(rendering_images) == 0:
            return rendering_images, voxel

        img_height, img_width, img_channels = rendering_images[0].shape
        processed_images = np.empty(shape=(0, img_height, img_width, img_channels))
        for img_idx, img in enumerate(rendering_images):
            processed_image = np.array(self._random_affine(Image.fromarray(np.uint8(img * 255))))
            processed_images = np.append(processed_images, [processed_image], axis=0)

        return processed_images, voxel


class RandomFlip(object):
    def __call__(self, rendering_images, voxel):
        assert (isinstance(rendering_images, np.ndarray))

        for img_idx, img in enumerate(rendering_images):
            if random() > 0.5:
                rendering_images[img_idx] = np.fliplr(img)

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
            processed_image = np.array(self._color_jitter(Image.fromarray(np.uint8(img * 255))))
            processed_images = np.append(processed_images, [processed_image], axis=0)

        return processed_images, voxel


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
            r, g, b = [
                np.random.randint(self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1)
                for i in range(3)
            ]
            alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
            img = img[:, :, :3]
            bg_color = np.array([[[r, g, b]]])
            img = alpha * bg_color + (1 - alpha) * img

            processed_images = np.append(processed_images, [img], axis=0)

        return processed_images, voxel
