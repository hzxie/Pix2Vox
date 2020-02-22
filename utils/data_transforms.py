# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# References:
# - https://github.com/xiumingzhang/GenRe-ShapeHD

import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
import os
import random
import torch


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

    def __call__(self, rendering_images, bounding_box=None):
        for t in self.transforms:
            if t.__class__.__name__ == 'RandomCrop' or t.__class__.__name__ == 'CenterCrop':
                rendering_images = t(rendering_images, bounding_box)
            else:
                rendering_images = t(rendering_images)

        return rendering_images


class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))
        array = np.transpose(rendering_images, (0, 3, 1, 2))
        # handle numpy array
        tensor = torch.from_numpy(array)

        # put it from HWC to CHW format
        return tensor.float()


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))
        rendering_images -= self.mean
        rendering_images /= self.std

        return rendering_images


class RandomPermuteRGB(object):
    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))

        random_permutation = np.random.permutation(3)
        for img_idx, img in enumerate(rendering_images):
            rendering_images[img_idx] = img[..., random_permutation]

        return rendering_images


class CenterCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, rendering_images, bounding_box=None):
        if len(rendering_images) == 0:
            return rendering_images

        crop_size_c = rendering_images[0].shape[2]
        processed_images = np.empty(shape=(0, self.img_size_h, self.img_size_w, crop_size_c))
        for img_idx, img in enumerate(rendering_images):
            img_height, img_width, _ = img.shape

            if bounding_box is not None:
                bounding_box = [
                    bounding_box[0] * img_width,
                    bounding_box[1] * img_height,
                    bounding_box[2] * img_width,
                    bounding_box[3] * img_height
                ]  # yapf: disable

                # Calculate the size of bounding boxes
                bbox_width = bounding_box[2] - bounding_box[0]
                bbox_height = bounding_box[3] - bounding_box[1]
                bbox_x_mid = (bounding_box[2] + bounding_box[0]) * .5
                bbox_y_mid = (bounding_box[3] + bounding_box[1]) * .5

                # Make the crop area as a square
                square_object_size = max(bbox_width, bbox_height)
                x_left = int(bbox_x_mid - square_object_size * .5)
                x_right = int(bbox_x_mid + square_object_size * .5)
                y_top = int(bbox_y_mid - square_object_size * .5)
                y_bottom = int(bbox_y_mid + square_object_size * .5)

                # If the crop position is out of the image, fix it with padding
                pad_x_left = 0
                if x_left < 0:
                    pad_x_left = -x_left
                    x_left = 0
                pad_x_right = 0
                if x_right >= img_width:
                    pad_x_right = x_right - img_width + 1
                    x_right = img_width - 1
                pad_y_top = 0
                if y_top < 0:
                    pad_y_top = -y_top
                    y_top = 0
                pad_y_bottom = 0
                if y_bottom >= img_height:
                    pad_y_bottom = y_bottom - img_height + 1
                    y_bottom = img_height - 1

                # Padding the image and resize the image
                processed_image = np.pad(img[y_top:y_bottom + 1, x_left:x_right + 1],
                                         ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)),
                                         mode='edge')
                processed_image = cv2.resize(processed_image, (self.img_size_w, self.img_size_h))
            else:
                if img_height > self.crop_size_h and img_width > self.crop_size_w:
                    x_left = int(img_width - self.crop_size_w) // 2
                    x_right = int(x_left + self.crop_size_w)
                    y_top = int(img_height - self.crop_size_h) // 2
                    y_bottom = int(y_top + self.crop_size_h)
                else:
                    x_left = 0
                    x_right = img_width
                    y_top = 0
                    y_bottom = img_height

                processed_image = cv2.resize(img[y_top:y_bottom, x_left:x_right], (self.img_size_w, self.img_size_h))

            processed_images = np.append(processed_images, [processed_image], axis=0)
            # Debug
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(img)
            # if not bounding_box is None:
            #     rect = patches.Rectangle((bounding_box[0], bounding_box[1]),
            #                              bbox_width,
            #                              bbox_height,
            #                              linewidth=1,
            #                              edgecolor='r',
            #                              facecolor='none')
            #     ax1.add_patch(rect)
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(processed_image)
            # plt.show()
        return processed_images


class RandomCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, rendering_images, bounding_box=None):
        if len(rendering_images) == 0:
            return rendering_images

        crop_size_c = rendering_images[0].shape[2]
        processed_images = np.empty(shape=(0, self.img_size_h, self.img_size_w, crop_size_c))
        for img_idx, img in enumerate(rendering_images):
            img_height, img_width, _ = img.shape

            if bounding_box is not None:
                bounding_box = [
                    bounding_box[0] * img_width,
                    bounding_box[1] * img_height,
                    bounding_box[2] * img_width,
                    bounding_box[3] * img_height
                ]  # yapf: disable

                # Calculate the size of bounding boxes
                bbox_width = bounding_box[2] - bounding_box[0]
                bbox_height = bounding_box[3] - bounding_box[1]
                bbox_x_mid = (bounding_box[2] + bounding_box[0]) * .5
                bbox_y_mid = (bounding_box[3] + bounding_box[1]) * .5

                # Make the crop area as a square
                square_object_size = max(bbox_width, bbox_height)
                square_object_size = square_object_size * random.uniform(0.8, 1.2)

                x_left = int(bbox_x_mid - square_object_size * random.uniform(.4, .6))
                x_right = int(bbox_x_mid + square_object_size * random.uniform(.4, .6))
                y_top = int(bbox_y_mid - square_object_size * random.uniform(.4, .6))
                y_bottom = int(bbox_y_mid + square_object_size * random.uniform(.4, .6))

                # If the crop position is out of the image, fix it with padding
                pad_x_left = 0
                if x_left < 0:
                    pad_x_left = -x_left
                    x_left = 0
                pad_x_right = 0
                if x_right >= img_width:
                    pad_x_right = x_right - img_width + 1
                    x_right = img_width - 1
                pad_y_top = 0
                if y_top < 0:
                    pad_y_top = -y_top
                    y_top = 0
                pad_y_bottom = 0
                if y_bottom >= img_height:
                    pad_y_bottom = y_bottom - img_height + 1
                    y_bottom = img_height - 1

                # Padding the image and resize the image
                processed_image = np.pad(img[y_top:y_bottom + 1, x_left:x_right + 1],
                                         ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)),
                                         mode='edge')
                processed_image = cv2.resize(processed_image, (self.img_size_w, self.img_size_h))
            else:
                if img_height > self.crop_size_h and img_width > self.crop_size_w:
                    x_left = int(img_width - self.crop_size_w) // 2
                    x_right = int(x_left + self.crop_size_w)
                    y_top = int(img_height - self.crop_size_h) // 2
                    y_bottom = int(y_top + self.crop_size_h)
                else:
                    x_left = 0
                    x_right = img_width
                    y_top = 0
                    y_bottom = img_height

                processed_image = cv2.resize(img[y_top:y_bottom, x_left:x_right], (self.img_size_w, self.img_size_h))

            processed_images = np.append(processed_images, [processed_image], axis=0)

        return processed_images


class RandomFlip(object):
    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))

        for img_idx, img in enumerate(rendering_images):
            if random.randint(0, 1):
                rendering_images[img_idx] = np.fliplr(img)

        return rendering_images


class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, rendering_images):
        if len(rendering_images) == 0:
            return rendering_images

        # Allocate new space for storing processed images
        img_height, img_width, img_channels = rendering_images[0].shape
        processed_images = np.empty(shape=(0, img_height, img_width, img_channels))

        # Randomize the value of changing brightness, contrast, and saturation
        brightness = 1 + np.random.uniform(low=-self.brightness, high=self.brightness)
        contrast = 1 + np.random.uniform(low=-self.contrast, high=self.contrast)
        saturation = 1 + np.random.uniform(low=-self.saturation, high=self.saturation)

        # Randomize the order of changing brightness, contrast, and saturation
        attr_names = ['brightness', 'contrast', 'saturation']
        attr_values = [brightness, contrast, saturation]    # The value of changing attrs
        attr_indexes = np.array(range(len(attr_names)))    # The order of changing attrs
        np.random.shuffle(attr_indexes)

        for img_idx, img in enumerate(rendering_images):
            processed_image = img
            for idx in attr_indexes:
                processed_image = self._adjust_image_attr(processed_image, attr_names[idx], attr_values[idx])

            processed_images = np.append(processed_images, [processed_image], axis=0)
            # print('ColorJitter', np.mean(ori_img), np.mean(processed_image))
            # fig = plt.figure(figsize=(8, 4))
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(ori_img)
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(processed_image)
            # plt.show()
        return processed_images

    def _adjust_image_attr(self, img, attr_name, attr_value):
        """
        Adjust or randomize the specified attribute of the image

        Args:
            img: Image in BGR format
                Numpy array of shape (h, w, 3)
            attr_name: Image attribute to adjust or randomize
                       'brightness', 'saturation', or 'contrast'
            attr_value: the alpha for blending is randomly drawn from [1 - d, 1 + d]

        Returns:
            Output image in BGR format
            Numpy array of the same shape as input
        """
        gs = self._bgr_to_gray(img)

        if attr_name == 'contrast':
            img = self._alpha_blend(img, np.mean(gs[:, :, 0]), attr_value)
        elif attr_name == 'saturation':
            img = self._alpha_blend(img, gs, attr_value)
        elif attr_name == 'brightness':
            img = self._alpha_blend(img, 0, attr_value)
        else:
            raise NotImplementedError(attr_name)
        return img

    def _bgr_to_gray(self, bgr):
        """
        Convert a RGB image to a grayscale image
            Differences from cv2.cvtColor():
                1. Input image can be float
                2. Output image has three repeated channels, other than a single channel

        Args:
            bgr: Image in BGR format
                 Numpy array of shape (h, w, 3)

        Returns:
            gs: Grayscale image
                Numpy array of the same shape as input; the three channels are the same
        """
        ch = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
        gs = np.dstack((ch, ch, ch))
        return gs

    def _alpha_blend(self, im1, im2, alpha):
        """
        Alpha blending of two images or one image and a scalar

        Args:
            im1, im2: Image or scalar
                Numpy array and a scalar or two numpy arrays of the same shape
            alpha: Weight of im1
                Float ranging usually from 0 to 1

        Returns:
            im_blend: Blended image -- alpha * im1 + (1 - alpha) * im2
                Numpy array of the same shape as input image
        """
        im_blend = alpha * im1 + (1 - alpha) * im2
        return im_blend


class RandomNoise(object):
    def __init__(self,
                 noise_std,
                 eigvals=(0.2175, 0.0188, 0.0045),
                 eigvecs=((-0.5675, 0.7192, 0.4009), (-0.5808, -0.0045, -0.8140), (-0.5836, -0.6948, 0.4203))):
        self.noise_std = noise_std
        self.eigvals = np.array(eigvals)
        self.eigvecs = np.array(eigvecs)

    def __call__(self, rendering_images):
        alpha = np.random.normal(loc=0, scale=self.noise_std, size=3)
        noise_rgb = \
            np.sum(
                np.multiply(
                    np.multiply(
                        self.eigvecs,
                        np.tile(alpha, (3, 1))
                    ),
                    np.tile(self.eigvals, (3, 1))
                ),
                axis=1
            )

        # Allocate new space for storing processed images
        img_height, img_width, img_channels = rendering_images[0].shape
        assert (img_channels == 3), "Please use RandomBackground to normalize image channels"
        processed_images = np.empty(shape=(0, img_height, img_width, img_channels))

        for img_idx, img in enumerate(rendering_images):
            processed_image = img[:, :, ::-1]    # BGR -> RGB
            for i in range(img_channels):
                processed_image[:, :, i] += noise_rgb[i]

            processed_image = processed_image[:, :, ::-1]    # RGB -> BGR
            processed_images = np.append(processed_images, [processed_image], axis=0)
            # from copy import deepcopy
            # ori_img = deepcopy(img)
            # print(noise_rgb, np.mean(processed_image), np.mean(ori_img))
            # print('RandomNoise', np.mean(ori_img), np.mean(processed_image))
            # fig = plt.figure(figsize=(8, 4))
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(ori_img)
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(processed_image)
            # plt.show()
        return processed_images


class RandomBackground(object):
    def __init__(self, random_bg_color_range, random_bg_folder_path=None):
        self.random_bg_color_range = random_bg_color_range
        self.random_bg_files = []
        if random_bg_folder_path is not None:
            self.random_bg_files = os.listdir(random_bg_folder_path)
            self.random_bg_files = [os.path.join(random_bg_folder_path, rbf) for rbf in self.random_bg_files]

    def __call__(self, rendering_images):
        if len(rendering_images) == 0:
            return rendering_images

        img_height, img_width, img_channels = rendering_images[0].shape
        # If the image has the alpha channel, add the background
        if not img_channels == 4:
            return rendering_images

        # Generate random background
        r, g, b = np.array([
            np.random.randint(self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1) for i in range(3)
        ]) / 255.

        random_bg = None
        if len(self.random_bg_files) > 0:
            random_bg_file_path = random.choice(self.random_bg_files)
            random_bg = cv2.imread(random_bg_file_path).astype(np.float32) / 255.

        # Apply random background
        processed_images = np.empty(shape=(0, img_height, img_width, img_channels - 1))
        for img_idx, img in enumerate(rendering_images):
            alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
            img = img[:, :, :3]
            bg_color = random_bg if random.randint(0, 1) and random_bg is not None else np.array([[[r, g, b]]])
            img = alpha * bg_color + (1 - alpha) * img

            processed_images = np.append(processed_images, [img], axis=0)

        return processed_images
