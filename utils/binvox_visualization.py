#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.ndimage

from mpl_toolkits.mplot3d import Axes3D

# Force matplotlib to not use any XWindow backend
matplotlib.use('Agg')
plt.style.use("ggplot")

def get_rendering_images(voxel, save_path):
    figure  = plt.figure()
    axis    = figure.gca(projection='3d')
    axis.voxels(voxel, edgecolor='k')

    plt.savefig(save_path)
    return scipy.ndimage.imread(save_path)

def get_voxel_views(voxel, save_dir, n_itr):
    rendering_view_pixels = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # View #1
    save_path = os.path.join(save_dir, 'Voxel-Rendering-itr-%d-00.png' % n_itr)
    rendering_view_pixels.append(get_rendering_images(voxel, save_path))

    # View #2
    save_path = os.path.join(save_dir, 'Voxel-Rendering-itr-%d-01.png' % n_itr)
    rendering_view_pixels.append(get_rendering_images(np.transpose(voxel, (0, 2, 1)), save_path))

    # View #3
    save_path = os.path.join(save_dir, 'Voxel-Rendering-itr-%d-02.png' % n_itr)
    rendering_view_pixels.append(get_rendering_images(np.transpose(voxel, (1, 0, 2)), save_path))

    return rendering_view_pixels
