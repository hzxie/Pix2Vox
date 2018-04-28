#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import scipy.ndimage

from mpl_toolkits.mplot3d import Axes3D

# Force matplotlib to not use any XWindow backend
matplotlib.use('Agg')

def get_voxel_views(voxels, save_dir, n_itr):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    
    save_path = os.path.join(save_dir, 'voxels-%06d.png' % n_itr)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return scipy.ndimage.imread(save_path)
