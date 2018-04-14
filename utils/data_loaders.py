#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset

from datetime import datetime as dt

class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    ''' ShapeNetDataset class used for PyTorch DataLoader
    '''
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rendering_images, voxel = self.get_datum(idx)

        if self.transform:
            rendering_images, voxel = self.transform(rendering_images, voxel)
        
        return rendering_images, voxel

    def get_datum(self, idx):
        rendering_image_paths = self.file_list[idx]['rendering_images']
        voxel_path = self.file_list[idx]['voxel']

        # Get data of rendering images
        rendering_images = []
        for image_path in rendering_image_paths:
            rendering_image = scipy.ndimage.imread(image_path)
            if len(rendering_image.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' % (dt.now(), image_path))
                sys.exit(-2)

            rendering_images.append(rendering_image)
            # For Debug
            plt.show(rendering_image)

        # Get data of voxel
        voxel = scipy.io.loadmat(voxel_path)
        if not voxel:
            print('[FATAL] %s Failed to get voxel data from file %s' % (dt.now(), voxel_path))
            sys.exit(-2)

        voxel = voxel['Volume']
        return np.asarray(rendering_images), voxel

# //////////////////////////////// = End of ShapeNetDataset Class Definition = //////////////////////////////// #

class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.dataset_query_path = cfg.DIR.DATASET_QUERY_PATH
        self.rendering_image_path_template = cfg.DIR.RENDERING_PATH
        self.voxel_path_template = cfg.DIR.VOXEL_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DIR.DATASET_TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_portion, n_views, transform=None):
        files = []
        
        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_folder_name = taxonomy['taxonomy_id']

            print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' % (dt.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
            files.append(self.get_files_of_taxonomy(taxonomy_folder_name, dataset_portion, n_views))

        return ShapeNetDataset(files, transform)

    def get_files_of_taxonomy(self, taxonomy_folder_name, dataset_portion, n_views):
        files_of_taxonomy = []
        samples = self.get_samples_of_taxonomy(taxonomy_folder_name, dataset_portion)

        for sample_name in samples:
            rendering_images_folder = self.rendering_image_path_template % (taxonomy_folder_name, sample_name)

            file = {
                'rendering_images': [],
                'voxel': self.voxel_path_template % (taxonomy_folder_name, sample_name)
            }
            # Get file list of rendering images
            for i in range(n_views):    # TODO, use random or other strategies according to viewpoints
                file['rendering_images'].append(os.path.join(rendering_images_folder, '%02d.png' % (i + 1)))

            # TODO: Meta data

            files_of_taxonomy.append(file)

        return files_of_taxonomy

    def get_samples_of_taxonomy(self, taxonomy_folder_name, dataset_portion):
        ''' Get the name list of samples of a taxonomy
        '''
        taxonomy_folder = os.path.join(self.dataset_query_path, taxonomy_folder_name)
        samples = [sample_name for sample_name in os.listdir(taxonomy_folder) 
                    if os.path.isdir(os.path.join(taxonomy_folder, sample_name))]
        n_samples = len(samples)

        return samples[int(n_samples * dataset_portion[0]):int(n_samples * dataset_portion[1])]

# //////////////////////////////// = End of ShapeNetDataGetter Class Definition = //////////////////////////////// #
