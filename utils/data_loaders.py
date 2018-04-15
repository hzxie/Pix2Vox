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
    def __init__(self, file_list_with_metadata, transform=None):
        self.file_list = file_list_with_metadata
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
            files.extend(self.get_files_of_taxonomy(taxonomy_folder_name, dataset_portion, n_views))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return ShapeNetDataset(files, transform)

    def get_files_of_taxonomy(self, taxonomy_folder_name, dataset_portion, n_views):
        files_of_taxonomy = []
        samples = self.get_samples_of_taxonomy(taxonomy_folder_name, dataset_portion)
        n_samples = len(samples)

        for sample_idx, sample_name in enumerate(samples):
            rendering_images_folder = self.rendering_image_path_template % (taxonomy_folder_name, sample_name)

            # Get meta data of rendering images
            rendering_positions = []
            with open(os.path.join(rendering_images_folder, 'rendering_metadata.txt'), encoding='utf-8') as file:
                rendering_metadata = file.readlines()

                for line_idx, line in enumerate(rendering_metadata):
                    values = line.split(' ')
                    if len(values) == 5:
                        rendering_positions.append({
                            'position': float(values[0]),
                            'index': line_idx
                        })
            
            if not len(rendering_positions) == 24:
                print('[WARN] There are less than 24 rendering images for this 3D object: %s' % rendering_images_folder)
            # Sort list of rendering images by positions
            rendering_positions.sort(key=lambda x: x['position'])

            # Get file list of rendering images
            selected_rendering_image_indexes = [int(x) for x in np.linspace(0, len(rendering_positions) - 1, num=n_views)]
            selected_rendering_images = []
            selected_rendering_image_positions = []
            for image_idx in selected_rendering_image_indexes:
                selected_rendering_images.append(os.path.join(rendering_images_folder, '%02d.png' % image_idx))
                selected_rendering_image_positions.append(rendering_positions[image_idx])

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'rendering_images': selected_rendering_images,
                'rendering_positions': selected_rendering_image_positions,
                'voxel': self.voxel_path_template % (taxonomy_folder_name, sample_name)
            })

            # Report the progress of reading dataset
            if sample_idx % 500 == 499 or sample_idx == n_samples - 1:
                print('[INFO] %s Collecting %d of %d' % (dt.now(), sample_idx + 1, n_samples))

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
