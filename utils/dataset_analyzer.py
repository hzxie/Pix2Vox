#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import numpy as np
import os
import scipy.ndimage
import sys

from datetime import datetime as dt
from fnmatch import fnmatch
from queue import Queue


def main():
    if not len(sys.argv) == 2:
        print('python dataset_analyzer.py input_file_folder')
        sys.exit(1)

    input_file_folder = sys.argv[1]
    if not os.path.exists(input_file_folder) or not os.path.isdir(input_file_folder):
        print('[ERROR] Input folder not exists!')
        sys.exit(2)

    FILE_NAME_PATTERN = '*.JPEG'
    folders_to_explore = Queue()
    folders_to_explore.put(input_file_folder)

    total_files = 0
    mean = np.asarray([0., 0., 0.])
    std = np.asarray([0., 0., 0.])
    while not folders_to_explore.empty():
        current_folder = folders_to_explore.get()

        if not os.path.exists(current_folder) or not os.path.isdir(current_folder):
            print('[WARN] %s Ignore folder: %s' % (dt.now(), current_folder))
            continue

        print('[INFO] %s Listing files in folder: %s' % (dt.now(), current_folder))
        n_folders = 0
        n_files = 0
        files = os.listdir(current_folder)
        for file_name in files:
            file_path = os.path.join(current_folder, file_name)
            if os.path.isdir(file_path):
                n_folders += 1
                folders_to_explore.put(file_path)
            elif os.path.isfile(file_path) and fnmatch(file_name, FILE_NAME_PATTERN):
                n_files += 1
                total_files += 1

                img = scipy.ndimage.imread(file_path)
                img_mean = np.mean(img, axis=(0, 1))
                img_std = np.var(img, axis=(0, 1))
                mean += img_mean
                std += img_std
        # print('[INFO] %s %d folders found, %d files found.' % (dt.now(), n_folders, n_files))
    print('[INFO] %s Mean = %s, Std = %s' % (dt.now(), mean / total_files, np.sqrt(std) / total_files))


if __name__ == '__main__':
    main()
