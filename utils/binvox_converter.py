#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# This script is used to convert OFF format to binvox.
# Please make sure that you have `binvox` installed.
# You can get it in http://www.patrickmin.com/binvox/

import numpy as np
import os
import subprocess
import sys

from datetime import datetime as dt
from glob import glob

import binvox_rw


def main():
    if not len(sys.argv) == 2:
        print('python binvox_converter.py input_file_folder')
        sys.exit(1)

    input_file_folder = sys.argv[1]
    if not os.path.exists(input_file_folder) or not os.path.isdir(input_file_folder):
        print('[ERROR] Input folder not exists!')
        sys.exit(2)

    N_VOX = 32
    MESH_EXTENSION = '*.off'

    folder_path = os.path.join(input_file_folder, MESH_EXTENSION)
    mesh_files = glob(folder_path)

    for m_file in mesh_files:
        file_path = os.path.join(input_file_folder, m_file)
        file_name, ext = os.path.splitext(m_file)
        binvox_file_path = os.path.join(input_file_folder, '%s.binvox' % file_name)

        if os.path.exists(binvox_file_path):
            print('[WARN] %s File: %s exists. It will be overwritten.' % (dt.now(), binvox_file_path))
            os.remove(binvox_file_path)

        print('[INFO] %s Processing file: %s' % (dt.now(), file_path))
        rc = subprocess.call(['binvox', '-d', str(N_VOX), '-e', '-cb', '-rotx', '-rotx', '-rotx', '-rotz', m_file])
        if not rc == 0:
            print('[WARN] %s Failed to convert file: %s' % (dt.now(), m_file))
            continue

        with open(binvox_file_path, 'rb') as file:
            v = binvox_rw.read_as_3d_array(file)

        v.data = np.transpose(v.data, (2, 0, 1))
        with open(binvox_file_path, 'wb') as file:
            binvox_rw.write(v, file)


if __name__ == '__main__':
    return_code = subprocess.call(['which', 'binvox'], stdout=subprocess.PIPE)
    if return_code == 0:
        main()
    else:
        print('[FATAL] %s Please make sure you have binvox installed.' % dt.now())
