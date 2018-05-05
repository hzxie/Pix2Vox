#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(async=True)
    
    return x
