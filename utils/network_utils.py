#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
import torch

from datetime import datetime as dt

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(async=True)
    
    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(file_path, epoch_idx, encoder, encoder_solver, \
        decoder, decoder_solver, refiner, refiner_solver, best_iou, best_epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    torch.save({
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'encoder_solver_state_dict': encoder_solver.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'decoder_solver_state_dict': decoder_solver.state_dict(),
        'refiner_state_dict': refiner.state_dict(), 
        'refiner_solver_state_dict': refiner_solver.state_dict()
    }, file_path)
