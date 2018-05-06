#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(async=True)
    
    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(file_name, epoch_idx, generator, generator_solver, image_encoder, image_encoder_solver):
    torch.save({
        'epoch_idx': epoch_idx,
        'generator_state_dict': generator.state_dict(),
        'generator_solver_state_dict': generator_solver.state_dict(),
        'image_encoder_state_dict': image_encoder.state_dict(),
        'image_encoder_solver_state_dict': image_encoder_solver.state_dict(),
    }, os.path.join(ckpt_dir, file_name))
