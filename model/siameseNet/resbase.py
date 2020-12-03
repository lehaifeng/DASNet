###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter

import resnet

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet']

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, dilated=True, norm_layer=None,root='./pretrain_models',
                 multi_grid=False, multi_dilation=None):
        super(BaseNet, self).__init__()

        # copying modules from pretrained models
        if backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=False, dilated=dilated,
                                              norm_layer=norm_layer, root=root,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer, root=root,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer, root=root,
                                               multi_grid=multi_grid,multi_dilation=multi_dilation)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False, dilated=dilated,
                                               norm_layer=norm_layer, root=root,
                                               multi_grid=multi_grid, multi_dilation=multi_dilation)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4




