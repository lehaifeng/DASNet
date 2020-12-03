###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample, normalize
from attention import PAM_Module
from attention import CAM_Module
from resbase import BaseNet
import torch.nn.functional as F

__all__ = ['DANet']


class DANet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """

    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer)

    def forward(self, x):

        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)

        return x[0],x[1],x[2]


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sa_output,sc_output,sasc_output

def cnn():
    model = DANet(512, backbone='resnet50')
    return model

class SiameseNet(nn.Module):
    def __init__(self,norm_flag = 'l2'):
        super(SiameseNet, self).__init__()
        self.CNN = cnn()
        if norm_flag == 'l2':
           self.norm = F.normalize
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()
    '''''''''
    def forward(self,t0,t1):
        out_t0_embedding = self.CNN(t0)
        out_t1_embedding = self.CNN(t1)
        #out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5),self.norm(out_t1_conv5)
        #out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7),self.norm(out_t1_fc7)
        out_t0_embedding_norm,out_t1_embedding_norm = self.norm(out_t0_embedding),self.norm(out_t1_embedding)
        return [out_t0_embedding_norm,out_t1_embedding_norm]
    '''''''''

    def forward(self,t0,t1):



        out_t0_conv5,out_t0_fc7,out_t0_embedding = self.CNN(t0)
        out_t1_conv5,out_t1_fc7,out_t1_embedding = self.CNN(t1)
        out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5,2,dim=1),self.norm(out_t1_conv5,2,dim=1)
        out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7,2,dim=1),self.norm(out_t1_fc7,2,dim=1)
        out_t0_embedding_norm,out_t1_embedding_norm = self.norm(out_t0_embedding,2,dim=1),self.norm(out_t1_embedding,2,dim=1)
        return [out_t0_conv5_norm,out_t1_conv5_norm],[out_t0_fc7_norm,out_t1_fc7_norm],[out_t0_embedding_norm,out_t1_embedding_norm]


if __name__ == '__main__':
    m = SiameseNet()
    print('gg')