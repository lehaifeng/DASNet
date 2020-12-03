import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from attention import *
# affine_par = True  # True: BN has learnable affine parameters, False: without learnable affine parameters of BatchNorm Layer


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        for i in self.bn1.parameters():
            i.requires_grad = True

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        for i in self.bn2.parameters():
            i.requires_grad = True
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=True)
        for i in self.bn3.parameters():
            i.requires_grad = True
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Residual_Covolution(nn.Module):
    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg



class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        for i in self.bn1.parameters():
            i.requires_grad = True
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        inter_channels = 2048 // 4
        self.conv5a = nn.Sequential(nn.Conv2d(2048, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(2048, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 512, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 512, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 512, 1))
        self.embedding_layer = nn.Conv2d(in_channels=2048,out_channels=512,kernel_size=1)

        #self.embedding_layer = nn.Conv2d(512, num_classes, kernel_size=1)
        # self.softmax = nn.Softmax()

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = True
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)  # 7x7Conv
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)  # res2
        x = self.layer2(x)  # res3
        x = self.layer3(x)  # res4
        conv_feature = self.layer4(x)  # res5
        feat1 = self.conv5a(conv_feature)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)
        feat2 = self.conv5c(conv_feature)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)
        # embedding_feature = self.embedding_layer(conv_feature)


        # return sa_output,sc_output, sasc_output
        return sa_output, sc_output,sasc_output  #


def PSPNet():
    """ """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

class SiameseNet(nn.Module):
    def __init__(self,norm_flag = 'l2'):
        super(SiameseNet, self).__init__()
        self.CNN = ResNet(Bottleneck, [3, 4, 6, 3])
        if norm_flag == 'l2':
           self.norm = F.normalize
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()

    def forward(self, t0, t1):


        out_t0_conv5,out_t0_fc7,out_t0_embedding = self.CNN(t0)
        out_t1_conv5,out_t1_fc7,out_t1_embedding = self.CNN(t1)
        out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5,2,dim=1),self.norm(out_t1_conv5,2,dim=1)
        out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7,2,dim=1),self.norm(out_t1_fc7,2,dim=1)
        out_t0_embedding_norm,out_t1_embedding_norm = self.norm(out_t0_embedding,2,dim=1),self.norm(out_t1_embedding,2,dim=1)
        return [out_t0_conv5_norm,out_t1_conv5_norm],[out_t0_fc7_norm,out_t1_fc7_norm],[out_t0_embedding_norm,out_t1_embedding_norm]

        # out_t0_conv5, out_t0_embedding = self.CNN(t0)
        # out_t1_conv5, out_t1_embedding = self.CNN(t1)
        # out_t0_conv5_norm, out_t1_conv5_norm = self.norm(out_t0_conv5, 2, dim=1), self.norm(out_t1_conv5, 2, dim=1)
        # out_t0_embedding_norm, out_t1_embedding_norm = self.norm(out_t0_embedding, 2, dim=1), self.norm(out_t1_embedding, 2,
        #                                                                                                 dim=1)
        # return [out_t0_conv5_norm, out_t1_conv5_norm],  [out_t0_embedding_norm,out_t1_embedding_norm]