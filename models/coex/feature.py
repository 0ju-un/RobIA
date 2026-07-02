#!/usr/bin/env python
from __future__ import print_function
import os
import numpy as np
from typing import Callable, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Submodule import SubModule
from .util_conv import BasicConv, Conv2x, BasicBlock, conv3x3, conv1x1, BasicConvCustomMoE, Conv2xCustomMoE

from .util_conv import BasicConv2d, BasicTransposeConv2d

from ..backbones import _gen_mobilenet_v2

import timm

import pdb



def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class PSMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(PSMBasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class FeatUp(SubModule):
    def __init__(self, cfg):
        super(FeatUp, self).__init__()
        self.cfg = cfg['backbone']
        self.type = self.cfg['type']
        chans = self.cfg['channels'][self.type]

        if not self.type == 'psm':
            self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
            self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
            self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
            self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

            self.weight_init()

    def forward(self, featL, featR=None, *args, **kwargs):
        x4, x8, x16, x32 = featL

        if self.type == 'psm':
            return featL, featR

        if featR is not None:
            y4, y8, y16, y32 = featR

            x16 = self.deconv32_16(x32, x16)
            y16 = self.deconv32_16(y32, y16)

            x8 = self.deconv16_8(x16, x8)
            y8 = self.deconv16_8(y16, y8)

            x4 = self.deconv8_4(x8, x4)
            y4 = self.deconv8_4(y8, y4)

            x4 = self.conv4(x4)
            y4 = self.conv4(y4)

            return [x4, x8, x16, x32], [y4, y8, y16, y32]
        else:
            x16 = self.deconv32_16(x32, x16)
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x4 = self.conv4(x4)
            return [x4, x8, x16, x32]


class Feature(SubModule):
    def __init__(self, cfg):
        super(Feature, self).__init__()
        self.cfg = cfg['backbone']
        self.type = self.cfg['type']
        chans = self.cfg['channels'][self.type]
        layers = self.cfg['layers'][self.type]

        kwargs = {}
        kwargs['gate_type'] = 'rowwise_self_attention_pool'

        pretrained = False if self.cfg['from_scratch'] else True

        model = _gen_mobilenet_v2(self.type, 1.0, pretrained=pretrained, features_only=True, **kwargs)

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])
    def forward(self, x, *args, **kwargs):
        x = self.bn1(self.conv_stem(x))
        x2 = self.block0(x)
        x4 = self.block1(x2)

        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x_out = [x4, x8, x16, x32]

        return x2, x_out
        
class FeatUpCustomMoE(FeatUp):
    def __init__(self, cfg, gate_type='linear'):
        super(FeatUpCustomMoE, self).__init__(cfg)
        self.up_moe_def = [False, False, False, True]
        chans = self.cfg['channels'][self.type]
        gate_type = 'rowwise_self_attention_pool'
        activation = 'sigmoid'
        conv = []

        if self.up_moe_def[0]:
            conv.append(lambda **kwargs: BasicConvCustomMoE(**kwargs, gate_type=gate_type, activation=activation))
        else:
            conv.append(BasicConv)

        for i, use_moe in enumerate(self.up_moe_def[1:]):
            if use_moe:
                conv.append(lambda i=i, **kwargs: Conv2xCustomMoE(**kwargs, gate_type=gate_type,activation=activation))
            else:
                conv.append(Conv2x)
        print(f'\n\nconv::\n{conv}\n\n')

        self.deconv32_16 = conv[3](in_channels=chans[4], out_channels=chans[3], deconv=True, concat=True)
        self.deconv16_8 = conv[2](in_channels=chans[3]*2, out_channels=chans[2], deconv=True, concat=True)
        self.deconv8_4 = conv[1](in_channels=chans[2]*2, out_channels=chans[1], deconv=True, concat=True)
        self.conv4 = conv[0](in_channels=chans[1]*2, out_channels=chans[1]*2, kernel_size=3, stride=1, padding=1)



    
    def forward(self, featL, featR=None):

        x_feats = list(featL) # x4, x8, x16, x32
        conv = [self.conv4, self.deconv8_4, self.deconv16_8, self.deconv32_16]

        gate_values = []


        for i in range(len(x_feats)-1):
            i_high = 3 - i - 1  # high resolution feature idx
            i_low = 3 - i  # row resolution feature idx (e.g., x32: row, x16: high)
            x_feats[i_high] = conv[i_low](x_feats[i_low], x_feats[i_high])

            if isinstance(x_feats[i_high], tuple):
                x_feats[i_high], gate = x_feats[i_high]
                gate_values.append(gate)
            else:
                gate_values.append([])
        x_feats[0] = conv[0](x_feats[0])
        if isinstance(x_feats[0], tuple):
            x_feats[0], gate = x_feats[0]
            gate_values.append(gate)
        else:
            gate_values.append([])


        return x_feats, gate_values
        
class FeatureCustomMoE(Feature):
    def __init__(self, cfg):
        super(FeatureCustomMoE, self).__init__(cfg)

    def forward(self, x):

        gate_values = []
        x_out = []

        x = self.bn1(self.conv_stem(x))
        for b in self.block0:
            gate_values_b = []
            for b_ in b:
                x = b_(x)
                if isinstance(x, tuple):
                    x, g = x
                    if isinstance(g, list):
                        gate_values_b.extend(g)
                    else:
                        gate_values_b.append(g)
            gate_values.append(gate_values_b)
        x2 = x

        for i in range(1, 5):  # block1 ~ block4
            block = getattr(self, f"block{i}")
            for b in block:
                gate_values_b = []
                for b_ in b:
                    x = b_(x)
                    if isinstance(x, tuple):
                        x, g = x
                        if isinstance(g, list):
                            gate_values_b.extend(g)
                        else:
                            gate_values_b.append(g)
                gate_values.append(gate_values_b)
            x_out.append(x)
        
        return x2, x_out, gate_values


