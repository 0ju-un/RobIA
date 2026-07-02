#!/usr/bin/env python
import torch
import torch.nn.functional as F

from .spixel_utils import spixel
from .Submodule import SubModule

import pdb


class Regression(SubModule):
    def __init__(self,
                 max_disparity=192,
                 top_k=2):
        super(Regression, self).__init__()
        self.D = int(max_disparity//4)
        self.top_k = top_k
        self.ind_init = False

    def forward(self, cost, spg=None, training=False):
        if spg is not None:
            b, _, h, w = spg.shape
        else:
            b,_,_, h, w = cost.shape

        corr, disp = self.topkpool(cost, self.top_k)
        corr = F.softmax(corr, 2)

        disp_4 = torch.sum(corr*disp, 2, keepdim=True)
        disp_4 = disp_4.reshape(b, 1, disp_4.shape[-2], disp_4.shape[-1])

        # !DEBUG
        # disp_1 = (spixel.upfeat(disp_4, spg, 4, 4))
        # # disp_1 = (spixel.upfeatHW(disp_4, spg, h, w))
        if spg is not None:
            # superpixel-based upsampling (from main network)
            disp_1 = spixel.upfeat(disp_4, spg, 4, 4)  # assumes 4x scale
        else:
            # bilinear upsampling (auxiliary usage)
            disp_1 = F.interpolate(disp_4, scale_factor=4, mode='bilinear', align_corners=True)

        disp_1 = disp_1.squeeze(1)*4  # + 1.5

        if training:
            disp_4 = disp_4.squeeze(1)*4  # + 1.5
            return [disp_1, disp_4]
        else:
            return [disp_1]

    def topkpool(self, cost, k):
        if k == 1:
            _, ind = cost.sort(2, True)
            pool_ind_ = ind[:, :, :k]
            b, _, _, h, w = pool_ind_.shape
            pool_ind = pool_ind_.new_zeros((b, 1, 3, h, w))
            pool_ind[:, :, 1:2] = pool_ind_
            pool_ind[:, :, 0:1] = torch.max(
                pool_ind_-1, pool_ind_.new_zeros(pool_ind_.shape))
            pool_ind[:, :, 2:] = torch.min(
                pool_ind_+1, self.D*pool_ind_.new_ones(pool_ind_.shape))
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        else:
            _, ind = cost.sort(2, True)
            pool_ind = ind[:, :, :k]
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        return cv, disp
