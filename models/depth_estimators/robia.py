from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..coex.feature import FeatUpCustomMoE, FeatureCustomMoE
from ..coex.utils import AttentionCostVolume
from ..coex.regression import Regression
from ..coex.util_conv import BasicConv, Conv2x

from ..losses import *

from .coex import CoEx

class RobIA(CoEx):
    def __init__(self, cfg):
        super(RobIA, self).__init__(cfg)

        # self.gate_type = self.cfg['MoE']['gate_type']
        self.feature = FeatureCustomMoE(self.cfg)
        self.up =  FeatUpCustomMoE(self.cfg)

    def forward(self, imL, imR=None, u0=None, v0=None, training=False):
        if imR is not None:
            assert imL.shape == imR.shape
            imL = torch.cat([imL, imR], 0)
            
        b, c, h, w = imL.shape

        gate_values = []

        v2, v, gate_values_feat = self.feature(imL)
        x2, y2 = v2.split(dim=0, split_size=b//2)
        gate_values.extend(gate_values_feat)

        v_up, gate_values_up = self.up(featL=v)
        x, y = [], []
        for v_ in v_up:
            x_, y_ = v_.split(dim=0, split_size=b//2)
            x.append(x_)
            y.append(y_)
        gate_values.extend(gate_values_up)


        stem_2v = self.stem_2(imL)
        stem_4v = self.stem_4(stem_2v)
        stem_2x, stem_2y = stem_2v.split(dim=0, split_size=b//2)
        stem_4x, stem_4y = stem_4v.split(dim=0, split_size=b//2)

        x[0] = torch.cat((x[0], stem_4x), 1)
        y[0] = torch.cat((y[0], stem_4y), 1)

        # Cost volume processing

        if self.corr_volume:
            cost = (self.cost_volume(x[0], y[0]))[:, :, :-1]
        else:
            refimg_fea = self.cost_conv(x[0])
            targetimg_fea = self.cost_conv(y[0])
            refimg_fea = self.cost_desc(refimg_fea)
            targetimg_fea = self.cost_desc(targetimg_fea)

            cost = Variable(
                torch.FloatTensor(
                    refimg_fea.size()[0],
                    refimg_fea.size()[1]*2,
                    self.D, 
                    refimg_fea.size()[2], 
                    refimg_fea.size()[3]).zero_()).cuda()
            for i in range(self.D):
                if i > 0:
                    cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                    cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
                else:
                    cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                    cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
            cost = cost.contiguous()

        cost = self.cost_agg(x, cost)

        # spixel guide comp
        xspx = self.spx_4(x[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        # Regression
        disp_pred = self.regression(cost, spx_pred, training=training)
        # if training:
        #     disp_pred.append(0)

        return disp_pred, gate_values
    
    def upsample_disp_to_gt(self, preds, _pad):
        pred_disps, _ = preds
        pred_disps = [F.interpolate(pred_disps[i].unsqueeze(1), scale_factor=2**(i*2)) for i in range(len(pred_disps))]
        pred_disp =pred_disps[0]
        ht, wd = pred_disp.shape[-2:]
        c = [_pad[2], ht-_pad[3], _pad[0], wd-_pad[1]]
        pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]
        pred_disps = [pred_disps[i][..., c[0]:c[1], c[2]:c[3]] for i in range(len(pred_disps))]

        preds = (pred_disps,) + preds[1:]
        return pred_disp, preds, c
    
    def compute_loss(self, image2, image3, predictions, gt, validgt, adapt_mode='full', loss_cfg=None, _pad=None, **kwargs):
        pred_disps, gates  = predictions

        validgt = validgt.to('cuda')
        gt = gt.to('cuda')

        loss_dict={}

        if len(validgt.shape) != 4:
            validgt = validgt.unsqueeze(1)

        proxy_aug_weight = kwargs.get('proxy_aug_weight', 0.)
        proxy_orig = kwargs['data'].get('proxy.orig.png', None)

        if proxy_orig is None:
            losses = [F.smooth_l1_loss(pred_disp_[validgt>0], gt[validgt>0], reduction='mean') for pred_disp_ in pred_disps]
            loss = sum([losses[i] * self.train_weights[i] for i in range(len(pred_disps))]) / \
            sum([1 * self.train_weights[i] for i in range(len(pred_disps))])
            # loss = sum([losses[i] * self.train_weights[i] for i in range(len(pred_disps))])
            loss_dict['sgm_loss'] = loss

        else:
            losses = [F.smooth_l1_loss(pred_disp_, gt, reduction='none') for pred_disp_ in pred_disps]

            validgt = (proxy_orig > 0).float()
            invalidgt = (proxy_orig == 0).float()

            loss = sum([losses[i][validgt > 0].mean() * self.train_weights[i] for i in range(len(pred_disps))]) / \
                sum([1 * self.train_weights[i] for i in range(len(pred_disps))])
            invalid_loss = sum([losses[i][invalidgt > 0].mean() * self.train_weights[i] for i in range(len(pred_disps))]) / \
                sum([1 * self.train_weights[i] for i in range(len(pred_disps))])

            loss = loss + proxy_aug_weight * invalid_loss
            loss_dict['invalid_loss'] = proxy_aug_weight * invalid_loss
            loss_dict['sgm_loss'] = loss

        loss_dict['train_loss'] = loss

        return loss, loss_dict
