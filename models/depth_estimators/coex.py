from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..coex.feature import Feature, FeatUp
from ..coex.utils import AttentionCostVolume
from ..coex.aggregation import Aggregation
from ..coex.regression import Regression
from ..coex.util_conv import BasicConv, Conv2x

from ..losses import *


import warnings
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class CoEx(nn.Module):
    def __init__(self, cfg):
        super(CoEx, self).__init__()
        self.cfg = cfg
        self.type = self.cfg['backbone']['type']
        chans = self.cfg['backbone']['channels'][self.type]\

        self.D = int(self.cfg['max_disparity']/4)

        self.train_weights= [1, 0.3]

        # set up the feature extraction first
        self.feature = Feature(self.cfg)
        self.up = FeatUp(self.cfg)

        self.corr_volume = cfg['corr_volume']
        if self.corr_volume:
            self.cost_volume = AttentionCostVolume(
                cfg['max_disparity'],
                chans[1]*2+self.cfg['spixel']['branch_channels'][1],
                chans[1]*2,
                1,
                weighted=cfg['matching_weighted'])
            matching_head = cfg['matching_head']
        else:
            self.cost_conv = BasicConv(
                chans[1]*2+self.cfg['spixel']['branch_channels'][1],
                chans[1]*2,
                kernel_size=3,
                padding=1,
                stride=1)
            self.cost_desc = nn.Conv2d(
                chans[1]*2,
                chans[1],
                kernel_size=1,
                padding=0,
                stride=1)
            matching_head = chans[1]*2

        self.cost_agg = Aggregation(
            cfg['backbone'],
            max_disparity=cfg['max_disparity'],
            matching_head=matching_head,
            gce=cfg['gce'],
            disp_strides=cfg['aggregation']['disp_strides'],
            channels=cfg['aggregation']['channels'],
            blocks_num=cfg['aggregation']['blocks_num'],
            spixel_branch_channels=cfg['spixel']['branch_channels'])

        self.regression = Regression(
            max_disparity=cfg['max_disparity'],
            top_k=cfg['regression']['top_k'])

        self.stem_2 = nn.Sequential(
            BasicConv(3, self.cfg['spixel']['branch_channels'][0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.cfg['spixel']['branch_channels'][0], self.cfg['spixel']['branch_channels'][0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.cfg['spixel']['branch_channels'][0]), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(self.cfg['spixel']['branch_channels'][0], self.cfg['spixel']['branch_channels'][1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.cfg['spixel']['branch_channels'][1], self.cfg['spixel']['branch_channels'][1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.cfg['spixel']['branch_channels'][1]), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(chans[1], 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(chans[1]*2+self.cfg['spixel']['branch_channels'][1], chans[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(chans[1], chans[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chans[1]), nn.ReLU()
            )

        FREEZE_BN = True
        if FREEZE_BN:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.eval()
                m.track_running_stats = False
                # m.running_mean = None
                # m.running_var = None

    def forward(self, imL, imR=None, u0=None, v0=None, training=False):
        if imR is not None:
            assert imL.shape == imR.shape
            imL = torch.cat([imL, imR], 0)

        b, c, h, w = imL.shape
        v2, v = self.feature(imL)
        x2, y2 = v2.split(dim=0, split_size=b//2)

        v = self.up(v)
        x, y = [], []
        for v_ in v:
            x_, y_ = v_.split(dim=0, split_size=b//2)
            x.append(x_)
            y.append(y_)

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
            refimg_fea = self.cost_conv(x[0]) #C: 96 -> 48 ; incorporate img_fea & superpixel_fea
            targetimg_fea = self.cost_conv(y[0])
            refimg_fea = self.cost_desc(refimg_fea) #C: 48 -> 24
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

        return disp_pred
    
    def remove_bn_layers(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                parent_name = name.rsplit('.', 1)[0] # get parent module's name
                if parent_name:
                    parent = dict(self.named_modules())[parent_name]
                    setattr(parent, name.split('.')[-1], nn.Identity())
    
    def upsample_disp_to_gt(self, preds, _pad):
        pred_disps = preds
        pred_disps = [F.interpolate(pred_disps[i].unsqueeze(1), scale_factor=2**(i*2)) for i in range(len(pred_disps))]
        pred_disp =pred_disps[0]
        ht, wd = pred_disp.shape[-2:]
        c = [_pad[2], ht-_pad[3], _pad[0], wd-_pad[1]]
        pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]
        pred_disps = [pred_disps[i][..., c[0]:c[1], c[2]:c[3]] for i in range(len(pred_disps))]

        preds = pred_disps
        return pred_disp, preds, c
    
    def compute_loss(self, image2, image3, predictions, gt, validgt, adapt_mode='full', **kwargs):
        if len(validgt.shape) != 4:
            validgt = validgt.unsqueeze(1)
        loss_dict = {}

        if adapt_mode == 'full':
            losses = [self_supervised_loss(pred_, image2, image3) for pred_ in predictions]
            loss = sum(losses).mean()
            loss_dict['self_sup_loss'] = loss

        elif adapt_mode == 'full++':
            proxy_aug_weight = kwargs.get('proxy_aug_weight', 0.)
            proxy_orig = kwargs['data'].get('proxy.orig.png', None)

            if proxy_orig is None:
                losses = [F.smooth_l1_loss(pred_disp_[validgt>0], gt[validgt>0], reduction='mean') for pred_disp_ in predictions]
                loss = sum([losses[i] * self.train_weights[i] for i in range(len(predictions))]) / \
                sum([1 * self.train_weights[i] for i in range(len(predictions))])
                loss_dict['sgm_loss'] = loss

            else:
                losses = [F.smooth_l1_loss(pred_disp_, gt, reduction='none') for pred_disp_ in predictions]

                validgt = (proxy_orig > 0).float()
                invalidgt = (proxy_orig == 0).float()

                loss = sum([losses[i][validgt > 0].mean() * self.train_weights[i] for i in range(len(predictions))]) / \
                    sum([1 * self.train_weights[i] for i in range(len(predictions))])
                invalid_loss = sum([losses[i][invalidgt > 0].mean() * self.train_weights[i] for i in range(len(predictions))]) / \
                    sum([1 * self.train_weights[i] for i in range(len(predictions))])

                loss = loss + proxy_aug_weight * invalid_loss
                loss_dict['invalid_loss'] = proxy_aug_weight * invalid_loss
                loss_dict['sgm_loss'] = loss

        elif adapt_mode == 'full_comb':
            lambda_photo, lambda_sgm = kwargs.get('lambda_val')
            self_sup_loss = sum([self_supervised_loss(pred_, image2, image3) for pred_ in predictions]).mean()
            losses = [F.smooth_l1_loss(pred_disp_[validgt>0], gt[validgt>0], reduction='mean') for pred_disp_ in predictions]
            sgm_loss = sum([losses[i] * self.train_weights[i] for i in range(len(predictions))]) / \
            sum([1 * self.train_weights[i] for i in range(len(predictions))])     

            loss = lambda_photo* self_sup_loss + lambda_sgm * sgm_loss
            loss_dict['sgm_loss'] = sgm_loss
            loss_dict['self_sup_loss'] = self_sup_loss

        loss_dict['train_loss'] = loss        


        return loss, loss_dict

    def aug_test(self, imL, imR, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        disp_map = self.forward(imL['img'][0], imR['img'][0], training=False)
        if 'moe' in self.cfg['backbone']['type']:
            disp_map = disp_map[0]
        if rescale:
            disp_map = resize(
                disp_map[0].unsqueeze(0),
                size=[int(x) for x in imL['img_metas'][0]['ori_shape']][:2],
                mode='bilinear',
                align_corners=True, #self.align_corners,
                warning=False)
        temp = []
        for i in range(1, len(imL['img'])):
            cur_disp_map = self.forward(imL['img'][i], imR['img'][i], training=False)
            if 'moe' in self.cfg['backbone']['type']:
                cur_disp_map = cur_disp_map[0]
            cur_disp_map = resize(
                cur_disp_map[0].unsqueeze(0),
                size=[int(x) for x in imL['img_metas'][i]['ori_shape']][:2],
                mode='bilinear',
                align_corners=True, #self.align_corners,
                warning=False)
            disp_map += cur_disp_map
            temp.append(cur_disp_map)
        disp_map /= len(imL['img'])
        return disp_map

    def aug_test_2(self, imL, imR, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        imL_ = torch.cat(imL['img'])
        imR_ = torch.cat(imR['img'])

        disp_maps = self.forward(imL_, imR_, training=False)
        disp_map = disp_maps[0].mean(dim=0).unsqueeze(0)

        return disp_map

