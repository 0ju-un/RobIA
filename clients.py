import configparser
from contextlib import nullcontext
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as vF
import numpy as np
import time
import cv2

import copy
from copy import deepcopy
import os
import shutil

from models import *
from misc import *
import sys

from dataloaders import tar_datasets as datasets

import tqdm
from ruamel.yaml import YAML
import datetime


import pandas as pd
import csv

NUM_ROUND = 10

# Maps a dataset family name (as written in the config) to its sub-domains,
FAMILY_DOMAINS = {
    'drivingstereo': ['dusky', 'cloudy', 'rainy'],
}


def expand_dataset_families(families):
    expanded = []
    for fam in families:
        fam = fam.strip()
        if fam not in FAMILY_DOMAINS:
            raise ValueError(
                f"Unknown dataset family '{fam}'. "
                f"Known families: {list(FAMILY_DOMAINS.keys())}"
            )
        expanded.extend(f"{fam}['{sub}']" for sub in FAMILY_DOMAINS[fam])
    return expanded


class StereoClient:

    def __init__(self, cfg, args, idx, logger=None, exp_dir=None, err10=False, visualize=True, visualize_interval=100):
        
        config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        config.read(cfg)
        self.config = config
        self.idx = idx
        self.gpu = int(config['adaptation']['gpu'])

        train_serial = f'{config["network"]["model"].replace("-", "_")}-{args.train_serial}'
        self.step = 0

        self.moe_cfg = config['MoE']

        self.args = args
        self.adapt_mode = config['adaptation']['adapt_mode']
        self.use_groundtruth_loss = config['adaptation'].getboolean('use_groundtruth_loss')
        self.aug_flag = config['adaptation'].getboolean('use_sgm_aug')
        self.sample_mode = config['adaptation']['sample_mode']

        self.model = config['network']['model']

        self.runs = []

        self.round = NUM_ROUND
        families = config['environment'].getlist('dataset')
        self.dataset_list = [expand_dataset_families(families)]
        datapath = config['environment']['datapath']
        for dataset in self.dataset_list:
            self.runs.append({'loaders':[datasets.fetch_single_dataloader(domain, datapath,logger=logger) for domain in dataset]})

        Net = build_model(self.model)
        model_cfg = YAML().load(
            open(config['coex']['yaml_path'], 'r')
        )
        model_cfg['MoE'] = config['MoE']
        model_cfg['backbone']['type'] = config['coex']['backbone']

        self.net = Net(model_cfg)

        self.net = nn.DataParallel(self.net)
        self.net.to('cuda:%d'%self.gpu)
        state_dict = torch.load(config['network']['checkpoint'], torch.device('cuda:%d'%self.gpu))

        # coex
        self.net.load_state_dict({k.replace('stereo','module').replace('feature.embedding','embedding'): v for k, v in state_dict['state_dict'].items()}, strict=True)
        self.net = self.net.module

        if self.aug_flag:
            teacher_ckpt =  config['network']['teacher_checkpoint']
            teacher_state_dict = torch.load(teacher_ckpt, torch.device('cuda:%d'%self.gpu))

            teacher_Net = build_model('coex')
            teacher_model_cfg = YAML().load(
                    open(config['coex']['yaml_path'], 'r')
                )
            teacher_model_cfg['backbone']['type'] = 'mobilenetv2_100'
            teacher_model_cfg['MoE'] = config['MoE']

            self.teacher_net = teacher_Net(teacher_model_cfg)
            self.teacher_net = nn.DataParallel(self.teacher_net)
            self.teacher_net.to('cuda:%d'%self.gpu)
            self.teacher_net.load_state_dict({k.replace('stereo','module').replace('feature.embedding','embedding'): v for k, v in teacher_state_dict['state_dict'].items()}, strict=False)
            self.teacher_net = self.teacher_net.module

            for name, param in self.teacher_net.named_parameters():
                param.requires_grad = False
                if 'bn' in name:
                    param.requires_grad = True
            for m in self.teacher_net.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.track_running_stats = False

        _freeze = config['freeze'].getboolean('freeze')
        print(f'feeze backbone:{_freeze}')
        if _freeze:
            _freeze_keys = config['freeze']['keys'].split()
            # select updated params
            for param in self.net.parameters():
                param.requires_grad = False
            for name, param in self.net.named_parameters():
                for _k in _freeze_keys:
                    if _k in name:
                        param.requires_grad = True

        # count_parameters(self.net)
        if logger is not None:
            logger.info(count_parameters(self.net, logger=logger))
        
        self.lr = config['adaptation']['lr']
        if 'lr_moe' in config['adaptation']:
            self.lr_moe = config['adaptation']['lr_moe']
        else:
            self.lr_moe = None
        if 'lr_teacher' in config['adaptation']:
            self.lr_teacher = float(config['adaptation']['lr_teacher'])

        self.accumulator = {}

        self.logger = logger
        self.logger.debug(self.args)
        self.logger.debug(self.runs)

        self.exp_dir = exp_dir
        cfg_path = os.path.join(self.exp_dir, os.path.basename(cfg))
        with open(cfg_path, 'w') as f:
            config.write(f)
        
        self.result_dict = {}


    def run(self):
        for i, run in enumerate(self.runs):
            loaders = run['loaders']
            self.optimizer = optim.Adam(self.net.parameters(), lr=float(self.lr), betas=(0.9, 0.999))
            self.optimizer = self.fetch_optimizer()
            optim_list = self.get_optim_params()

            if self.aug_flag:
                lr_teacher = self.lr_teacher
                optim_list.append({'params': self.teacher_net.parameters(), 'lr': float(lr_teacher), 'betas': (0.9, 0.999)})
                self.optimizer = optim.Adam(optim_list)
            else:
                self.optimizer = optim.Adam(optim_list)
            self.curr_round = 0
            for i in range(self.round):
                print(f'********* Round {i+1} start ***********')
                
                for idx, loader in enumerate(loaders):
                    dataset_str = loader.dataset_str
                    self.train_one_epoch(loader, dataset_str)

                
                self.print_stats('AVG')
                self.accumulator = {}
                self.curr_round = self.curr_round + 1

        result_df = pd.DataFrame(self.result_dict)
        result_df.to_csv(os.path.join(self.exp_dir,'result.csv'), index=False)

        # self.print_gpu_info()
        
    def train_one_epoch(self, loader, dataset_str):
        args=self.args
        domain = loader.domain

        self.pbar = tqdm.tqdm(total=loader.__len__, file=sys.stdout)

        self.net.eval()
        with torch.no_grad() if (self.adapt_mode == 'none') else nullcontext():

            for batch_idx, data in enumerate(loader):

                if self.adapt_mode != 'none':
                    self.optimizer.zero_grad()

                if 'proxy.png' in data:
                    data['validpr'] = (data['proxy.png']>0).float()
                    data['proxy.png'], data['validpr'] = data['proxy.png'].to('cuda:%d'%self.gpu), data['validpr'].to('cuda:%d'%self.gpu)

                data['image_02.jpg'], data['image_03.jpg'] = data['image_02.jpg'].to('cuda:%d'%self.gpu), data['image_03.jpg'].to('cuda:%d'%self.gpu)

                if data['image_02.jpg'].shape[-1] != data['proxy.png'].shape[-1]:
                    data['proxy.png'] = data['proxy.png'][...,:data['image_02.jpg'].shape[-1]]
                    data['validpr'] = data['validpr'][...,:data['image_02.jpg'].shape[-1]]

                # pad images
                ht, wt = data['image_02.jpg'].shape[-2], data['image_02.jpg'].shape[-1]
                pad_ht = (((ht // 128) + 1) * 128 - ht) % 128
                pad_wd = (((wt // 128) + 1) * 128 - wt) % 128
                _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
                data['image_02.jpg'] = F.pad(data['image_02.jpg'], _pad, mode='replicate')
                data['image_03.jpg'] = F.pad(data['image_03.jpg'], _pad, mode='replicate')

                if self.aug_flag:
                    # BN teacher forward
                    data['proxy.orig.png'] = data['proxy.png'].clone()

                    pseudo_disp_maps = self.teacher_net(data['image_02.jpg'], data['image_03.jpg'], training=True)
                    ht, wt = data['proxy.png'].shape[-2], data['proxy.png'].shape[-1]
                    pad_ht = (((ht // 128) + 1) * 128 - ht) % 128
                    pad_wd = (((wt // 128) + 1) * 128 - wt) % 128
                    _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
                    pseudo_disp_map, pseudo_disp_maps, _ = self.teacher_net.upsample_disp_to_gt(pseudo_disp_maps, _pad)

                    # pseudo_disp_map = pseudo_disp_map.detach()

                    pseudo_disp_map = torch.where(data['proxy.png'] == 0, pseudo_disp_map[0], data['proxy.png'])
                    data['proxy.png'] = pseudo_disp_map
                    data['validpr'] = (data['proxy.png'] > 0).float()

                ## prediction
                gates = None
                start_time = time.time()

                pred_disps = self.net(data['image_02.jpg'], data['image_03.jpg'], training=True)

                ## adaptation
                embedding_pred_disps = None
                if self.aug_flag:
                    ht, wt = data['proxy.png'].shape[-2], data['proxy.png'].shape[-1]
                    pad_ht = (((ht // 128) + 1) * 128 - ht) % 128
                    pad_wd = (((wt // 128) + 1) * 128 - wt) % 128
                    _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
                pred_disp, pred_disps, c = self.net.upsample_disp_to_gt(pred_disps, _pad)

                # upsample and remove padding from all predictions (if needed for adaptation)
                if self.adapt_mode != 'none':
                    data['image_02.jpg'] = data['image_02.jpg'][..., c[0]:c[1], c[2]:c[3]]
                    data['image_03.jpg'] = data['image_03.jpg'][..., c[0]:c[1], c[2]:c[3]]

                if self.use_groundtruth_loss:
                    valid_gt = ((data['groundtruth.png'].to('cuda') > 0) & (data['proxy.png'] > 0)).float()
                    loss, loss_log = self.net.compute_loss(data['image_02.jpg'], data['image_03.jpg'],
                                                            pred_disps, data['groundtruth.png'], valid_gt, adapt_mode=self.adapt_mode,
                                                            loss_cfg=self.moe_cfg, _pad=_pad)

                loss, loss_log = self.net.compute_loss(data['image_02.jpg'], data['image_03.jpg'],
                            pred_disps, data['proxy.png'], data['validpr'], adapt_mode=self.adapt_mode,
                            loss_cfg=self.moe_cfg, _pad=_pad,
                            proxy_aug_weight=self.config['adaptation'].getfloat('proxy_aug_weight', 0.0),
                            data=data)
                if self.aug_flag:
                    # BN Teacher loss
                    loss_teacher, _ = self.teacher_net.compute_loss(data['image_02.jpg'], data['image_03.jpg'],
                                                                pseudo_disp_maps, data['proxy.png'], data['validpr'], adapt_mode=self.adapt_mode,
                                                                loss_cfg=self.moe_cfg, _pad=_pad,
                                                                proxy_aug_weight=self.config['adaptation'].getfloat('proxy_aug_weight', 0.0),
                                                                data=data)

                if self.aug_flag:
                    loss.backward(retain_graph=True)
                    loss_teacher.backward()
                else:
                    loss.backward()

                self.optimizer.step()
                pred_disp = pred_disp.detach()
                self.step += 1

                result = {}
                if 'groundtruth.png' in data:
                    data['validgt'] = (data['groundtruth.png'] > 0).float()
                    result = kitti_metrics(pred_disp.cpu().numpy(), data['groundtruth.png'].numpy(), data['validgt'].numpy())

                result['disp'] = pred_disp

                for k in result:
                    if k != 'disp':
                        k_d = f'{domain}_{k}'
                        k_avg = f'AVG_{k}'
                        if k_d not in self.accumulator:
                            self.accumulator[k_d] = []
                        if k_avg not in self.accumulator:
                            self.accumulator[k_avg] = []
                        self.accumulator[k_d].append(result[k])
                        self.accumulator[k_avg].append(result[k])

                self.pbar.set_description("Thread %d, Seq: %s, Frame %s, bad3: %2.2f"%(self.idx, dataset_str, data['__key__'][0], result['bad 3'] if 'bad 3' in result else np.nan))
                self.pbar.update(1)


        if self.pbar is not None:
            self.pbar.close()

        self.print_stats(domain)
        self.save_stats(domain)



    def print_stats(self, domain):
        metrs = ''
        for k in self.accumulator:
            if k in [f'{domain}_bad 3', f'{domain}_epe']:
                metrs += '& %s : %.2f '%(k,np.array(self.accumulator[k]).mean())

        print("\nThread %d results on Seq %s:\\\\ \n%s \\\\"%(self.idx,domain,str(metrs)))
        if self.logger is not None:
            self.logger.info("\nThread %d results on Seq %s:\\\\ \n%s \\\\"%(self.idx,domain,str(metrs)))

    def save_stats(self, domain):
        for k in self.accumulator:
            if k in [f'{domain}_bad 3', f'{domain}_epe']:
                if k in self.result_dict:
                    self.result_dict[k].append(f'{np.array(self.accumulator[k]).mean():.2f}')
                else:
                    self.result_dict[k] = [f'{np.array(self.accumulator[k]).mean():.2f}']
        # print('GPU INFO.....')
        # print(torch.cuda.memory_summary(), end='')
    
    def print_gpu_info(self):
        print('GPU INFO.....')
        gpu_info = torch.cuda.memory_summary()
        print(gpu_info, end='')
        if self.logger is not None:
            self.logger.info(torch.cuda.memory_summary())
    
    def fetch_optimizer(self):
        moe_parameters = []
        other_parameters = []
        if self.lr_moe is not None:
            for n, p in self.net.named_parameters():
                if 'embed' in n or 'gate' in n:
                    moe_parameters.append(p)
                else:
                    other_parameters.append(p)
            if moe_parameters:
                return optim.Adam([{'params': other_parameters, 'lr': float(self.lr), 'betas': (0.9, 0.999)},
                                        {'params': moe_parameters, 'lr': float(self.lr_moe), 'betas': (0.9, 0.999)}])
        else:
            return optim.Adam(self.net.parameters(), lr=float(self.lr), betas=(0.9, 0.999))
    
    def get_optim_params(self):
        moe_parameters = []
        other_parameters = []
        if self.lr_moe is not None:
            for n, p in self.net.named_parameters():
                if 'embed' in n or 'gate' in n:
                    moe_parameters.append(p)
                else:
                    other_parameters.append(p)
            if moe_parameters:
                return [{'params': other_parameters, 'lr': float(self.lr), 'betas': (0.9, 0.999)},
                                        {'params': moe_parameters, 'lr': float(self.lr_moe), 'betas': (0.9, 0.999)}]
        else:
            return [{'params':self.net.parameters(), 'lr':float(self.lr), 'betas':(0.9, 0.999)}]