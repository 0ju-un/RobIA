import numpy as np
import torch

import os
import math
import cv2
import tarfile
import random

import webdataset as wds
from PIL import Image
import io
from itertools import islice
import more_itertools


def gt_decoder(value):
    return np.expand_dims(cv2.imdecode(np.fromstring(value, np.uint8), -1), -1).transpose(2, 0, 1).astype(np.float32) / 256.

def proxy_decoder(value):
    return np.expand_dims(cv2.imdecode(np.fromstring(value, np.uint8), -1), -1).transpose(2, 0, 1).astype(np.float32) 

def jpg_decoder(value):
    return np.array(Image.open(io.BytesIO(value))).transpose(2, 0, 1).astype(np.float32) / 255.
    return v


drivingstereo_lut = {
    'rainy': [
        '2018-08-17-09-45'], 
    'dusky': [
        '2018-10-11-17-08'],
    'cloudy': [
        '2018-10-15-11-43'], 
    'rainy2': [
        '2018-10-17-14-35'],
    'rainy3': [
        '2018-10-22-10-44'],
    'rainy4': [
        '2018-10-25-07-37'],
    'dusky2': [
        '2018-10-16-07-40',
        '2018-10-16-11-13',
        '2018-10-16-11-43',
        '2018-10-24-11-01'],
    'cloudy2': [
        '2018-10-17-14-35',
        '2018-10-17-15-38',
        '2018-10-18-10-39',
        '2018-10-18-15-04',
        '2018-10-19-10-33'],
}

def fetch_single_dataloader(dataset_str, datapath, logger=None):
    subs = -1 # -1 means all sub-sequences from the domain are sampled
    if 'dsec' in dataset_str:
        proxy16=True
    else:
        proxy16=False

    datapath = os.path.join(datapath,dataset_str.split('[')[0])

    dataset_lut = dataset_str.replace('[','_lut[')
    lut = eval(dataset_lut)

    samples = 0
    if subs == -1:
        sequences = [ '%s/%s.tar'%(datapath,s) for s in lut]
    else:
        sequences = [ '%s/%s.tar'%(datapath,s) for s in random.sample(lut,subs)]

    dataset = wds.WebDataset(sequences[0]).decode(
        wds.handle_extension("image_02.jpg", jpg_decoder),
        wds.handle_extension("image_03.jpg", jpg_decoder),
        wds.handle_extension("groundtruth.png", gt_decoder),
        wds.handle_extension("proxy.png", proxy_decoder) if not proxy16 else wds.handle_extension("proxy.png", gt_decoder),
        wds.imagehandler("torchrgb"))

    with tarfile.open(sequences[0]) as archive:
        samples += (sum(1 for name in archive.getnames() if 'image_02' in name))
        a=6

    for s in sequences[1:]:
        _d = wds.WebDataset(s).decode(
            wds.handle_extension("image_02.jpg", jpg_decoder),
            wds.handle_extension("image_03.jpg", jpg_decoder),
            wds.handle_extension("groundtruth.png", gt_decoder),
            wds.handle_extension("proxy.png", proxy_decoder) if not proxy16 else wds.handle_extension("proxy.png", gt_decoder),
            wds.imagehandler("torchrgb")
        )
        dataset += _d

        with tarfile.open(s) as archive:
            samples += (sum(1 for name in archive.getnames() if 'image_02' in name))

    dataset = dataset.slice(500)
    samples = 500

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, persistent_workers=False,
                    pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
    loader.__len__= int(samples)
    loader.dataset_str = dataset_str
    domain = dataset_str.split("'")[-2]
    loader.domain = domain

    return loader

