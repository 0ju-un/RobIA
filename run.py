import configparser
import random

from clients import StereoClient

import argparse
import torch
import numpy as np

import os
import logging
import datetime
import wandb

WORK_DIR = './work_dirs'

parser = argparse.ArgumentParser(description='FedStereo')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--cfg', type=str, default='cfgs/clients/client.ini')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--visualize', type=bool, default=False)
parser.add_argument('--err10', type=bool, default=False)
parser.add_argument('--wandb', action='store_true')

parser.add_argument('--maxdisp', type=int, default=192)
parser.add_argument('--training', type=bool, default=True)

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    clients_file = args.cfg
    clients_ids = 0
    print('Clients:\n%s'%str(clients_file))

    assert os.path.isdir(WORK_DIR)
    train_serial = str(datetime.datetime.now())

    args.train_serial = train_serial
    LOG_DIR = os.path.join(WORK_DIR, train_serial)
    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger(name='')
    root_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(LOG_DIR, 'train.log'))
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    
    client = StereoClient(clients_file, args, clients_ids,
                          logger=root_logger,
                          exp_dir=LOG_DIR,
                          visualize=args.visualize,
                          err10=args.err10)
    client.run()

if __name__ == '__main__':
   main()