import json
import os
import argparse
from time import time
from typing import List, Dict, Any
from functools import reduce

import numpy as np
import dgl

import torch
import torch.nn as nn

from data.load_data import load_data
from model.RouteGNN import NetlistGNN
# from log.store_scatter import store_scatter
# from utils.output import printout_xf1
from train.train_wirelength import train_wirelength

import warnings

argparser = argparse.ArgumentParser("Training")

argparser.add_argument('--name', type=str, default='wirelength')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=20)
argparser.add_argument('--train_epoch', type=int, default=5)
argparser.add_argument('--batch', type=int, default=1)
argparser.add_argument('--lr', type=float, default=2e-4)
argparser.add_argument('--weight_decay', type=float, default=1e-5)
argparser.add_argument('--lr_decay', type=float, default=2e-2)
argparser.add_argument('--beta', type=float, default=0.5)

argparser.add_argument('--win_x', type=float, default=32)
argparser.add_argument('--win_y', type=float, default=40)

argparser.add_argument('--model', type=str, default='')  # ''
argparser.add_argument('--trans', type=bool, default=False)  # ''
argparser.add_argument('--layers', type=int, default=3)  # 3
argparser.add_argument('--node_feats', type=int, default=64)  # 64
argparser.add_argument('--net_feats', type=int, default=128)  # 128
argparser.add_argument('--pin_feats', type=int, default=16)  # 16
argparser.add_argument('--hanna_feats', type=int, default=4)  # 4
argparser.add_argument('--topo_geom', type=str, default='both')  # default
argparser.add_argument('--topo_conv_type', type=str, default='CFCNN')  # CFCNN
argparser.add_argument('--agg_type', type=str, default='max')  # max
argparser.add_argument('--cat_raw', type=bool, default=True)  # True
argparser.add_argument('--add_pos', type=bool, default=True)  # True

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--scalefac', type=float, default=7.0)
argparser.add_argument('--outtype', type=str, default='tanh')
argparser.add_argument('--binx', type=int, default=32)
argparser.add_argument('--biny', type=int, default=40)

argparser.add_argument('--graph_scale', type=int, default=10000)
args = argparser.parse_args()
NETLISTS_DIR=f'{os.path.abspath(".")}/data/data'
MODEL_DIR="./param"
LOG_DIR=f"./log/{args.name}-{args.test}"
FIG_DIR=f'./log/{args.name}-{args.test}_temp'

train_netlists_names=[
    'superblue2',
    'superblue3',
    'superblue6',
    'superblue7',
    'superblue9',
    'superblue11',
    'superblue14',
    # 'superblue19',
    ]
validation_netlists_names=[
                        'superblue16',
                        # 'superblue19',
                        ]
test_netlists_names=[
                    'superblue19',
                    ]

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
if not os.path.isdir(FIG_DIR):
    os.mkdir(FIG_DIR)

train_wirelength(args,
                netlists_dir=NETLISTS_DIR,
                train_netlists_names=train_netlists_names,
                validation_netlists_names=validation_netlists_names,
                test_netlists_names=test_netlists_names,
                log_dir=LOG_DIR,
                fig_dir=FIG_DIR,
                model_dir=MODEL_DIR
                )