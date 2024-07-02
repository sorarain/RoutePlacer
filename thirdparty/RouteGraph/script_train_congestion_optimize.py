import os
import argparse

from train.train_congestion_optimized import train_congestion


argparser = argparse.ArgumentParser("Training")

argparser.add_argument('--name', type=str, default='congestion')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--train_epoch', type=int, default=100)
argparser.add_argument('--eval_every_n_epoch', type=int, default=5)
argparser.add_argument('--batch_size', type=int, default=5)
argparser.add_argument('--lr', type=float, default=3e-4)
argparser.add_argument('--min_lr', type=float, default=1e-6)
argparser.add_argument('--weight_decay', type=float, default=1e-5)
# argparser.add_argument('--lr_decay', type=float, default=2e-2)
argparser.add_argument('--beta', type=float, default=0.5)

argparser.add_argument('--app_name', type=str, default='')
argparser.add_argument('--win_x', type=float, default=32)
argparser.add_argument('--win_y', type=float, default=40)
argparser.add_argument('--win_cap', type=int, default=5)

argparser.add_argument('--model', type=str, default='')  # ''
argparser.add_argument('--trans', type=bool, default=False)  # ''
argparser.add_argument('--layers', type=int, default=3)  # 3
argparser.add_argument('--node_feats', type=int, default=64)  # 64
argparser.add_argument('--net_feats', type=int, default=128)  # 128
argparser.add_argument('--pin_feats', type=int, default=16)  # 16
argparser.add_argument('--hanna_feats', type=int, default=4)  # 4
argparser.add_argument('--edge_feats', type=int, default=4)  # 4
argparser.add_argument('--topo_geom', type=str, default='default')  # default
argparser.add_argument('--recurrent', type=bool, default=False)  # False
argparser.add_argument('--topo_conv_type', type=str, default='CFCNN')  # CFCNN
argparser.add_argument('--grid_conv_type', type=str, default='SAGE')  # CFCNN
argparser.add_argument('--agg_type', type=str, default='max')  # max
argparser.add_argument('--cat_raw', type=bool, default=True)  # True
argparser.add_argument('--add_pos', type=bool, default=False)  # True

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='100000')
argparser.add_argument('--idx', type=int, default=8)
argparser.add_argument('--itermax', type=int, default=2500)
# argparser.add_argument('--scalefac', type=float, default=7.0)
# argparser.add_argument('--outtype', type=str, default='tanh')
argparser.add_argument('--binx', type=int, default=32)
argparser.add_argument('--biny', type=int, default=40)

argparser.add_argument('--graph_scale', type=int, default=10000)
args = argparser.parse_args()
NETLISTS_DIR=f'./data/collect'
MODEL_DIR="./param"
LOG_DIR=f"./log/{args.name}-{args.test}"
FIG_DIR=f'./log/{args.name}-{args.test}_temp'

train_netlists_names=[
    # 'superblue1',
    # 'superblue2',
    # 'superblue3',
    # 'superblue6',
    # 'superblue7',
    # 'superblue9',
    # 'superblue11',
    # 'superblue12',
    'superblue14',
    'superblue16',
    'superblue19',
    ]
validation_netlists_names=[
                        # 'superblue14',
                        # 'superblue19',
                        # 'superblue7',
                        'superblue12',
                        ]
test_netlists_names=[
                    # 'superblue19',
                    # 'superblue7',
                    ]

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
if not os.path.isdir(FIG_DIR):
    os.mkdir(FIG_DIR)

train_congestion(args,
                netlists_dir=NETLISTS_DIR,
                train_netlists_names=train_netlists_names,
                validation_netlists_names=validation_netlists_names,
                test_netlists_names=test_netlists_names,
                log_dir=LOG_DIR,
                fig_dir=FIG_DIR,
                model_dir=MODEL_DIR
                )