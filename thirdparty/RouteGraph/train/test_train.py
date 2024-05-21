import argparse

from train import train_congestion

argparser = argparse.ArgumentParser("Training")

argparser.add_argument('--name', type=str, default='main')
argparser.add_argument('--epochs', type=int, default=3)
argparser.add_argument('--train_epoch', type=int, default=5)
argparser.add_argument('--batch', type=int, default=1)
argparser.add_argument('--lr', type=float, default=2e-4)
argparser.add_argument('--weight_decay', type=float, default=2e-4)
argparser.add_argument('--lr_decay', type=float, default=2e-2)
argparser.add_argument('--beta', type=float, default=0.5)

argparser.add_argument('--app_name', type=str, default='')
argparser.add_argument('--win_x', type=float, default=32)
argparser.add_argument('--win_y', type=float, default=40)
argparser.add_argument('--win_cap', type=int, default=5)

argparser.add_argument('--model', type=str, default='')  # ''
argparser.add_argument('--trans', type=bool, default=False)  # ''
argparser.add_argument('--layers', type=int, default=2)  # 2
argparser.add_argument('--node_feats', type=int, default=32)  # 64
argparser.add_argument('--net_feats', type=int, default=32)  # 128
argparser.add_argument('--pin_feats', type=int, default=3)  # 16
argparser.add_argument('--edge_feats', type=int, default=4)
argparser.add_argument('--hanna_feats', type=int, default=32)  # 4
argparser.add_argument('--topo_geom', type=str, default='both')  # default
argparser.add_argument('--add_pos', type=bool, default=False)  # False
argparser.add_argument('--recurrent', type=bool, default=False)  # False
argparser.add_argument('--topo_conv_type', type=str, default='CFCNN')  # CFCNN
argparser.add_argument('--geom_conv_type', type=str, default='SAGE')  # SAGE
argparser.add_argument('--agg_type', type=str, default='max')  # max
argparser.add_argument('--cat_raw', type=bool, default=True)  # True
argparser.add_argument('--pos_code', type=float, default=0.0)  # 0.0

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='100000')
argparser.add_argument('--idx', type=int, default=8)
argparser.add_argument('--itermax', type=int, default=2500)
argparser.add_argument('--scalefac', type=float, default=7.0)
argparser.add_argument('--outtype', type=str, default='tanh')
argparser.add_argument('--binx', type=int, default=32)
argparser.add_argument('--biny', type=int, default=40)


args = argparser.parse_args()

train_congestion(args)