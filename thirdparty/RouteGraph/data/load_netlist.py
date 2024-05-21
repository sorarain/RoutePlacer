import os
import os.path as osp
import sys

sys.path.append(os.path.abspath("."))
# from data.graph import load_graph

import numpy as np
import itertools
import dgl
import pickle
import tqdm
import pandas as pd
from queue import Queue
import torch
from torch.nn import functional as F
import time
import json

from data.utils import get_edge,get_pos,get_cell_size,get_node_pin_num,get_h_net_density_grid,get_v_net_density_grid,get_net2hpwl,get_node_congestion_label,get_pin_density,get_node_density_vectorized
from data.utils import feature_grid2node_vectorized,transform_graph2edges,fo_average

def node_pairs_among(nodes, max_cap=-1):
    us = []
    vs = []
    if max_cap == -1 or len(nodes) <= max_cap:
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                us.append(u)
                vs.append(v)
    else:
        for u in nodes:
            vs_ = np.random.permutation(nodes)
            left = max_cap - 1
            for v_ in vs_:
                if left == 0:
                    break
                if u == v_:
                    continue
                us.append(u)
                vs.append(v_)
                left -= 1
    return us, vs

def build_near_graph(part_hetero_graph,sub_node_pos,
                            h_net_density_grid,
                            v_net_density_grid,
                            # pin_density_grid,
                            # node_density_grid,
                            bin_x,bin_y):
    sub_node_pos = sub_node_pos.numpy()
    cell_size = part_hetero_graph.nodes['cell'].data['hv'][:,:2].clone().detach()
    sub_node_pos_ = sub_node_pos.copy()
    sub_node_pos = sub_node_pos[np.logical_not(np.isinf(1.0/(sub_node_pos[:,0]+sub_node_pos[:,1])))]
    cell_size = cell_size.numpy()
    cell_size = cell_size[np.logical_not(np.isinf(1.0/(sub_node_pos_[:,0]+sub_node_pos_[:,1]))),:]
    # sub_node_pos = sub_node_pos[np.logical_not(np.isinf(1.0/(cell_size[:,0]*cell_size[:,1])))]
    xl,yl,xh,yh = np.min(sub_node_pos[:,0]),np.min(sub_node_pos[:,1]),np.max(sub_node_pos[:,0]),np.max(sub_node_pos[:,1])
    num_node = part_hetero_graph.num_nodes(ntype='cell')
    num_net = part_hetero_graph.num_nodes(ntype='net')
    num_bin_x,num_bin_y = int(np.ceil((xh - xl) / bin_x)),int(np.ceil((yh - yl) / bin_y))

    def distance_among(a: int, b: int) -> float:
        return ((sub_node_pos[a,0] + cell_size[a,0] * 0.5 - sub_node_pos[b,0] - cell_size[b,0] * 0.5) ** 2
                + (sub_node_pos[a,1] + cell_size[a,1] * 0.5 - sub_node_pos[b,1] - cell_size[b,1] * 0.5) ** 2) ** 0.5
    us4, vs4 = [], []
    use_tqdm = True
    win_x = 32
    win_y = 40
    for off_idx, (x_offset, y_offset) in enumerate(
            [(0, 0), (win_x / 2, 0), (0, win_y / 2), (win_x / 2, win_y / 2)]):
        partition = range(len(sub_node_pos))
        box_node = {}
        for i, sx, sy, px, py in \
                zip(partition, cell_size[:,0], cell_size[:,1], sub_node_pos[:,0], sub_node_pos[:,1]):
            if px == 0 and py == 0:
                continue
            px += x_offset
            py += y_offset
            x_1, x_2 = int(px / win_x), int((px + sx) / win_x)
            y_1, y_2 = int(py / win_y), int((py + sy) / win_y)
            for x in range(x_1, x_2 + 1):
                for y in range(y_1, y_2 + 1):
                    box_node.setdefault(f'{x}-{y}', []).append(i)
        us, vs = [], []
        for nodes in box_node.values():
            us_, vs_ = node_pairs_among(nodes, max_cap=20)
            us.extend(us_)
            vs.extend(vs_)
        us4.extend(us)
        vs4.extend(vs)
    # iter_uv4 = tqdm.tqdm(zip(us4, vs4), total=len(us4)) if use_tqdm else zip(us4, vs4)
    # dis4 = [[distance_among(u, v) / 24] for u, v in iter_uv4]
    dis4 = torch.zeros([len(us4),1])

    us,vs = part_hetero_graph.edges(etype='pins')
    grid_graph = dgl.heterograph({
        ('cell', 'pins', 'net'): (us, vs),
        ('net', 'pinned', 'cell'): (vs, us),
        ('cell','near','cell'): (us4, vs4)
    },num_nodes_dict={'cell':part_hetero_graph.num_nodes(ntype='cell'),'net':part_hetero_graph.num_nodes(ntype='net')})
    
    grid_graph.nodes['cell'].data['hv'] = part_hetero_graph.nodes['cell'].data['hv']
    grid_graph.nodes['cell'].data['pos'] = part_hetero_graph.nodes['cell'].data['pos']
    grid_graph.nodes['net'].data['hv'] = part_hetero_graph.nodes['net'].data['hv']
    grid_graph.nodes['net'].data['degree'] = part_hetero_graph.nodes['net'].data['degree']

    grid_graph.edges['pinned'].data['feats'] = part_hetero_graph.edges['pinned'].data['feats']
    grid_graph.edges['near'].data['feats'] = torch.tensor(dis4,dtype=torch.float32)

    return grid_graph