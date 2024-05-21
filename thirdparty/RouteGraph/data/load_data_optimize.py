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


def load_graph(netlist_dir):
    netlist_process_dir = os.path.join(netlist_dir,"common")

    edges = get_edge(netlist_process_dir)

    node_pin_num = np.load(os.path.join(netlist_process_dir,"pdata.npy"))
    cell_size_x = np.load(os.path.join(netlist_process_dir,"sizdata_x.npy"))
    cell_size_y = np.load(os.path.join(netlist_process_dir,"sizdata_y.npy"))

    edge_iter = tqdm.tqdm(edges.items(),total=len(edges),leave=False)
    us = []
    vs = []
    pins_feats = []
    net_degree = []
    for net,list_node_feats in edge_iter:
        net_degree.append(len(list_node_feats))
        xs,ys = [], []
        pxs,pys = [], []
        for node, pin_px, pin_py, pin_io in list_node_feats:
            us.append(node)
            vs.append(net)
            pins_feats.append([pin_px, pin_py, pin_io])
    pins_feats = torch.tensor(pins_feats,dtype=torch.float32)
    hetero_graph = dgl.heterograph({
        ('cell','pins','net'):(us,vs),
        ('net','pinned','cell'):(vs,us),
    },num_nodes_dict={'cell':len(node_pin_num),'net':len(edges)})
    hetero_graph.nodes['cell'].data['pos'] = torch.zeros([len(node_pin_num),2],dtype=torch.float32)
    hetero_graph.nodes['cell'].data['hv'] = torch.zeros([len(node_pin_num),5],dtype=torch.float32)
    hetero_graph.nodes['cell'].data['label'] = torch.zeros([len(node_pin_num)],dtype=torch.float32)
    hetero_graph.nodes['net'].data['hv'] = torch.zeros([len(edges),7],dtype=torch.float32)
    hetero_graph.nodes['net'].data['degree'] = net_degree = torch.unsqueeze(torch.tensor(net_degree,dtype=torch.float32),dim=-1)

    hetero_graph.edges['pinned'].data['feats'] = pins_feats
    hetero_graph.nodes['cell'].data['node_pin_num'] = torch.tensor(node_pin_num,dtype=torch.float32)
    hetero_graph.nodes['cell'].data['cell_size_x'] = torch.tensor(cell_size_x,dtype=torch.float32)
    hetero_graph.nodes['cell'].data['cell_size_y'] = torch.tensor(cell_size_y,dtype=torch.float32)
    us,vs = [],[]
    for net, list_node_feats in edges.items():
        nodes = [node_feats[0] for node_feats in list_node_feats]
        us_, vs_ = node_pairs_among(nodes, max_cap=8)
        us.extend(us_)
        vs.extend(vs_)
    homo_graph = dgl.add_self_loop(dgl.graph((us, vs), num_nodes=len(node_pin_num)))
    p_gs = dgl.metis_partition(homo_graph,int(np.ceil(len(node_pin_num)/10000)))
    partition_list = []
    for k,val in p_gs.items():
        nids = val.ndata[dgl.NID].numpy().tolist()
        partition_list.append(nids)
    

    list_hetero_graph = []
    list_route_graph = []
    iter_partition_list = tqdm.tqdm(enumerate(partition_list), total=len(partition_list))
    all_net_degree_dict = {}
    node_belong_partition = np.ones(hetero_graph.num_nodes(ntype='cell'))
    for i,partition in enumerate(partition_list):
        for node in list(set(partition)):
            node_belong_partition[node] = i
    for net_id, node_id in zip(*[ns.tolist() for ns in hetero_graph.edges(etype='pinned')]):
        belong = node_belong_partition[node_id]
        all_net_degree_dict.setdefault(belong,{}).setdefault(net_id,0)
        all_net_degree_dict[belong][net_id] += 1
    
    for i,partition in iter_partition_list:
        partition_set = set(partition)
        new_net_degree_dict = all_net_degree_dict[i]
        keep_nets_id = np.array(list(new_net_degree_dict.keys()))
        keep_nets_degree = np.array(list(new_net_degree_dict.values()))
        part_hetero_graph = dgl.node_subgraph(hetero_graph, nodes={'cell': partition, 'net': keep_nets_id})
        list_hetero_graph.append(part_hetero_graph)
    del homo_graph
    return hetero_graph,list_hetero_graph

def load_data(netlist_dir:str,args,save_type:int=1):
    print(f"load {netlist_dir.split('/')[-1]}:")
    if save_type == 1 and os.path.exists(osp.join(netlist_dir,f'{args.app_name}graph_grid.pickle')):
        with open(osp.join(netlist_dir,f'{args.app_name}graph_grid.pickle'),"rb") as f:
            tuple_graph = pickle.load(f)
            hetero_graph,list_hetero_graph = tuple_graph
    else:
        hetero_graph,list_hetero_graph = load_graph(netlist_dir)
        tuple_graph = (hetero_graph,list_hetero_graph)
        with open(osp.join(netlist_dir,f'{args.app_name}graph_grid.pickle'),"wb") as f:
            pickle.dump(tuple_graph, f)
    
    if save_type == 1 and os.path.exists(osp.join(netlist_dir,f'{args.app_name}batch_input.pickle')):
        with open(osp.join(netlist_dir,f'{args.app_name}batch_input.pickle'),"rb") as f:
            batch_input = pickle.load(f)
    else:
        netlist_process_dir = os.path.join(netlist_dir,"common")
        edges = get_edge(netlist_process_dir)
        cnt = 0

        batch_input = []
        for i in range(2000):
            if not os.path.exists(os.path.join(netlist_dir,str(i))):
                continue
            input_dir = os.path.join(netlist_dir,str(i))
            input_process_dir = f"{input_dir}_processed"
            input_dict = {}
            node_pos = get_pos(input_process_dir)

            h_net_density_grid = get_h_net_density_grid(input_dir)
            v_net_density_grid = get_v_net_density_grid(input_dir)

            raw_labels = get_node_congestion_label(input_process_dir)
            labels = np.zeros(node_pos.shape[0],dtype=np.float32)
            index = raw_labels.shape[0]
            labels[:index] = raw_labels
            labels = torch.tensor(labels,dtype=torch.float32)

            # bin_x,bin_y = 32,40
            # pin_density_grid = get_pin_density(h_net_density_grid, 32, 40, node_pos,edges)
            with open(os.path.join(netlist_process_dir,"route_info.json"),"r") as f:
                route_info = json.load(f)
            bin_x,bin_y = route_info['bin_size_x'],route_info['bin_size_y']
            input_dict['bin_x'],input_dict['bin_y'] = bin_x,bin_y
            node_density_grid = get_node_density_vectorized(h_net_density_grid, bin_x, bin_y, node_pos)

            # h_net_density_grid = weight_process(h_net_density_grid, node_pos, bin_x, bin_y)
            # v_net_density_grid = weight_process(v_net_density_grid, node_pos, bin_x, bin_y)
            node_hv = torch.cat(
                        [
                            torch.tensor(
                                np.stack([hetero_graph.nodes['cell'].data['cell_size_x'],
                                        hetero_graph.nodes['cell'].data['cell_size_y'],
                                        hetero_graph.nodes['cell'].data['node_pin_num']], axis=-1),
                                        dtype=torch.float32
                                        ),
                            torch_feature_grid2node_weighted(
                                    torch.tensor(np.stack([h_net_density_grid, v_net_density_grid], axis=-1), dtype=torch.float32),
                                    bin_x,
                                    bin_y,
                                    torch.tensor(node_pos, dtype=torch.float32)
                                )
                        ],
            # feature_grid2node(pin_density_grid, 32, 40, node_pos),
            # feature_grid2node(node_density_grid, 32, 40, node_pos),
                        dim=-1
                    )

            edge_iter = tqdm.tqdm(edges.items(),total=len(edges),leave=False)
            us = []
            vs = []
            pins_feats = []
            net_span_feat = []
            net_degree = []
            for net,list_node_feats in edge_iter:
                net_degree.append(len(list_node_feats))
                xs,ys = [], []
                pxs,pys = [], []
                for node, pin_px, pin_py, pin_io in list_node_feats:
                    us.append(node)
                    vs.append(net)
                    pins_feats.append([pin_px, pin_py, pin_io])
                    x,y = node_pos[node,:]
                    px = x + pin_px
                    py = y + pin_py
                    xs.append(px)
                    ys.append(py)
                    pxs.append(int(px / bin_x))
                    pys.append(int(py / bin_y))
                min_x,max_x,min_y,max_y = min(xs),max(xs),min(ys),max(ys)
                span_h = max_x - min_x + 1
                span_v = max_y - min_y + 1
                min_px,max_px,min_py,max_py = min(pxs),max(pxs),min(pys),max(pys)
                span_ph = max_px - min_px + 1
                span_pv = max_py - min_py + 1
                net_span_feat.append([span_h ,span_v, span_h * span_v, 
                                    span_ph, span_pv, span_ph * span_pv,len(list_node_feats)])
            net_degree_ = np.array(net_degree,dtype=np.float32)
            net_degree = torch.unsqueeze(torch.tensor(net_degree,dtype=torch.float32),dim=-1)
            net_span_feat = torch.tensor(net_span_feat,dtype=torch.float32)
            input_dict['h_net_density_grid'] = h_net_density_grid
            input_dict['v_net_density_grid'] = v_net_density_grid
            input_dict['node_density_grid'] = feature_grid2node_vectorized(node_density_grid, bin_x, bin_y, node_pos)
            input_dict['hv'] = node_hv
            input_dict['net_hv'] = torch.cat([net_span_feat],dim=-1)
            input_dict['pos'] = torch.tensor(node_pos,dtype=torch.float32)
            input_dict['label'] = labels
            input_dict['bin_x'] = bin_x
            input_dict['bin_y'] = bin_y
            batch_input.append(input_dict)
            # cnt+=1
            # if cnt > 3:
            #     break
        with open(osp.join(netlist_dir,f'{args.app_name}batch_input.pickle'),"wb") as f:
            pickle.dump(batch_input, f)
    return tuple_graph,batch_input

# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)

# def feature_grid2node_weight(grid_feature, bin_x, bin_y, node_pos):
#     node_feature = feature_grid2node(grid_feature, bin_x, bin_y, node_pos)
#     node_weight_feature = np.zeros_like(node_feature, dtype=np.float32)
#     dir = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]

#     cnt = 0
#     for node_feat,pos in zip(node_feature,node_pos):
#         x,y = pos
#         grid_index_x_center, grid_index_y_center = int(x / bin_x), int(y / bin_y)
#         weight_array_list = []
#         feat_array_list = []
#         for d in dir:
#             dx, dy = d
#             grid_index_x = grid_index_x_center + dx
#             grid_index_y = grid_index_y_center + dy
#             if grid_index_x < 0 or grid_index_x >= grid_feature.shape[0]:
#                 continue
#             if grid_index_y < 0 or grid_index_y >= grid_feature.shape[1]:
#                 continue
#             grid_pos_x, grid_pos_y = grid_index_x * bin_x + bin_x * 0.5, grid_index_y * bin_y + bin_y * 0.5
#             weight_pos = 1.0 / (np.sqrt((x - grid_pos_x)**2 + (y - grid_pos_y)**2) + 1e-3)
#             # weight_grid_feature[grid_index_x, grid_index_y] += weight * node_feat
#             weight_array_list.append(weight_pos)
#             feat_array_list.append(grid_feature[grid_index_x, grid_index_y])
#         node_weight_feature[cnt] = np.sum(softmax(np.array(weight_array_list)) * np.array(feat_array_list))
#         cnt+=1
#     return node_weight_feature

def torch_feature_grid2node_weighted(grid_feature: torch.Tensor, bin_x: int, bin_y: int, node_pos: torch.Tensor): # TODO: test gradient flow, if any inplace opt causes issues, use clone()  
    n_bin_x, n_bin_y, n_feats = grid_feature.shape
    n_node, _ = node_pos.shape
    
    padded_grid_feat = F.pad(grid_feature, (0,0,1,1,1,1), value=0) # zero padding the grid_feature
    grid_index_x_center, grid_index_y_center = (node_pos[:, 0] // bin_x).long(), (node_pos[:, 1] // bin_y).long() # (n_node,), plus one due to zero-padding
    
    grid_index_x = torch.stack([grid_index_x_center + shift for shift in [-1, 0, 1]], dim=1).repeat_interleave(repeats=3, dim=1) # (n_nodes, 9), long tensor 
    grid_index_y = torch.stack([grid_index_y_center + shift for shift in [-1, 0, 1]], dim=1).repeat(1, 3) # (n_nodes, 9), long tensor
    expanded_grid_feat = padded_grid_feat[grid_index_x.view(-1), grid_index_y.view(-1)].reshape(n_node, 9, n_feats) # (n_nodes, 9, n_feats)
    
    grid_pos_x = torch.stack([(grid_index_x_center + shift - 0.5) * bin_x for shift in [-1, 0, 1]], dim=1).repeat_interleave(repeats=3, dim=1) # (n_nodes, 9), minus 0.5 due to zero-padding
    grid_pos_y = torch.stack([(grid_index_y_center + shift - 0.5) * bin_y for shift in [-1, 0, 1]], dim=1).repeat(1, 3) # (n_nodes, 9)
    grid_pos = torch.stack([grid_pos_x, grid_pos_y], dim=-1) # (n_node, 9, 2)
    diff = node_pos.unsqueeze(1) - grid_pos # (n_node, 9, 2)
    weights = 1 / (diff.norm(p="fro", dim=-1) + 1e-3) # (n_node, 9)
    
    weights[grid_index_x_center == 1, :3] = -float('Inf')
    weights[grid_index_x_center == n_bin_x, -3:] = -float('Inf')
    weights[grid_index_y_center == 1, 0] = weights[grid_index_y_center == 1, 3] = weights[grid_index_y_center == 1, 6] = -float('Inf')
    weights[grid_index_y_center == n_bin_y, 2] = weights[grid_index_y_center == n_bin_y, 5] = weights[grid_index_y_center == n_bin_y, 8] = -float('Inf')
    
    weights = torch.softmax(weights, dim=-1)
    node_feats = (weights.unsqueeze(-1) * expanded_grid_feat).sum(1) # (n_node, n_feat)
    return node_feats

# def weight_process(grid_feature, node_pos, bin_x, bin_y):
#     node_feature = feature_grid2node(grid_feature, bin_x, bin_y, node_pos)
#     weight_grid_feature = np.zeros_like(grid_feature,dtype=np.float32)
#     weight_array_list = [[[] for _ in range(grid_feature.shape[1])] for _ in range(grid_feature.shape[0])]
#     feat_array_list = [[[] for _ in range(grid_feature.shape[1])] for _ in range(grid_feature.shape[0])]
#     dir = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]

#     for node_feat,pos in zip(node_feature,node_pos):
#         x,y = pos
#         grid_index_x_center, grid_index_y_center = int(x / bin_x), int(y / bin_y)

#         for d in dir:
#             dx, dy = d
#             grid_index_x = grid_index_x_center + dx
#             grid_index_y = grid_index_y_center + dy
#             if grid_index_x < 0 or grid_index_x >= grid_feature.shape[0]:
#                 continue
#             if grid_index_y < 0 or grid_index_y >= grid_feature.shape[1]:
#                 continue
#             grid_pos_x, grid_pos_y = grid_index_x * bin_x + bin_x * 0.5, grid_index_y * bin_y + bin_y * 0.5
#             weight_pos = 1.0 / (np.sqrt((x - grid_pos_x)**2 + (y - grid_pos_y)**2) + 1e-3)
#             # weight_grid_feature[grid_index_x, grid_index_y] += weight * node_feat
#             weight_array_list[grid_index_x][grid_index_y].append(weight_pos)
#             feat_array_list[grid_index_x][grid_index_y].append(node_feat)
#     for i in range(grid_feature.shape[0]):
#         for j in range(grid_feature.shape[1]):
#             if len(weight_array_list[i][j]) == 0:
#                 continue
#             softmax_weight = softmax(np.array(weight_array_list[i][j]))
#             weight_grid_feature[i, j] = np.sum(softmax_weight * np.array(feat_array_list[i][j]))
#             assert not np.isnan(weight_grid_feature[i, j])
#     return weight_grid_feature
            
def build_grid_graph(part_hetero_graph,sub_node_pos,
                            h_net_density_grid,
                            v_net_density_grid,
                            # pin_density_grid,
                            # node_density_grid,
                            bin_x,bin_y):
    sub_node_pos = sub_node_pos.numpy()
    cell_size = part_hetero_graph.nodes['cell'].data['hv'][:,:2].clone().detach()
    sub_node_pos_ = sub_node_pos.copy()
    sub_node_pos = sub_node_pos[np.logical_not(np.isinf(1.0/(sub_node_pos[:,0]+sub_node_pos[:,1])))]
    # sub_node_pos = sub_node_pos[np.logical_not(np.isinf(1.0/(cell_size[:,0]*cell_size[:,1])))]
    xl,yl,xh,yh = np.min(sub_node_pos[:,0]),np.min(sub_node_pos[:,1]),np.max(sub_node_pos[:,0]),np.max(sub_node_pos[:,1])
    num_node = part_hetero_graph.num_nodes(ntype='cell')
    num_net = part_hetero_graph.num_nodes(ntype='net')
    num_bin_x,num_bin_y = int(np.ceil((xh - xl) / bin_x)),int(np.ceil((yh - yl) / bin_y))
    g_point_index = np.arange(0,num_bin_x*num_bin_y).reshape((num_bin_x,num_bin_y))
    grid_us,grid_vs = [],[]
    grid_us = np.concatenate([g_point_index[:-1,:].ravel(),g_point_index[:,:-1].ravel("F"),g_point_index[1:,:].ravel(),g_point_index[:,1:].ravel("F"),g_point_index.ravel()])
    grid_vs = np.concatenate([g_point_index[1:,:].ravel(),g_point_index[:,1:].ravel("F"),g_point_index[:-1,:].ravel(),g_point_index[:,:-1].ravel("F"),g_point_index.ravel()])

    # pin_cell_us = np.arange(num_node)[np.logical_not(np.isinf(1.0/(sub_node_pos_[:,0]+sub_node_pos_[:,1])))]
    pin_cell_us = np.arange(sub_node_pos.shape[0])
    # pin_cell_us = np.arange(num_node)[np.logical_not(np.isinf(1.0/(cell_size[:,0]*cell_size[:,1])))]
    pin_grid_vs = np.floor(sub_node_pos[:,0] / bin_x).clip(0,num_bin_x-1) * num_bin_y + np.floor(sub_node_pos[:,1] / bin_y).clip(0,num_bin_y-1)

    grid_pos = np.vstack([np.floor(g_point_index.reshape(-1) / num_bin_y) * bin_x + bin_x * 0.5 + xl,np.floor(g_point_index.reshape(-1) % num_bin_y) * bin_y + bin_y * 0.5 + yl]).T
    grid_index = np.int32(np.floor(np.vstack([grid_pos[:,0] / bin_x,grid_pos[:,1] / bin_y]).T))
    grid_index[:,0] = grid_index[:,0].clip(0,h_net_density_grid.shape[0] - 1)
    grid_index[:,1] = grid_index[:,1].clip(0,h_net_density_grid.shape[1] - 1)
    # print(xl,yl,xh,yh)
    # print(num_bin_x,num_bin_y)
    # print(np.max(np.floor(g_point_index.reshape(-1) / num_bin_y)))
    # print(np.max(grid_pos[:,0]),np.max(grid_pos[:,1]))
    # print(h_net_density_grid.shape)
    # print(h_net_density_grid[grid_index[:,0],grid_index[:,1]].shape)
    grid_density = np.vstack([h_net_density_grid[grid_index[:,0],grid_index[:,1]],
                            v_net_density_grid[grid_index[:,0],grid_index[:,1]],
                            # pin_density_grid[grid_index[:,0],grid_index[:,1]],
                            # node_density_grid[grid_index[:,0],grid_index[:,1]]
                            ]).T
    grid_feat = np.hstack([
        grid_density,
        grid_pos
    ])

    us,vs = part_hetero_graph.edges(etype='pins')
    grid_graph = dgl.heterograph({
        ('cell', 'pins', 'net'): (us, vs),
        ('net', 'pinned', 'cell'): (vs, us),
        ('cell', 'point-to', 'gcell'):(pin_cell_us, pin_grid_vs),
        ('gcell', 'point-from', 'cell'):(pin_grid_vs, pin_cell_us),
        ('gcell','connect','gcell'):(grid_us,grid_vs),
    },num_nodes_dict={'cell':part_hetero_graph.num_nodes(ntype='cell'),'net':part_hetero_graph.num_nodes(ntype='net'),'gcell':num_bin_x*num_bin_y})
    
    grid_graph.nodes['cell'].data['hv'] = part_hetero_graph.nodes['cell'].data['hv']
    grid_graph.nodes['cell'].data['pos'] = part_hetero_graph.nodes['cell'].data['pos']
    grid_graph.nodes['net'].data['hv'] = part_hetero_graph.nodes['net'].data['hv']
    grid_graph.nodes['net'].data['degree'] = part_hetero_graph.nodes['net'].data['degree']
    grid_graph.nodes['gcell'].data['hv'] = torch.tensor(grid_feat,dtype=torch.float32)

    grid_graph.edges['pinned'].data['feats'] = part_hetero_graph.edges['pinned'].data['feats']

    node_pos_center = np.zeros_like(sub_node_pos,dtype=np.float32)
    cell_size = cell_size.numpy()
    cell_size = cell_size[np.logical_not(np.isinf(1.0/(sub_node_pos_[:,0]+sub_node_pos_[:,1]))),:]
    node_pos_center[:,0] = sub_node_pos[:,0] + cell_size[:,0] / 2
    node_pos_center[:,1] = sub_node_pos[:,1] + cell_size[:,1] / 2
    dis = (np.sum((node_pos_center[np.int32(pin_cell_us).reshape(-1),:] - grid_pos[np.int32(pin_grid_vs).reshape(-1),:])**2,axis=1))**0.5
    grid_pin_feats = np.concatenate([
        dis.reshape((-1,1)) / 20,
        # np.ones((len(pin_cell_us),1)) * bin_x,
        # np.ones((len(pin_cell_us),1)) * bin_y,
    ],axis=1)
    grid_graph.edges['point-from'].data['feats'] = torch.tensor(grid_pin_feats,dtype=torch.float32)
    
    return grid_graph

if __name__ == '__main__':
    load_data("/root/autodl-tmp/data/superblue19",2)
    # load_grid_data('/root/autodl-tmp/data/superblue19',2)
