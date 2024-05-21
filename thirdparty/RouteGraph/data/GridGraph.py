import os
import sys

sys.path.append(os.path.abspath("."))

import numpy as np
import dgl
import pickle
import tqdm
import pandas as pd
import torch
import time
import os.path as osp

from data.partition_graph import partition_graph

from data.utils import get_edge,get_pos,get_cell_size,get_node_pin_num,get_h_net_density_grid,get_v_net_density_grid,get_net2hpwl,get_node_congestion_label,get_pin_density,get_node_density
from data.utils import feature_grid2node,transform_graph2edges,fo_average

def build_grid_graph(part_hetero_graph,sub_node_pos,
                            h_net_density_grid,
                            v_net_density_grid,
                            pin_density_grid,
                            node_density_grid,
                            bin_x,bin_y):
    bin_x,bin_y = 27,35
    sub_node_pos = sub_node_pos.numpy()
    cell_size = part_hetero_graph.nodes['cell'].data['hv'][:,:2]
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

    pin_cell_us = np.arange(num_node)[np.logical_not(np.isinf(1.0/(sub_node_pos_[:,0]+sub_node_pos_[:,1])))]
    # pin_cell_us = np.arange(num_node)[np.logical_not(np.isinf(1.0/(cell_size[:,0]*cell_size[:,1])))]
    pin_grid_vs = np.floor(sub_node_pos[:,0] / bin_x).clip(0,num_bin_x-1) * num_bin_y + np.floor(sub_node_pos[:,1] / bin_y).clip(0,num_bin_y-1)

    grid_pos = np.vstack([np.floor(g_point_index.reshape(-1) / num_bin_y) * bin_x + bin_x * 0.5 + xl,np.floor(g_point_index.reshape(-1) % num_bin_y) * bin_y + bin_y * 0.5 + yl]).T
    grid_index = np.int32(np.floor(np.vstack([grid_pos[:,0] / 32,grid_pos[:,1] / 40]).T))
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
                            pin_density_grid[grid_index[:,0],grid_index[:,1]],
                            node_density_grid[grid_index[:,0],grid_index[:,1]]]).T
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

def load_grid_graph(netlist_dir):
    netlist_process_dir = f"{netlist_dir}_processed"

    edges = get_edge(netlist_process_dir)
    node_pos = get_pos(netlist_process_dir)

    cell_size_x,cell_size_y = get_cell_size(netlist_process_dir)

    node_pin_num = get_node_pin_num(edges, node_pos.shape[0])

    h_net_density_grid = get_h_net_density_grid(netlist_dir)
    v_net_density_grid = get_v_net_density_grid(netlist_dir)

    # net2hpwl = get_net2hpwl(netlist_dir)
    # net2hpwl[net2hpwl < 1e-4] = 1e-4
    # net2hpwl = np.log10(net2hpwl)
    # net2hpwl = torch.tensor(net2hpwl,dtype=torch.float32)

    raw_labels = get_node_congestion_label(netlist_process_dir)
    labels = np.zeros(node_pos.shape[0],dtype=np.float32)
    index = raw_labels.shape[0]
    labels[:index] = raw_labels
    labels = torch.tensor(labels,dtype=torch.float32)



    #bin_x,bin_y 32,40
    bin_x,bin_y = 32,40
    pin_density_grid = get_pin_density(h_net_density_grid, 32, 40, node_pos,edges)
    node_density_grid = get_node_density(h_net_density_grid, 32, 40, node_pos)
    node_hv = torch.tensor(np.vstack((
        cell_size_x,cell_size_y,node_pin_num,
        feature_grid2node(h_net_density_grid, 32, 40, node_pos),
        feature_grid2node(v_net_density_grid, 32, 40, node_pos),
        feature_grid2node(pin_density_grid, 32, 40, node_pos),
        feature_grid2node(node_density_grid, 32, 40, node_pos),
    )),dtype=torch.float32).t()
    
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
    pins_feats = torch.tensor(pins_feats,dtype=torch.float32)
    net_degree_ = np.array(net_degree,dtype=np.float32)
    net_degree = torch.unsqueeze(torch.tensor(net_degree,dtype=torch.float32),dim=-1)
    net_span_feat = torch.tensor(net_span_feat,dtype=torch.float32)
    net_hv = torch.cat([net_degree,net_span_feat],dim=-1)
    node_pos = torch.tensor(node_pos,dtype=torch.float32)
    hetero_graph = dgl.heterograph({
        ('cell','pins','net'):(us,vs),
        ('net','pinned','cell'):(vs,us),
    },num_nodes_dict={'cell':len(node_pos),'net':len(net_degree)})



    hetero_graph.nodes['cell'].data['pos'] = node_pos
    hetero_graph.nodes['cell'].data['hv'] = node_hv
    hetero_graph.nodes['cell'].data['label'] = labels
    hetero_graph.nodes['net'].data['hv'] = net_span_feat
    hetero_graph.nodes['net'].data['degree'] = net_degree
    # hetero_graph.nodes['net'].data['label'] = net2hpwl
    hetero_graph.edges['pinned'].data['feats'] = pins_feats

    ###########
    # partition_list = partition_graph(hetero_graph,netlist_process_dir.split('/')[-1],netlist_process_dir)
    us,vs = [],[]
    for net, list_node_feats in edges.items():
        nodes = [node_feats[0] for node_feats in list_node_feats]
        us_, vs_ = node_pairs_among(nodes, max_cap=8)
        us.extend(us_)
        vs.extend(vs_)
    homo_graph = dgl.add_self_loop(dgl.graph((us, vs), num_nodes=len(node_pos)))
    p_gs = dgl.metis_partition(homo_graph,int(np.ceil(len(node_pos)/10000)))
    partition_list = []
    for k,val in p_gs.items():
        nids = val.ndata[dgl.NID].numpy().tolist()
        partition_list.append(nids)
    # homo_graph.ndata['feat'] = node_hv[:len(node_pos), :]
    # extra = fo_average(homo_graph)
    # hetero_graph.nodes['cell'].data['hv'] = torch.cat([node_hv,extra], dim=1)
    ###########

    list_hetero_graph = []
    list_route_graph = []
    iter_partition_list = tqdm.tqdm(enumerate(partition_list), total=len(partition_list))
    total_route_time = 0
    total_sub_time = 0
    total_init_time = 0
    total_time = 0
    num_grid_point = 0
    num_grid_edges = 0
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
        # total_a = time.time()
        partition_set = set(partition)
        new_net_degree_dict = all_net_degree_dict[i]
        keep_nets_id = np.array(list(new_net_degree_dict.keys()))
        keep_nets_degree = np.array(list(new_net_degree_dict.values()))
        # total_init_time += time.time() - total_a
        # total_b = time.time()
        part_hetero_graph = dgl.node_subgraph(hetero_graph, nodes={'cell': partition, 'net': keep_nets_id})
        # total_sub_time += time.time() - total_b
        list_hetero_graph.append(part_hetero_graph)
        sub_node_pos = part_hetero_graph.nodes['cell'].data['pos']
        # total_b = time.time()
        sub_route_graph = build_grid_graph(part_hetero_graph,sub_node_pos,
                            h_net_density_grid,
                            v_net_density_grid,
                            pin_density_grid,
                            node_density_grid,
                            32,40)
        # total_route_time += time.time() - total_b
        list_route_graph.append(sub_route_graph)
        num_grid_point += sub_route_graph.num_nodes(ntype='gcell')
        num_grid_edges += sub_route_graph.num_edges(etype='connect')
        # total_time += time.time() - total_a
    # print(f"total init time {total_init_time}s average time {total_init_time/len(partition_list)}s")
    # print(f"total sub time {total_sub_time}s average time {total_sub_time/len(partition_list)}s")
    # print(f"total route time {total_route_time}s average time {total_route_time/len(partition_list)}s")
    # print(f"total time {total_time}s average time {total_time/len(partition_list)}s")
    list_tuple_graph = list(zip(list_hetero_graph, list_route_graph))
    print(f"total num grid point {num_grid_point}")
    print(f"total num grid edges {num_grid_edges}")
    # with open(osp.join(netlist_process_dir,"graph.pickle"),"wb+") as f:
    #     pickle.dump(list_tuple_graph, f)
    # return num_hanna_point,num_hanna_edges
    del homo_graph
    return list_hetero_graph,list_route_graph