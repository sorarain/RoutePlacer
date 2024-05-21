import os
import sys

sys.path.append(os.path.abspath("."))

import numpy as np
import itertools
import dgl
import pickle
import tqdm
import pandas as pd
from queue import Queue
import torch
import time
import os.path as osp

from data.partition_graph import partition_graph


def get_w():
    param_dir_list = [
        'test/dac2012/superblue2.json',
        'test/dac2012/superblue3.json',
        'test/dac2012/superblue6.json',
        'test/dac2012/superblue7.json',
        'test/dac2012/superblue9.json',
        'test/dac2012/superblue11.json',
        'test/dac2012/superblue12.json',
        'test/dac2012/superblue14.json',
        'test/dac2012/superblue16.json',
        'test/dac2012/superblue19.json',
    ]
    netlist_name_list = [
        'superblue2',
        'superblue3',
        'superblue6',
        'superblue7',
        'superblue9',
        'superblue11',
        'superblue12',
        'superblue14',
        'superblue16',
        'superblue19',
    ]

    w_list = []
    netlist_info = []

    for param_dir,netlist_name in zip(param_dir_list,netlist_name_list):
        params = Params.Params()
        params.load(param_dir)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        num_pins = len(placedb.pin_offset_x)
        num_hanan_points = 0
        for net2pinid in placedb.net2pin_map:
            num_hanan_points += len(net2pinid) * len(net2pinid)
        print("-----------------")
        print(f"netlist {netlist_name}")
        print(f"num pins {num_pins}")
        print(f"num hanan points {num_hanan_points}")
        print(f"hanan points / pins {num_hanan_points / num_pins}")
        netlist_info.append([netlist_name,num_pins,num_hanan_points,num_hanan_points / num_pins])
        w_list.append(num_hanan_points / num_pins)
    print(w_list)
    print(np.mean(w_list))
    netlist_info = np.array(netlist_info)
    df = pd.DataFrame()
    df['netlist_name'] = netlist_info[:,0]
    df['num pins'] = netlist_info[:,1]
    df['num hanan points'] = netlist_info[:,2]
    df['hanan points / pins'] = netlist_info[:,3]
    df.to_excel("./info.xlsx",index=False)

def get_edge(netlist_dir):
    edge_dict = {
        0:[
            [0,20,20,0],
            [2,30,20,1],
            [3,10,20,1]
        ],
        1:[
            [3,20,20,0],
            [5,20,-20,1],
        ],
        2:[
            [1,20,20,0],
            [3,30,20,1],
            [4,20,20,1],
        ]
    }
    # with open(f"/root/autodl-tmp/{netlist_name}_processed/edge.pkl",'rb') as fp:
    with open(os.path.join(netlist_dir,"edge.pkl"),'rb') as fp:
        edge_dict = pickle.load(fp)
    return edge_dict
def get_pos(netlist_dir):
    node_pos = np.array([
        [100,200],
        [100,400],
        [200,100],
        [200,300],
        [200,500],
        [300,300],
    ])
    # node_pos_x = np.load(f"/root/autodl-tmp/{netlist_name}_processed/xdata_900.npy")
    # node_pos_y = np.load(f"/root/autodl-tmp/{netlist_name}_processed/ydata_900.npy")
    num = 0
    for i in range(1000):
        if os.path.exists(os.path.join(netlist_dir,f"xdata_{i}.npy")):
            num = i
            break
    node_pos_x = np.load(os.path.join(netlist_dir,f"xdata_{num}.npy"))
    node_pos_y = np.load(os.path.join(netlist_dir,f"ydata_{num}.npy"))
    node_pos = np.vstack((node_pos_x,node_pos_y)).T
    return node_pos

def get_cell_size(netlist_dir):
    size_x = np.array([
        10,10,10,10,10,10
    ])
    size_y = np.array([
        10,10,10,10,10,10
    ])
    size_x,size_y = np.load(osp.join(netlist_dir,"sizdata_x.npy")),np.load(osp.join(netlist_dir,"sizdata_y.npy"))
    return size_x,size_y

def get_node_pin_num(edges,num_nodes) -> np.ndarray:
    node_pin_num = np.zeros((num_nodes),dtype=np.float32)
    for _,list_node_feats in tqdm.tqdm(edges.items(),total=len(edges.keys()),leave=False):
        for node,_,_,_ in list_node_feats:
            node_pin_num[node] += 1
    return node_pin_num

def get_h_net_density_grid(netlist_dir) -> np.ndarray:
    num = -1
    for i in range(1000):
        if os.path.exists(f"{netlist_dir}/iter_{i}_bad_cmap_h.npy"):
            num = i
            break
    assert os.path.exists(f"{netlist_dir}/iter_{num}_bad_cmap_h.npy"),f"{netlist_dir}/iter_{num}_bad_cmap_h.npy"
    return np.load(f"{netlist_dir}/iter_{num}_bad_cmap_h.npy")
    # return 0.5 * np.ones((1000,1000))

def get_v_net_density_grid(netlist_dir) -> np.ndarray:
    num = -1
    for i in range(1000):
        if os.path.exists(f"{netlist_dir}/iter_{i}_bad_cmap_v.npy"):
            num = i
            break
    assert os.path.exists(f"{netlist_dir}/iter_{num}_bad_cmap_v.npy")
    return np.load(f"{netlist_dir}/iter_{num}_bad_cmap_v.npy")
    # return 0.5 * np.ones((1000,1000))

def get_net2hpwl(netlist_dir) -> np.ndarray:
    num = -1
    netlist_dir = os.path.join("/",*netlist_dir.split("/")[:-1],"superblue_0425_withHPWL",netlist_dir.split("/")[-1])
    for i in range(1000):
        if os.path.exists(f"{netlist_dir}/iter_{i}_net2hpwl.npy"):
            num = i
            break
    assert os.path.exists(f"{netlist_dir}/iter_{num}_net2hpwl.npy")
    return np.load(f"{netlist_dir}/iter_{num}_net2hpwl.npy")

def get_node_congestion_label(netlist_dir) -> np.ndarray:
    num = -1
    for i in range(1000):
        if os.path.exists(f"{netlist_dir}/iter_{i}_node_label_full_100000_.npy"):
            num = i
            break
    assert os.path.exists(f"{netlist_dir}/iter_{num}_node_label_full_100000_.npy")
    return np.load(f"{netlist_dir}/iter_{num}_node_label_full_100000_.npy")[:,8]

def get_pin_density(density_map,bin_x,bin_y,node_pos,edge) -> np.ndarray:
    pin_density = np.zeros_like(density_map,dtype=np.float32)
    for _,list_node_feats in edge.items():
        for node,pin_offset_x,pin_offset_y,_ in list_node_feats:
            pin_x,pin_y = node_pos[node]
            pin_x += pin_offset_x
            pin_y += pin_offset_y
            index_x,index_y = int(pin_x / bin_x),int(pin_y / bin_y)
            pin_density[index_x,index_y] += 1
    return pin_density

def get_node_density(density_map,bin_x,bin_y,node_pos) -> np.ndarray:
    node_density = np.zeros_like(density_map,dtype=np.float32)
    for x,y in node_pos:
        index_x,index_y = int(x / bin_x),int(y / bin_y)
        node_density[index_x,index_y] += 1
    return node_density

def feature_grid2node(grid_feature: np.ndarray, bin_x, bin_y, node_pos) -> np.ndarray:
    return np.array([
        grid_feature[int (x / bin_x),int(y / bin_y) ] for x,y in node_pos
    ],dtype=np.float32)

def fo_average(g):
    pass

def feature_grid2hanna(grid_feature: np.ndarray, bin_x, bin_y, pos1_x, pos1_y, pos2_x, pos2_y):
    begin_index_x = int(pos1_x / bin_x)
    begin_index_y = int(pos1_y / bin_y)
    end_index_y = int(pos2_y / bin_y)
    return np.mean(grid_feature[begin_index_x,begin_index_y:end_index_y+1])


def intersect(pos_x_1_1,pos_y_1_1,pos_x_2_1,pos_y_2_1,
            pos_x_1_2,pos_y_1_2,pos_x_2_2,pos_y_2_2):
    # pos_x_1_2,pos_y_1_2,pos_x_2_2,pos_y_2_2,_ = line_2_info
    if pos_x_1_1 == pos_x_2_1:
        return not(pos_y_1_1 > pos_y_2_2 or pos_y_2_1 < pos_y_1_2)
    if pos_y_1_1 == pos_y_2_1:
        return not(pos_x_1_1 > pos_x_2_2 or pos_x_2_1 < pos_x_1_2)
    else:
        assert 0==1,"error in intersect"

def build_group_hanna_point(pos_1_dict,hanna_point_info,max_pos_2,min_pos_2,id_num_hanna_points,key,
                            route_edge_us,route_edge_vs,pin_dict,h_net_density_grid,v_net_density_grid,pin_density_grid,node_density_grid,bin_x,bin_y):#key=['x','y']
    pos_1_list = list(set(pos_1_dict.keys()))
    id2pinpos_y = set()
    for pos_1 in pos_1_list:
        for pos_2 in pos_1_dict[pos_1]:
            id2pinpos_y.add(pos_2)
    id2pinpos_y = list(id2pinpos_y)
    id2pinpos_y.sort()
    pinpos_y2id = {}
    for id,pinpos_y in enumerate(id2pinpos_y):
        pinpos_y2id[pinpos_y] = id
    # print("----------------")
    # print(min_pos_2,max_pos_2)
    # print(pin_dict)
    # print(pos_1_dict)
    pos_1_list.sort()
    tmp_pre_hanna_point_list = []
    for i,pos_1 in enumerate(pos_1_list):
        #########create group hanna point###########
        pos_2_list = list(set(pos_1_dict[pos_1].copy()))
        pos_2_list.sort()
        pre_pos_2 = min_pos_2
        tmp_hanna_point_list = []
        if pos_2_list[-1] != max_pos_2:
            pos_2_list.append(max_pos_2)
        for j,pos_2 in enumerate(pos_2_list):
            pre_pos_x,pre_pos_y = pos_1,pre_pos_2
            pos_x,pos_y = pos_1,pos_2
            flag = 0
            if pos_2 == min_pos_2:
                if (pos_x,pos_y) in pin_dict:
                    tmp_hanna_point_list.append([pos_x,pos_y,pos_x,pos_y,pin_dict[(pos_x,pos_y)]])
                continue
            # feature_grid2hanna(h_net_density_grid,bin_x,bin_y,pre_pos_x,pre_pos_y,pos_x,pos_y)
            # feature_grid2hanna(v_net_density_grid,bin_x,bin_y,pre_pos_x,pre_pos_y,pos_x,pos_y)
            # feature_grid2hanna(pin_density_grid,bin_x,bin_y,pre_pos_x,pre_pos_y,pos_x,pos_y)
            # feature_grid2hanna(node_density_grid,bin_x,bin_y,pre_pos_x,pre_pos_y,pos_x,pos_y)
            
            hanna_point_info.append([pre_pos_x,pre_pos_y,pos_x,pos_y,
                                                    feature_grid2hanna(h_net_density_grid,bin_x,bin_y,pre_pos_x,pre_pos_y,pos_x,pos_y),
                                                    feature_grid2hanna(v_net_density_grid,bin_x,bin_y,pre_pos_x,pre_pos_y,pos_x,pos_y),
                                                    feature_grid2hanna(pin_density_grid,bin_x,bin_y,pre_pos_x,pre_pos_y,pos_x,pos_y),
                                                    feature_grid2hanna(node_density_grid,bin_x,bin_y,pre_pos_x,pre_pos_y,pos_x,pos_y),
                                                    ])#flag 0:v  1:h
            #########create group hanna point###########
            #########connect to pin in this line###########
            flag_1,flag_2 = pre_pos_y,pos_y
            pre_pin_pos = (pre_pos_x,pre_pos_y)
            if pre_pin_pos in pin_dict:
                pre_pin = pin_dict[pre_pin_pos]
                route_edge_us.append(pre_pin)
                route_edge_vs.append(id_num_hanna_points)
                route_edge_us.append(id_num_hanna_points)
                route_edge_vs.append(pre_pin)
                flag_1 = id2pinpos_y[pinpos_y2id[pre_pos_y]+1]
            if (pos_x,pos_y) in pin_dict:
                pin = pin_dict[(pos_x,pos_y)]
                route_edge_us.append(pin)
                route_edge_vs.append(id_num_hanna_points)
                route_edge_us.append(id_num_hanna_points)
                route_edge_vs.append(pin)
                flag_2 = id2pinpos_y[pinpos_y2id[pos_y]-1]
            tmp_hanna_point_list.append([pre_pos_x,flag_1,pos_x,flag_2,id_num_hanna_points])
            if (pos_x,pos_y) in pin_dict:
                tmp_hanna_point_list.append([pos_x,pos_y,pos_x,pos_y,pin_dict[(pos_x,pos_y)]])
            # if j == len(pos_2_list) and (not pos_2 == max_pos_2):

            id_num_hanna_points+=1
            pre_pos_2 = pos_2
            #########connect to pin in this line###########
        #########connect to pin in pre line###########
        tmp_id = 0
        tmp_pre_id = 0
        while tmp_pre_id < len(tmp_pre_hanna_point_list) and tmp_id < len(tmp_hanna_point_list):
            pre_pos_x_1,pre_pos_y_1,pos_x_1,pos_y_1,id_1 = tmp_pre_hanna_point_list[tmp_pre_id]
            pre_pos_x_2,pre_pos_y_2,pos_x_2,pos_y_2,id_2 = tmp_hanna_point_list[tmp_id]
            if intersect(pre_pos_x_1,pre_pos_y_1,pos_x_1,pos_y_1,pre_pos_x_2,pre_pos_y_2,pos_x_2,pos_y_2):
                route_edge_us.append(id_1)
                route_edge_vs.append(id_2)
                route_edge_us.append(id_2)
                route_edge_vs.append(id_1)
            if pre_pos_x_1 == pos_x_1:
                if pos_y_1 < pos_y_2:
                    tmp_pre_id+=1
                else:
                    tmp_id+=1
            else:
                if pos_x_1 < pos_x_2:
                    tmp_pre_id+=1
                else:
                    tmp_id+=1
        tmp_pre_hanna_point_list = tmp_hanna_point_list.copy()
                
    # print(hanna_point_info)
    # print(route_edge_us)
    # print(route_edge_vs)
    return route_edge_us, route_edge_vs, hanna_point_info, id_num_hanna_points
        #########connect to pin in pre line###########




def build_hanan_grid(pin_xs,pin_ys,nodes,num_hanna_points,h_net_density_grid,v_net_density_grid,pin_density_grid,node_density_grid,bin_x,bin_y):
    pre_num_hanna_points = num_hanna_points
    pos_x_dict = {}
    pos_y_dict = {}
    pin_point_dict = {}
    hanna_point_info = []
    pin_dict = {}
    route_edge_us = []
    route_edge_vs = []
    pin_edge_nodes = []
    pin_edge_hanna_points = []
    min_pos_x = min(pin_xs)
    min_pos_y = min(pin_ys)
    max_pos_x = max(pin_xs)
    max_pos_y = max(pin_ys)
    for pin_x,pin_y,node in zip(pin_xs,pin_ys,nodes):
        pos_x_dict.setdefault(pin_x,[]).append(pin_y)
        pos_y_dict.setdefault(pin_y,[]).append(pin_x)
        if (pin_x,pin_y) in pin_dict:
            pin_edge_hanna_points.append(pin_dict[(pin_x,pin_y)])
        else:
            pin_dict[(pin_x,pin_y)] = num_hanna_points
            pin_edge_hanna_points.append(num_hanna_points)
            hanna_point_info.append([pin_x,pin_y,pin_x,pin_y,
                                    feature_grid2hanna(h_net_density_grid,bin_x,bin_y,pin_x,pin_y,pin_x,pin_y),
                                    feature_grid2hanna(v_net_density_grid,bin_x,bin_y,pin_x,pin_y,pin_x,pin_y),
                                    feature_grid2hanna(pin_density_grid,bin_x,bin_y,pin_x,pin_y,pin_x,pin_y),
                                    feature_grid2hanna(node_density_grid,bin_x,bin_y,pin_x,pin_y,pin_x,pin_y),    
            ])
            num_hanna_points+=1
        pin_point_dict[(pin_x,pin_y)] = node
        pin_edge_nodes.append(node)
    
    route_edge_us, route_edge_vs, hanna_point_info, num_hanna_points = build_group_hanna_point(pos_x_dict,hanna_point_info,max_pos_y,min_pos_y,num_hanna_points,'x',
                            route_edge_us,route_edge_vs,pin_dict,h_net_density_grid,v_net_density_grid,pin_density_grid,node_density_grid,bin_x,bin_y)


    return route_edge_us, route_edge_vs, pin_edge_nodes, pin_edge_hanna_points, hanna_point_info, num_hanna_points - pre_num_hanna_points

def transform_graph2edges(graph):
    num_nets = graph.num_nodes(ntype='net')
    nets,cells = graph.edges(etype='pinned')
    edges_feats = graph.edges['pinned'].data['feats']
    edges = {}
    iter_info = zip(nets,cells,edges_feats)
    for net,cell,pin_feats in iter_info:
        pin_px, pin_py, pin_io = pin_feats
        edges.setdefault(net.item(),[]).append([cell.item(),pin_px.item(), pin_py.item(), pin_io.item()])
    return edges

def build_route_graph(graph,node_pos,h_net_density_grid,v_net_density_grid,pin_density_grid,node_density_grid,bin_x,bin_y):
    # edges = get_edge()
    # node_pos = get_pos()
    edges = transform_graph2edges(graph)
    edge_iter = edges.items()
    us = []
    vs = []
    route_edge_us = []
    route_edge_vs = []
    pin_edge_nodes = []
    pin_edge_hanna_points = []
    hanna_point_info = []
    num_hanna_points = 0
    total_time = 0
    total_init_time = 0
    total_route_time = 0
    total_append_time = 0
    for net,list_node_feats in edge_iter:
        # total_a = time.time()
        pin_xs = []
        pin_ys = []
        nodes = []
        for node,pin_px,pin_py,pin_io in list_node_feats:
            us.append(node)
            vs.append(net)
            nodes.append(node)
            px,py = node_pos[node,:]
            pin_xs.append(px + pin_px)
            pin_ys.append(py + pin_py)
        # total_init_time += time.time() - total_a
        # total_b = time.time()
        sub_route_edge_us,sub_route_edge_vs,sub_pin_edge_nodes,sub_pin_edge_hanna_points, sub_hanna_point_info, sub_num_hanna_point = build_hanan_grid(pin_xs,pin_ys,nodes,num_hanna_points,h_net_density_grid,v_net_density_grid,pin_density_grid,node_density_grid,bin_x,bin_y)
        # total_route_time += time.time() - total_b
        # total_b = time.time()
        route_edge_us.extend(sub_route_edge_us)
        route_edge_vs.extend(sub_route_edge_vs)
        pin_edge_nodes.extend(sub_pin_edge_nodes)
        pin_edge_hanna_points.extend(sub_pin_edge_hanna_points)
        hanna_point_info.extend(sub_hanna_point_info)
        num_hanna_points+=sub_num_hanna_point
        # total_append_time += time.time() - total_b
        # total_time += time.time() - total_a
    route_graph = dgl.heterograph({
        ('cell','pins','net'):(us,vs),
        ('net','pinned','cell'):(vs,us),
        ('cell','point-to','hanna'):(pin_edge_nodes,pin_edge_hanna_points),
        ('hanna','point-from','cell'):(pin_edge_hanna_points,pin_edge_nodes),
        ('hanna','connect','hanna'):(route_edge_us,route_edge_vs)
    },num_nodes_dict={'cell':graph.num_nodes(ntype='cell'),'net':graph.num_nodes(ntype='net'),'hanna':num_hanna_points})#
    route_graph.nodes['cell'].data['hv'] = graph.nodes['cell'].data['hv']
    route_graph.nodes['cell'].data['pos'] = graph.nodes['cell'].data['pos']
    route_graph.nodes['net'].data['hv'] = graph.nodes['net'].data['hv']
    route_graph.nodes['net'].data['degree'] = graph.nodes['net'].data['degree']
    route_graph.nodes['hanna'].data['hv'] = torch.tensor(hanna_point_info,dtype=torch.float32)

    route_graph.edges['pinned'].data['feats'] = graph.edges['pinned'].data['feats']
    # print("--------------")
    # print(f"total init time {total_init_time}s averange time is {total_init_time/len(edges)}s")
    # print(f"total route time {total_route_time}s averange time is {total_route_time/len(edges)}s")
    # print(f"total append time {total_append_time}s averange time is {total_append_time/len(edges)}s")
    # print(f"total time {total_time}s averange time is {total_time/len(edges)}s")
    return route_graph



def load_graph(netlist_dir):
    netlist_process_dir = f"{netlist_dir}_processed"

    edges = get_edge(netlist_process_dir)
    node_pos = get_pos(netlist_process_dir)

    cell_size_x,cell_size_y = get_cell_size(netlist_process_dir)

    node_pin_num = get_node_pin_num(edges, node_pos.shape[0])

    h_net_density_grid = get_h_net_density_grid(netlist_dir)
    v_net_density_grid = get_v_net_density_grid(netlist_dir)

    net2hpwl = get_net2hpwl(netlist_dir)
    net2hpwl[net2hpwl < 1e-4] = 1e-4
    net2hpwl = np.log10(net2hpwl)
    net2hpwl = torch.tensor(net2hpwl,dtype=torch.float32)

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
            px =x + pin_px
            py =y + pin_py
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

    # extra = fo_average(hetero_graph)

    hetero_graph.nodes['cell'].data['pos'] = node_pos
    hetero_graph.nodes['cell'].data['hv'] = node_hv
    hetero_graph.nodes['cell'].data['label'] = labels
    hetero_graph.nodes['net'].data['hv'] = net_span_feat
    hetero_graph.nodes['net'].data['degree'] = net_degree
    hetero_graph.nodes['net'].data['label'] = net2hpwl
    hetero_graph.edges['pinned'].data['feats'] = pins_feats
    # partition_list = partition_graph(hetero_graph,netlist_process_dir.split('/')[-1],netlist_process_dir)
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

    list_hetero_graph = []
    list_route_graph = []
    iter_partition_list = tqdm.tqdm(enumerate(partition_list), total=len(partition_list))
    total_route_time = 0
    total_sub_time = 0
    total_init_time = 0
    total_time = 0
    num_hanna_point = 0
    num_hanna_edges = 0
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
        sub_route_graph = build_route_graph(part_hetero_graph,sub_node_pos,
                            h_net_density_grid,
                            v_net_density_grid,
                            pin_density_grid,
                            node_density_grid,
                            32,40)
        # total_route_time += time.time() - total_b
        list_route_graph.append(sub_route_graph)
        # total_time += time.time() - total_a
        num_hanna_point += sub_route_graph.num_nodes(ntype='hanna')
        num_hanna_edges += sub_route_graph.num_edges(etype='connect')
    # print(f"total init time {total_init_time}s average time {total_init_time/len(partition_list)}s")
    # print(f"total sub time {total_sub_time}s average time {total_sub_time/len(partition_list)}s")
    # print(f"total route time {total_route_time}s average time {total_route_time/len(partition_list)}s")
    # print(f"total time {total_time}s average time {total_time/len(partition_list)}s")
    print(f"total hanna point {num_hanna_point}")
    print(f"total hanna edges {num_hanna_edges}")
    list_tuple_graph = list(zip(list_hetero_graph, list_route_graph))
    # with open(osp.join(netlist_process_dir,"graph.pickle"),"wb+") as f:
    #     pickle.dump(list_tuple_graph, f)
    # return num_hanna_point,num_hanna_edges
    return list_hetero_graph,list_route_graph


if __name__ == '__main__':
    netlist_name_list = [
        # 'superblue1',
        # 'superblue2',
        # 'superblue3',
        # 'superblue5',
        # 'superblue6',
        # 'superblue7',
        # 'superblue9',
        # 'superblue11',
        # 'superblue14',
        # 'superblue16',
        'superblue19',
    ]
    netlist_info = []
    for netlist_name in netlist_name_list:
        num_hanna_point,num_hanna_edges = load_graph(f"/root/autodl-tmp/data/{netlist_name}")
        netlist_info.append([netlist_name,num_hanna_point,num_hanna_edges])
        # netlist_info = np.array(netlist_info)
        # df = pd.DataFrame()
        # df['netlist_name'] = netlist_info[:,0]
        # df['num hanan points'] = netlist_info[:,1]
        # df['hanan edges'] = netlist_info[:,2]
        # df.to_excel("./group_hanna_info.xlsx",index=False)
        # netlist_info = list(netlist_info)
    # netlist_info = np.array(netlist_info)
    # df = pd.DataFrame()
    # df['netlist_name'] = netlist_info[:,0]
    # df['num hanan points'] = netlist_info[:,1]
    # df['hanan edges'] = netlist_info[:,2]
    # df.to_excel("./group_hanna_info.xlsx",index=False)
    # print(graph)
    # with open("/root/test.pickle",'wb+') as fp:
    #     pickle.dump(graph,fp)
    # get_w()



def build_hanan_grid_simple(pin_xs,pin_ys,nodes,num_hanna_points):
    pin_point_dict = {}
    hanna_point_dict = {}
    for pin_x,pin_y,node in zip(pin_xs,pin_ys,nodes):
        pin_point_dict[(pin_x,pin_y)] = node
    sorted_pin_xs = pin_xs.copy()
    sorted_pin_ys = pin_ys.copy()
    sorted_pin_xs.sort()
    sorted_pin_ys.sort()
    id_hanna_point = 0
    for x,y in itertools.product(sorted_pin_xs,sorted_pin_ys):#这步必须保证是按序进行笛卡尔积的！！！！！
        if (x,y) in pin_point_dict:
            pin2node = pin_point_dict[(x,y)]
        else:
            pin2node = -1
        hanna_point_dict[(x,y)] = (id_hanna_point,pin2node)
        id_hanna_point+=1
    
    route_edge_us = []
    route_edge_vs = []
    pin_edge_nodes = []
    pin_edge_hanna_points = []
    num_pins = len(pin_xs)
    for id_hanna_point,pin2node in hanna_point_dict.values():
        id_x = id_hanna_point % num_pins
        id_y = int(id_hanna_point / num_pins)
        for dx,dy in [[0,1],[0,-1],[1,0],[-1,0]]:
            id_x_ = id_x + dx
            id_y_ = id_y + dy
            if id_x_ < 0 or id_x_ >= num_pins or id_y_ < 0 or id_y_ >= num_pins:
                continue
            id_hanna_point_ = id_x_ + id_y_ * num_pins
            route_edge_us.append(id_hanna_point + num_hanna_points)
            route_edge_vs.append(id_hanna_point_ + num_hanna_points)
        if pin2node != -1:
            pin_edge_nodes.append(pin2node)
            pin_edge_hanna_points.append(id_hanna_point + num_hanna_points)
    return route_edge_us,route_edge_vs,pin_edge_nodes,pin_edge_hanna_points,id_hanna_point