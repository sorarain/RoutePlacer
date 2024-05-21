import os
import sys

sys.path.append(os.path.abspath("."))

import numpy as np
import torch
import dgl
import dgl.function as fn
import pickle
import tqdm
import os.path as osp

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
    max_num=2000
    for i in range(max_num):
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
    max_num=2000
    for i in range(max_num):
        if os.path.exists(f"{netlist_dir}/iter_{i}_bad_cmap_h.npy"):
            num = i
            break
    assert os.path.exists(f"{netlist_dir}/iter_{num}_bad_cmap_h.npy"),f"{netlist_dir}/iter_{num}_bad_cmap_h.npy"
    return np.load(f"{netlist_dir}/iter_{num}_bad_cmap_h.npy")
    # return 0.5 * np.ones((1000,1000))

def get_v_net_density_grid(netlist_dir) -> np.ndarray:
    num = -1
    max_num=2000
    for i in range(max_num):
        if os.path.exists(f"{netlist_dir}/iter_{i}_bad_cmap_v.npy"):
            num = i
            break
    assert os.path.exists(f"{netlist_dir}/iter_{num}_bad_cmap_v.npy")
    return np.load(f"{netlist_dir}/iter_{num}_bad_cmap_v.npy")
    # return 0.5 * np.ones((1000,1000))

def get_net2hpwl(netlist_dir) -> np.ndarray:
    num = -1
    max_num=2000
    netlist_dir = os.path.join("/",*netlist_dir.split("/")[:-1],"superblue_0425_withHPWL",netlist_dir.split("/")[-1])
    for i in range(max_num):
        if os.path.exists(f"{netlist_dir}/iter_{i}_net2hpwl.npy"):
            num = i
            break
    assert os.path.exists(f"{netlist_dir}/iter_{num}_net2hpwl.npy")
    return np.load(f"{netlist_dir}/iter_{num}_net2hpwl.npy")

def get_node_congestion_label(netlist_dir) -> np.ndarray:
    num = -1
    max_num=2000
    for i in range(max_num):
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

def get_node_density_vectorized(density_map, bin_x, bin_y, node_pos) -> np.ndarray: # vectorized, x5 faster
    node_density = np.zeros_like(density_map, dtype=np.float32)
    index_x, index_y = (node_pos[:, 0] // bin_x).astype(int), (node_pos[:, 1] // bin_y).astype(int)
    unique_pos, counts = np.unique((index_x, index_y), axis=1, return_counts=True)
    node_density[unique_pos[0], unique_pos[1]] = counts
    return node_density

def feature_grid2node(grid_feature: np.ndarray, bin_x, bin_y, node_pos) -> np.ndarray:
    return np.array([
        grid_feature[int (x / bin_x),int(y / bin_y) ] for x,y in node_pos
    ],dtype=np.float32)
    
def feature_grid2node_vectorized(grid_feature: np.ndarray, bin_x, bin_y, node_pos: np.ndarray) -> np.ndarray: # vectorized, x100 faster
    x_idx, y_idx = (node_pos[:, 0]//bin_x).astype(int), (node_pos[:, 1]//bin_y).astype(int)   
    node_features = grid_feature[x_idx, y_idx]
    return node_features

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

def fo_average(g):
    device = torch.device("cuda")
    g = g.to(device)
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    g.ndata['addnlfeat'] = (g.ndata['feat']) / degrees.view(-1, 1)
    g.ndata['inter'] = torch.zeros_like(g.ndata['feat'],device=device)
    g.ndata['wts'] = torch.ones(g.number_of_nodes(),device=device) / degrees
    g.ndata['wtmsg'] = torch.zeros_like(g.ndata['wts'],device=device)
    g.update_all(message_func=fn.copy_u(u='addnlfeat', out='inter'),
                 reduce_func=fn.sum(msg='inter', out='addnlfeat'))
    g.update_all(message_func=fn.copy_u(u='wts', out='wtmsg'),
                 reduce_func=fn.sum(msg='wtmsg', out='wts'))
    hop1 = g.ndata['addnlfeat'] / (g.ndata['wts'].view(-1, 1))
    return hop1.cpu()