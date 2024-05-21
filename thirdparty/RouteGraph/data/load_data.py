import os
import os.path as osp
import sys

sys.path.append(os.path.abspath("."))
from data.graph import load_graph

import numpy as np
import itertools
import dgl
import pickle
import tqdm
import pandas as pd
from queue import Queue
import torch
import time
from data.graph import load_graph
from data.GridGraph import load_grid_graph


def load_data(netlist_dir:str,save_type:int=1):
    if save_type == 1 and os.path.exists(osp.join(netlist_dir,'graph_geompart.pickle')):
        with open(osp.join(netlist_dir,'graph_geompart.pickle'),"rb") as f:
            list_tuple_graph = pickle.load(f)
            return list_tuple_graph
    list_hetero_graph,list_route_graph = load_graph(netlist_dir)
    list_tuple_graph = list(zip(list_hetero_graph, list_route_graph))
    with open(osp.join(netlist_dir,'graph_geompart.pickle'),"wb") as f:
        pickle.dump(list_tuple_graph, f)
    return list_tuple_graph

def load_grid_data(netlist_dir:str,args,save_type:int=1):
    print(f"load {netlist_dir.split('/')[-1]}:")
    if save_type == 1 and os.path.exists(osp.join(netlist_dir,f'{args.app_name}graph_grid.pickle')):
        with open(osp.join(netlist_dir,f'{args.app_name}graph_grid.pickle'),"rb") as f:
            list_tuple_graph = pickle.load(f)
            return list_tuple_graph
    list_hetero_graph,list_route_graph = load_grid_graph(netlist_dir)
    list_tuple_graph = list(zip(list_hetero_graph, list_route_graph))
    with open(osp.join(netlist_dir,f'{args.app_name}graph_grid.pickle'),"wb") as f:
        pickle.dump(list_tuple_graph, f)
    return list_tuple_graph

if __name__ == '__main__':
    # load_data("/root/autodl-tmp/data/superblue19",2)
    load_grid_data('/root/autodl-tmp/data/superblue19',2)
