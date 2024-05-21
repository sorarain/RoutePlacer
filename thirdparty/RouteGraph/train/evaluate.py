import json
import os
from time import time
from datetime import datetime
import tqdm
import numpy
import pandas
from typing import List, Dict, Any, Tuple
from functools import reduce
import pickle
import numpy as np
import dgl
import argparse
import torch
import torch.nn as nn

from data.load_data_optimize import load_data,build_grid_graph
from data.load_netlist import build_near_graph
from model.RouteGNN import RouteGNN
from model.NetlistGNN import NetlistGNN
from utils.output import printout, get_grid_level_corr, mean_dict
from log.store_cong import store_cong_from_node
import warnings
import signal

timeout=1200


def train_congestion(
        args,
        netlists_dir = None,
        train_netlists_names=None,
        validation_netlists_names = None,
        test_netlists_names = None,
        log_dir = None,
        fig_dir = None,
        model_dir = None,):
    warnings.filterwarnings("ignore")
    np.set_printoptions(precision=3, suppress=True)

    logs: List[Dict[str, Any]] = []
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)
    if not args.device == 'cpu':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)

    config = {
        'N_LAYER': args.layers,
        'NODE_FEATS': args.node_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'HANNA_FEATS': args.hanna_feats,
        'EDGE_FEATS': args.edge_feats,
    }

    # load data
    def load_netlists(netlists_names:List[str]):
        list_tuple_graph = []
        list_input = []
        for netlist_name in netlists_names:
            tuple_graph,batch_input = load_data(os.path.join(netlists_dir,netlist_name),args)
            list_tuple_graph.append(tuple_graph)
            list_input.append(batch_input)

        return list_tuple_graph,list_input
    
    validation_list_netlist,validation_list_input = load_netlists(validation_netlists_names)

    print('###MODEL###')
    #model feature sizes
    # in_node_feats = train_list_netlist[0][0][1].nodes['cell'].data['hv'].shape[1]+2*args.add_pos
    # in_net_feats = train_list_netlist[0][0][1].nodes['net'].data['hv'].shape[1]
    # in_hanna_feats = train_list_netlist[0][1][1].nodes['gcell'].data['hv'].shape[1]
    # in_pin_feats = train_list_netlist[0][0][1].edges['pinned'].data['feats'].shape[1]
    # in_edge_feats = train_list_netlist[0][0][1].edges['point-from'].data['feats'].shape[1]
    in_node_feats = 5
    in_net_feats = 7
    in_hanna_feats = 4
    in_pin_feats = 3
    in_edge_feats = 1

    if args.name == 'netlistgnn':
        model = NetlistGNN(
        in_node_feats=in_node_feats,
        in_net_feats=in_net_feats,
        in_pin_feats=in_pin_feats,
        in_edge_feats=in_edge_feats,
        config=config,
        n_target=1,
        # activation=args.outtype,
        topo_conv_type=args.topo_conv_type,
        geom_conv_type=args.grid_conv_type,
        agg_type=args.agg_type,
        cat_raw=args.cat_raw
    ).to(device)
    else:
        model = RouteGNN(
            in_node_feats=in_node_feats,
            in_net_feats=in_net_feats,
            in_hanna_feats=in_hanna_feats,
            in_pin_feats=in_pin_feats,
            in_edge_feats=in_edge_feats,
            config=config,
            n_target=1,
            # activation=args.outtype,
            topo_conv_type=args.topo_conv_type,
            grid_conv_type=args.grid_conv_type,
            agg_type=args.agg_type,
            cat_raw=args.cat_raw
        ).to(device)
    #load model
    if args.model:
        model_dicts = torch.load(f'param/{args.model}.pkl', map_location=device)
        model.load_state_dict(model_dicts)
        model.eval()
    n_param = 0
    for name, param in model.named_parameters():
        print(f'\t{name}: {param.shape}')
        n_param += reduce(lambda x, y: x * y, param.shape)
    print(f'# of parameters: {n_param}')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch, eta_min=args.min_lr)
    # if args.beta < 1e-5:
    #     print(f'### USE L1Loss ###')
    #     loss_f = nn.L1Loss()
    # elif args.beta > 7.0:
    #     print(f'### USE MSELoss ###')
    #     loss_f = nn.MSELoss()
    # else:
    #     print(f'### USE SmoothL1Loss with beta={args.beta} ###')
    #     loss_f = nn.SmoothL1Loss(beta=args.beta)

    best_rmse = 1e8

    def forward(hanna_graph):
        if args.add_pos:
            in_node_feat = torch.cat([hanna_graph.nodes['cell'].data['hv'],hanna_graph.nodes['cell'].data['pos']],dim=-1)
        else:
            in_node_feat = hanna_graph.nodes['cell'].data['hv']
        in_net_feat = hanna_graph.nodes['net'].data['hv']
        in_pin_feat = hanna_graph.edges['pinned'].data['feats']
        if args.name == 'netlistgnn':
            in_edge_feat = hanna_graph.edges['near'].data['feats']
            pred_cell, _ = model.forward(in_node_feat=in_node_feat,in_net_feat=in_net_feat,
                                in_pin_feat=in_pin_feat,in_edge_feat=in_edge_feat,node_net_graph=hanna_graph)
        else:
            in_hanna_feat = hanna_graph.nodes['gcell'].data['hv']
            pred_cell, _ = model.forward(in_node_feat=in_node_feat,in_net_feat=in_net_feat,
                                    in_pin_feat=in_pin_feat,in_hanna_feat=in_hanna_feat,node_net_graph=hanna_graph)
        # if args.scalefac:
        #     pred_cell = pred_cell * args.scalefac
        #     pred_net = pred_net * args.scalefac
        
        return pred_cell
                    
    def evaluate(ltgs: List[List[Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]]], ltinput:List[List[Dict]], set_name:str,
                 explicit_names:List[str], verbose =True):
        model.eval()
        ds =[]
        for data_name,tuple_graph,batch_input in zip(explicit_names, ltgs, ltinput):
            hetero_graph,list_hetero_graph = tuple_graph
            list_route_graph = []
            cnt = 0
            for input_dict in batch_input:
                n_node = hetero_graph.num_nodes(ntype='cell')
                outputdata = np.zeros((n_node, 5))
                h_net_density_grid = input_dict['h_net_density_grid']
                v_net_density_grid = input_dict['v_net_density_grid']
                p=0
                cnt += 1
                # if cnt < 3:
                #     continue
                try:
                    signal.alarm(timeout)
                    t0 = time()
                    for sub_hetero_graph in list_hetero_graph:
                        sub_hetero_graph.nodes['cell'].data['pos'] = input_dict['pos'][sub_hetero_graph.nodes['cell'].data[dgl.NID]]
                        sub_hetero_graph.nodes['cell'].data['hv'] = input_dict['hv'][sub_hetero_graph.nodes['cell'].data[dgl.NID],:]
                        sub_hetero_graph.nodes['cell'].data['label'] = input_dict['label'][sub_hetero_graph.nodes['cell'].data[dgl.NID]]
                        sub_hetero_graph.nodes['net'].data['hv'] = input_dict['net_hv'][sub_hetero_graph.nodes['net'].data[dgl.NID],:]
                        
                        sub_node_pos = input_dict['pos'][sub_hetero_graph.nodes['cell'].data[dgl.NID]]
                        if args.name == 'netlistgnn':
                            sub_route_graph = build_near_graph(sub_hetero_graph,sub_node_pos,
                                h_net_density_grid,
                                v_net_density_grid,
                                input_dict['bin_x'],input_dict['bin_y'])
                        else:
                            sub_route_graph = build_grid_graph(sub_hetero_graph,sub_node_pos,
                                    h_net_density_grid,
                                    v_net_density_grid,
                                    input_dict['bin_x'],input_dict['bin_y'])
                        sub_route_graph = sub_route_graph.to(device)
                        optimizer.zero_grad()
                        cell_pred = forward(sub_route_graph)
                        cell_label = sub_hetero_graph.nodes['cell'].data['label']
                        ln = len(cell_label)
                        # density = hetero_graph.nodes['cell'].data['hv'][:,6].cpu().data.numpy()
                        density = torch.from_numpy(input_dict['node_density_grid'])[sub_hetero_graph.nodes['cell'].data[dgl.NID]]
                        output_pos = (sub_hetero_graph.nodes['cell'].data['pos'].cpu().data.numpy())
                        ######
                        density[np.isinf(1.0 / (output_pos[:,0] + output_pos[:,1]))] = 0.0
                        assert not torch.any(torch.isinf(cell_pred))
                        assert not torch.any(torch.isnan(cell_pred))
                        ######

                        # print(p,n_node,ln,cell_label.size(),cell_pred.size())
                        outputdata[p:p+ln,0], outputdata[p:p+ln,1] = cell_label.cpu().detach().numpy().flatten(), cell_pred.cpu().detach().numpy().flatten()
                        outputdata[p:p+ln,2:4], outputdata[p:p+ln, 4] = output_pos, density
                        p += ln
                except TimeoutError as e:
                    eval_time = 1200
                    print(f"time exceed 20min")
                else:
                    eval_time = time() - t0
                    print(f"eval time is {eval_time}")
                # bad_node = outputdata[:, 4] < 0.5
                # outputdata[bad_node, 1] = outputdata[bad_node, 0]
                # d = printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}node_level_',
                #     verbose=False)
                # d1, d2 = get_grid_level_corr(outputdata[:, :4], args.binx, args.biny,
                #                             int(np.rint(np.max(outputdata[:, 2]) / args.binx)) + 1,
                #                             int(np.rint(np.max(outputdata[:, 3]) / args.biny)) + 1,
                #                             set_name=set_name, verbose=False)
                # d.update(d1)
                # d.update(d2)
                d={}
                d['time'] = eval_time
                ds.append(d)
                
        mean_ds = mean_dict(ds)
        logs[-1].update(mean_ds)
        if log_dir is not None:
            with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
                json.dump(ds, fp)

    for epoch in range(0, args.train_epoch + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})
        t0 = time()
        # evaluate(train_list_netlist, train_list_input, 'train_', train_netlists_names, verbose=False)
        evaluate(validation_list_netlist, validation_list_input, 'validate_', validation_netlists_names)
        # if len(test_list_netlist):
        #     evaluate(test_list_netlist, test_list_input, 'test_', test_netlists_names)
        print("\tinference time", time() - t0)
        logs[-1].update({'eval_time': time() - t0})
        # if log_dir is not None:
        #     with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
        #         json.dump(logs, fp)








