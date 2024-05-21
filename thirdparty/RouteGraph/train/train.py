import json
import os
from time import time

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

from data.load_data import load_data,load_grid_data
from model.RouteGNN import NetlistGNN
from utils.output import printout, get_grid_level_corr, mean_dict
from log.store_cong import store_cong_from_node

import warnings


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
        for netlist_name in netlists_names:
            list_tuple_graph.append(load_grid_data(os.path.join(netlists_dir,netlist_name),args))
        return list_tuple_graph
    
    train_list_netlist = load_netlists(train_netlists_names)
    validation_list_netlist = load_netlists(validation_netlists_names)
    test_list_netlist = load_netlists(test_netlists_names)

    #temporory training/ testing set


    print('###MODEL###')
    #model feature sizes
    in_node_feats = train_list_netlist[0][0][1].nodes['cell'].data['hv'].shape[1]+2*args.add_pos
    in_net_feats = train_list_netlist[0][0][1].nodes['net'].data['hv'].shape[1]
    in_hanna_feats = train_list_netlist[0][1][1].nodes['gcell'].data['hv'].shape[1]
    in_pin_feats = train_list_netlist[0][0][1].edges['pinned'].data['feats'].shape[1]
    in_edge_feats = train_list_netlist[0][0][1].edges['point-from'].data['feats'].shape[1]

    model = NetlistGNN(
        in_node_feats=in_node_feats,
        in_net_feats=in_net_feats,
        in_hanna_feats=in_hanna_feats,
        in_pin_feats=in_pin_feats,
        in_edge_feats=in_edge_feats,
        config=config,
        n_target=1,
        activation=args.outtype,
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))
    if args.beta < 1e-5:
        print(f'### USE L1Loss ###')
        loss_f = nn.L1Loss()
    elif args.beta > 7.0:
        print(f'### USE MSELoss ###')
        loss_f = nn.MSELoss()
    else:
        print(f'### USE SmoothL1Loss with beta={args.beta} ###')
        loss_f = nn.SmoothL1Loss(beta=args.beta)

    best_rmse = 1e8

    def to_device(a,b):
        return a.to(device), b.to(device)
    def forward(hanna_graph):
        if args.add_pos:
            in_node_feat = torch.cat([hanna_graph.nodes['cell'].data['hv'],hanna_graph.nodes['cell'].data['pos']],dim=-1)
        else:
            in_node_feat = hanna_graph.nodes['cell'].data['hv']
        in_net_feat = hanna_graph.nodes['net'].data['hv']
        in_pin_feat = hanna_graph.edges['pinned'].data['feats']
        in_hanna_feat = hanna_graph.nodes['gcell'].data['hv']
        pred_cell, pred_net = model.forward(in_node_feat=in_node_feat,in_net_feat=in_net_feat,
                                in_pin_feat=in_pin_feat,in_hanna_feat=in_hanna_feat,node_net_graph=hanna_graph)
        if args.scalefac:
            pred_cell = pred_cell * args.scalefac
            pred_net = pred_net * args.scalefac
        return pred_cell, pred_net

    #training
    def train(ltgs:List[List[Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]]]):
        ltg = []
        for ltg_ in ltgs:
            ltg.extend(ltg_)
        if args.trans:
            for p in model.net_readout_params:
                p.train()
        else:
            model.train()
        t1 = time()
        losses = []
        n_tuples = len(ltg)
        for i, (hetero_graph, hanna_graph) in enumerate(ltg):
            hetero_graph, hanna_graph = to_device(hetero_graph,hanna_graph)
            optimizer.zero_grad()
            cell_pred, net_pred = forward(hanna_graph)
            cell_label = hetero_graph.nodes['cell'].data['label']
            weight = 1.0 / hetero_graph.nodes['cell'].data['hv'][:, 6]
            weight[torch.isinf(weight)] = 0.0
            #########
            weight[torch.isinf(1.0 / (hetero_graph.nodes['cell'].data['pos'][:, 0] + hetero_graph.nodes['cell'].data['pos'][:, 1]))] = 0.0
            ########
            # cell_loss =loss_f(cell_pred.squeeze().view(-1), cell_label.float())
            cell_loss = torch.sum(((cell_pred.view(-1) - cell_label.float()) ** 2) * weight) / (torch.sum(weight)+1e-4)
            loss = cell_loss
            assert not torch.any(torch.isnan(cell_pred)),f"{torch.where(torch.isnan(cell_pred))}"
            assert not torch.any(torch.isinf(cell_pred))
            assert not torch.isnan(cell_loss),f"{weight} {torch.sum(weight)} {torch.isnan(cell_pred)}"
            assert not torch.isinf(cell_loss)
            losses.append(loss)
            if len(losses) >= args.batch or i == n_tuples - 1:
                sum(losses).backward()
                optimizer.step()
                losses.clear()
        scheduler.step()
        print(f"\tTraining time per epoch: {time() - t1}")

    def evaluate(ltgs: List[List[Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]]], set_name:str,
                 explicit_names:List[str], verbose =True):
        model.eval()
        ds =[]
        for data_name, ltg in zip(explicit_names, ltgs):
            n_node = sum(map(lambda x: x[0].num_nodes(ntype='cell'), ltg))
            outputdata = np.zeros((n_node, 5))
            p = 0
            for i, (hetero_graph, hanna_graph) in enumerate(ltg):
                hetero_graph, hanna_graph = to_device(hetero_graph, hanna_graph)
                cell_label = hetero_graph.nodes['cell'].data['label'].cpu().data.numpy().flatten()
                ln = len(cell_label)
                cell_pred, net_pred = forward(hanna_graph)
                cell_pred = cell_pred.cpu().data.numpy().flatten()
                density = hetero_graph.nodes['cell'].data['hv'][:,6].cpu().data.numpy()
                output_pos = (hetero_graph.nodes['cell'].data['pos'].cpu().data.numpy())
                ######
                density[np.isinf(1.0 / (output_pos[:,0] + output_pos[:,1]))] = 0.0
                assert not np.any(np.isinf(cell_pred))
                assert not np.any(np.isnan(cell_pred))
                ######

                outputdata[p:p+ln,0], outputdata[p:p+ln,1] = cell_label, cell_pred
                outputdata[p:p+ln,2:4], outputdata[p:p+ln, 4] = output_pos, density
                p += ln
            outputdata = outputdata[:p, :]
        if args.topo_geom != 'topo':
            bad_node = outputdata[:, 4] < 0.5
            outputdata[bad_node, 1] = outputdata[bad_node, 0]
        print(f'\t{data_name}:')
        d = printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}node_level_',
                     verbose=verbose)
        # save model
        if fig_dir is not None and data_name == 'superblue19':
            store_cong_from_node(outputdata[:, 0], outputdata[:, 1], outputdata[:, 2], outputdata[:, 3],
                                    args.binx, args.biny, [321, 518],
                                    f'{args.name}-{data_name}', epoch=epoch, fig_dir=fig_dir)

        if model_dir is not None and set_name == 'validate_':
            rmse = d[f'{set_name}node_level_f1']
            nonlocal best_rmse
            if rmse < best_rmse:
                best_rmse = rmse
                print(f'\tSaving model to {model_dir}/{args.name}.pkl ...:')
                torch.save(model.state_dict(), f'{model_dir}/{args.name}.pkl')
        d1, d2 = get_grid_level_corr(outputdata[:, :4], args.binx, args.biny,
                                     int(np.rint(np.max(outputdata[:, 2]) / args.binx)) + 1,
                                     int(np.rint(np.max(outputdata[:, 3]) / args.biny)) + 1,
                                     set_name=set_name, verbose=verbose)
        d.update(d1)
        d.update(d2)
        ds.append(d)
        logs[-1].update(mean_dict(ds))

    for epoch in range(0, args.epochs + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})
        t0 = time()
        if epoch:
            for _ in range(args.train_epoch):
                train(train_list_netlist)
        logs[-1].update({'train_time': time() - t0})
        t2  = time()
        evaluate(train_list_netlist, 'train_', train_netlists_names, verbose=False)
        if len(validation_list_netlist):
            evaluate(validation_list_netlist, 'validate_', validation_netlists_names)
        if len(test_list_netlist):
            evaluate(test_list_netlist, 'test_', test_netlists_names)
        print("\tinference time", time() - t2)
        logs[-1].update({'eval_time': time() - t2})
        if log_dir is not None:
            with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
                json.dump(logs, fp)








