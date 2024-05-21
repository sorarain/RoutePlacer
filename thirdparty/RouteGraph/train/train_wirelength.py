import json
import os
import argparse
from time import time
from typing import List, Dict, Any
from functools import reduce

import numpy as np
import dgl

import torch
import torch.nn as nn

from data.load_data import load_data
from model.RouteGNN import NetlistGNN
from log.store_scatter import store_scatter
from utils.output import printout_xf1

import warnings


def train_wirelength(
    args,
    netlists_dir=None,
    train_netlists_names=None,
    validation_netlists_names = None,
    test_netlists_names = None,
    log_dir = None,
    fig_dir = None,
    model_dir = None
):
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
    }

    def load_netlists(netlists_names:List[str]):
        list_tuple_graph = []
        for netlist_name in netlists_names:
            list_tuple_graph.extend(load_data(os.path.join(netlists_dir,netlist_name)))
        return list_tuple_graph


    train_list_tuple_graph, validate_list_tuple_graph, test_list_tuple_graph = load_netlists(train_netlists_names), load_netlists(validation_netlists_names), load_netlists(test_netlists_names)

    n_train_cell = sum(map(lambda x: x[0].num_nodes(ntype='cell'), train_list_tuple_graph))
    n_train_net = sum(map(lambda x: x[0].num_nodes(ntype='net'), train_list_tuple_graph))
    n_validate_cell = sum(map(lambda x: x[0].num_nodes(ntype='cell'), validate_list_tuple_graph))
    n_validate_net = sum(map(lambda x: x[0].num_nodes(ntype='net'), validate_list_tuple_graph))
    n_test_cell = sum(map(lambda x: x[0].num_nodes(ntype='cell'), test_list_tuple_graph))
    n_test_net = sum(map(lambda x: x[0].num_nodes(ntype='net'), test_list_tuple_graph))

    print('##### MODEL #####')
    in_node_feats = train_list_tuple_graph[0][1].nodes['cell'].data['hv'].shape[1]+2*args.add_pos
    in_net_feats = train_list_tuple_graph[0][1].nodes['net'].data['hv'].shape[1]
    in_hanna_feats = train_list_tuple_graph[0][1].nodes['hanna'].data['hv'].shape[1]
    in_pin_feats = train_list_tuple_graph[0][1].edges['pinned'].data['feats'].shape[1]
    model = NetlistGNN(
        in_node_feats=in_node_feats,
        in_net_feats=in_net_feats,
        in_hanna_feats=in_hanna_feats,
        in_pin_feats=in_pin_feats,
        config=config,
        n_target=1,
        activation=args.outtype,
        topo_conv_type=args.topo_conv_type,
        agg_type=args.agg_type,
        cat_raw=args.cat_raw
    ).to(device)
    if args.model:
        model_dicts = torch.load(f'param/{args.model}.pkl', map_location=device)
        model.load_state_dict(model_dicts)
        model.eval()
    n_param = 0
    for name, param in model.named_parameters():
        print(f'\t{name}: {param.shape}')
        n_param += reduce(lambda x, y: x * y, param.shape)
    print(f'# of parameters: {n_param}')

    if args.beta < 1e-5:
        print(f'### USE L1Loss ###')
        loss_f = nn.L1Loss()
    elif args.beta > 7.0:
        print(f'### USE MSELoss ###')
        loss_f = nn.MSELoss()
    else:
        print(f'### USE SmoothL1Loss with beta={args.beta} ###')
        loss_f = nn.SmoothL1Loss(beta=args.beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))

    best_rmse = 1e9


    def to_device(a, b):
        return a.to(device), b.to(device)
    
    def forward(hanna_graph):
        if args.add_pos:
            in_node_feat = torch.cat([hanna_graph.nodes['cell'].data['hv'],hanna_graph.nodes['cell'].data['pos']],dim=-1)
        else:
            in_node_feat = hanna_graph.nodes['cell'].data['hv']
        in_net_feat = hanna_graph.nodes['net'].data['hv']
        in_pin_feat = hanna_graph.edges['pinned'].data['feats']
        in_hanna_feat = hanna_graph.nodes['hanna'].data['hv']
        pred_cell, pred_net = model.forward(in_node_feat=in_node_feat,in_net_feat=in_net_feat,
                                in_pin_feat=in_pin_feat,in_hanna_feat=in_hanna_feat,node_net_graph=hanna_graph)
        return pred_cell, pred_net


    for epoch in range(0, args.epochs + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})


        def train(ltg):
            if args.trans:
                for p in model.net_readout_params:
                    p.train()
            else:
                model.train()
            t1 = time()
            losses = []
            n_tuples = len(ltg)
            for j, (hetero_graph, hanna_graph) in enumerate(ltg):
                hetero_graph, hanna_graph = to_device(hetero_graph, hanna_graph)
                optimizer.zero_grad()
                _, pred = forward(hanna_graph)
                batch_labels = hetero_graph.nodes['net'].data['label']
                pred = pred.squeeze()
                loss = loss_f(pred.view(-1), batch_labels.float())
                losses.append(loss)
                if len(losses) >= args.batch or j == n_tuples - 1:
                    sum(losses).backward()
                    optimizer.step()
                    losses.clear()
            scheduler.step()
            print(f"\tTraining time per epoch: {time() - t1}")


        def evaluate(ltg, set_name):
            model.eval()
            print(f'\tEvaluate {set_name}:')
            n_net = sum(map(lambda x: x[0].num_nodes(ntype='net'),ltg))
            all_tgt = torch.zeros([n_net],dtype=torch.float32,device=device)
            all_prd = torch.zeros([n_net],dtype=torch.float32,device=device)
            weight = 1e-5 * torch.ones([n_net],dtype=torch.float32,device=device)
            with torch.no_grad():
                for j, (hetero_graph, hanna_graph) in enumerate(ltg):
                    hetero_graph, hanna_graph = to_device(hetero_graph, hanna_graph)
                    _, prd = forward(hanna_graph)
                    index = hetero_graph.nodes['net'].data[dgl.NID]
                    all_prd[index] += prd.squeeze()
                    weight[index] += 1
                    all_tgt[index] = hetero_graph.nodes['net'].data['label']
            all_prd *= 1.0 / weight
            all_tgt, all_prd = np.array(all_tgt.cpu()), np.array(all_prd.cpu())
            d = printout_xf1(all_tgt, all_prd, "\t\t", f'{set_name}')
            logs[-1].update(d)
            store_scatter(all_tgt, all_prd, f'{args.name}-{set_name}', epoch=epoch, fig_dir=fig_dir)
            nonlocal best_rmse
            if best_rmse > d[f"{set_name}rmse"] and set_name == 'validate_':
                best_rmse = d[f"{set_name}rmse"]
                print(f"\tSaving model to {model_dir}/{args.name}.pkl ...:")
                torch.save(model.state_dict(), f'{model_dir}/{args.name}.pkl')


        t0 = time()
        if epoch:
            for _ in range(args.train_epoch):
                train(train_list_tuple_graph)
        logs[-1].update({'train_time': time() - t0})
        t2 = time()
        evaluate(train_list_tuple_graph, 'train_')
        evaluate(validate_list_tuple_graph, 'validate_')
        evaluate(test_list_tuple_graph, 'test_')
        print("\tinference time", time() - t2)
        logs[-1].update({'eval_time': time() - t2})
        with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
            json.dump(logs, fp)
