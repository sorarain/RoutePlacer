import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import NNConv, SAGEConv, GATConv, HeteroGraphConv, GraphConv, CFConv
from typing import Tuple, Dict, Any, List, Union
import json
import pickle
import tqdm
import copy


class NodeNetGNN(nn.Module):
    def __init__(self, hidden_node_feats: int, hidden_net_feats: int, hidden_pin_feats: int, hidden_hanna_feats: int, hidden_edge_feats: int,
                 out_node_feats: int, out_net_feats: int, topo_conv_type, grid_conv_type, agg_type):
        super(NodeNetGNN, self).__init__()
        assert topo_conv_type in ['MPNN', 'SAGE', 'CFCNN', 'GCN'], f'{topo_conv_type} not in MPNN/SAGE/CFCNN/GCN'
        assert grid_conv_type in ['MPNN', 'SAGE', 'CFCNN', 'GCN'], f'{grid_conv_type} not in MPNN/SAGE/CFCNN/GCN'
        self.topo_conv_type = topo_conv_type
        self.grid_conv_type = grid_conv_type
        self.net_lin = nn.Linear(hidden_net_feats, hidden_net_feats)
        self.topo_lin = nn.Linear(hidden_pin_feats, hidden_net_feats * out_node_feats)
        self.topo_weight = nn.Linear(hidden_pin_feats, 1)
        self.grid_weight = nn.Linear(hidden_edge_feats, 1)

        def topo_edge_func(efeat):
            return self.topo_lin(efeat)

        
        self.hetero_conv = HeteroGraphConv({
            'pins': GraphConv(in_feats=hidden_node_feats, out_feats=out_net_feats),
            'pinned':
                NNConv(in_feats=hidden_net_feats, out_feats=out_node_feats,
                       edge_func=topo_edge_func) if topo_conv_type == 'MPNN' else
                SAGEConv(in_feats=(hidden_net_feats, hidden_node_feats), out_feats=out_node_feats,
                         aggregator_type='pool') if topo_conv_type == 'SAGE' else
                CFConv(node_in_feats=hidden_net_feats, edge_in_feats=hidden_pin_feats,
                       hidden_feats=hidden_node_feats, out_feats=out_node_feats) if topo_conv_type == 'CFCNN' else
                GraphConv(in_feats=hidden_net_feats, out_feats=out_node_feats),
            'connect':
                GraphConv(in_feats=hidden_hanna_feats, out_feats=hidden_hanna_feats),
                # NNConv(in_feats=hidden_node_feats, out_feats=out_node_feats,
                #        edge_func=geom_edge_func) if route_conv_type == 'MPNN' else
                # SAGEConv(in_feats=hidden_node_feats, out_feats=out_node_feats,
                #          aggregator_type='pool') if route_conv_type == 'SAGE' else
                # CFConv(node_in_feats=hidden_node_feats, edge_in_feats=hidden_edge_feats,
                #        hidden_feats=hidden_node_feats, out_feats=out_node_feats) if route_conv_type == 'CFCNN' else
                # GATConv(in_feats=hidden_node_feats, out_feats=out_node_feats, num_heads=1),
            'point-to': 
                GraphConv(in_feats=hidden_node_feats, out_feats=hidden_hanna_feats),
            'point-from': 
                NNConv(in_feats=hidden_hanna_feats, out_feats=hidden_node_feats,
                       edge_func=topo_edge_func) if grid_conv_type == 'MPNN' else
                SAGEConv(in_feats=(hidden_hanna_feats, hidden_node_feats), out_feats=hidden_node_feats,
                         aggregator_type='pool') if grid_conv_type == 'SAGE' else
                CFConv(node_in_feats=hidden_hanna_feats, edge_in_feats=hidden_pin_feats,
                       hidden_feats=hidden_node_feats, out_feats=hidden_node_feats) if grid_conv_type == 'CFCNN' else
                GraphConv(in_feats=hidden_hanna_feats, out_feats=hidden_node_feats),
        }, aggregate=agg_type)

    def forward(self, g: dgl.DGLHeteroGraph, node_feat: torch.Tensor, net_feat: torch.Tensor,
                pin_feat: torch.Tensor, hanna_feat: torch.Tensor, edge_feat: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = {
            'cell': node_feat,
            'net': net_feat,
            'gcell': hanna_feat
        }

        mod_kwargs = {}
        if self.topo_conv_type == 'MPNN':
            mod_kwargs['pinned'] = {'efeat': pin_feat}
        elif self.topo_conv_type == 'SAGE':
            mod_kwargs['pinned'] = {'edge_weight': torch.sigmoid(self.topo_weight(pin_feat))}
        elif self.topo_conv_type == 'CFCNN':
            mod_kwargs['pinned'] = {'edge_feats': pin_feat}
        if self.grid_conv_type == 'MPNN':
            mod_kwargs['point-from'] = {'efeat': edge_feat}
        elif self.grid_conv_type == 'SAGE':
            mod_kwargs['point-from'] = {'edge_weight': torch.sigmoid(self.grid_weight(edge_feat))}
        elif self.grid_conv_type == 'CFCNN':
            mod_kwargs['point-from'] = {'edge_feats': edge_feat}

        h1 = self.hetero_conv.forward(g, h, mod_kwargs=mod_kwargs)

        return h1['cell'], h1['net'] + self.net_lin(net_feat), h1['gcell']
#         return h1['node'], h1['net']


class RouteGNN(nn.Module):
    def __init__(self, in_node_feats: int, in_net_feats: int, in_pin_feats: int, in_hanna_feats: int, in_edge_feats: int,
                 n_target: int, config: Dict[str, Any],
                #  activation: str = 'sig',
                 topo_conv_type='CFCNN', grid_conv_type='SAGE', agg_type='max', cat_raw=True):
        super(RouteGNN, self).__init__()
        self.in_node_feats = in_node_feats
        self.in_net_feats = in_net_feats
        self.in_pin_feats = in_pin_feats
        self.in_hanna_feats = in_hanna_feats
        self.n_layer = config['N_LAYER']
        self.out_node_feats = config['NODE_FEATS']
        self.out_net_feats = config['NET_FEATS']
        self.hidden_node_feats = self.out_node_feats
        self.hidden_pin_feats = config['PIN_FEATS']
        self.hidden_hanna_feats = config['HANNA_FEATS']
        self.hidden_edge_feats = config['EDGE_FEATS']
        self.in_edge_feats = in_edge_feats
        self.hidden_net_feats = self.out_net_feats
        self.cat_raw = cat_raw

        self.node_lin = nn.Linear(self.in_node_feats, self.hidden_node_feats)
        self.net_lin = nn.Linear(self.in_net_feats, self.hidden_net_feats)
        self.pin_lin = nn.Linear(self.in_pin_feats, self.hidden_pin_feats)
        self.hanna_lin = nn.Linear(self.in_hanna_feats, self.hidden_hanna_feats)
        self.edge_lin = nn.Linear(self.in_edge_feats, self.hidden_edge_feats)
        self.list_node_net_gnn = nn.ModuleList(
            [NodeNetGNN(self.hidden_node_feats, self.hidden_net_feats,
                        self.hidden_pin_feats, self.hidden_hanna_feats,self.hidden_edge_feats,
                        self.out_node_feats, self.out_net_feats,
                        topo_conv_type, grid_conv_type, agg_type) for _ in range(self.n_layer)])
        self.n_target = n_target
        if cat_raw:
            self.output_layer_1 = nn.Linear(self.in_node_feats + self.hidden_node_feats, self.hidden_node_feats)
            self.output_layer_net_1 = nn.Linear(self.in_net_feats + self.hidden_net_feats, self.hidden_net_feats)
        else:
            self.output_layer_1 = nn.Linear(self.hidden_node_feats, self.hidden_node_feats)
            self.output_layer_net_1 = nn.Linear(self.hidden_net_feats, self.hidden_net_feats)
        self.output_layer_2 = nn.Linear(self.hidden_node_feats, self.hidden_node_feats)
        self.output_layer_3 = nn.Linear(self.hidden_node_feats, self.n_target)
        self.output_layer_net_2 = nn.Linear(self.hidden_net_feats, self.hidden_net_feats)
        self.output_layer_net_3 = nn.Linear(self.hidden_net_feats, 1)
        self.output_layer_net_x1 = nn.Linear(self.in_net_feats, 64)
        self.output_layer_net_x2 = nn.Linear(64, 64)
        self.output_layer_net_x3 = nn.Linear(64, 1)
        # self.activation = activation
        self.net_readout_params = [
            self.output_layer_net_1, self.output_layer_net_2, self.output_layer_net_3,
            self.output_layer_net_x1, self.output_layer_net_x2, self.output_layer_net_x3,
        ]

    def forward(self, in_node_feat: torch.Tensor, in_net_feat: torch.Tensor,
                in_pin_feat: torch.Tensor, in_hanna_feat: torch.Tensor,
                node_net_graph: dgl.DGLHeteroGraph = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_net_feat = torch.log10(in_net_feat + 1e-4)
        in_hanna_feat = torch.log10(in_hanna_feat + 1e-4)
        in_node_feat[:,-2:] = torch.log10(in_node_feat[:,-2:] + 1e-4)
        in_node_feat.requires_grad=True
        in_edge_feat=node_net_graph.edges['point-from'].data['feats']
        node_feat = F.leaky_relu(self.node_lin(in_node_feat))
        net_feat0 = net_feat = F.leaky_relu(self.net_lin(in_net_feat))
        pin_feat = F.leaky_relu(self.pin_lin(in_pin_feat))
        hanna_feat = F.leaky_relu(self.hanna_lin(in_hanna_feat))
        edge_feat = F.leaky_relu(self.edge_lin(in_edge_feat))

        for i in range(self.n_layer):
            node_feat, net_feat, hanna_feat = self.list_node_net_gnn[i].forward(
                node_net_graph, node_feat, net_feat, pin_feat, hanna_feat, edge_feat)
            node_feat, net_feat = F.leaky_relu(node_feat), F.leaky_relu(net_feat)

        if self.cat_raw:
            node_feat = torch.cat([in_node_feat, node_feat], dim=-1)
            net_feat = torch.cat([in_net_feat, net_feat], dim=-1)
        output_predictions = self.output_layer_3(F.leaky_relu(
            self.output_layer_2(F.leaky_relu(
                self.output_layer_1(node_feat)
            ))
        ))
        net_feat1 = net_feat0 + F.relu(self.output_layer_net_1(net_feat))
        net_feat2 = net_feat1 + F.relu(self.output_layer_net_2(net_feat1))
        net_feat3 = self.output_layer_net_3(net_feat2)
        net_feat_x1 = self.output_layer_net_x1(in_net_feat)
        net_feat_x2 = self.output_layer_net_x2(F.relu(net_feat_x1))
        output_net_predictions = self.output_layer_net_x3(F.relu(net_feat_x2)) + F.tanh(net_feat3)
#         if self.activation == 'sig':
#             output_predictions = torch.sigmoid(output_predictions)
# #             output_net_predictions = torch.sigmoid(output_net_predictions)
#         elif self.activation == 'tanh':
#             output_predictions = torch.tanh(output_predictions)
# #             output_net_predictions = torch.tanh(output_net_predictions)
#         else:
#             assert False, f'Undefined activation {self.activation}'
        output_predictions = torch.relu(output_predictions)
        return output_predictions, output_net_predictions

if __name__ =='__main__':
    num_cell = 6
    num_net = 3
    num_hanna = 17
    num_pin = 8
    with open("../test/test_netlist_graph.json","r") as f:
        netlist_edge = json.load(f)
    netlist_us,netlist_vs = [],[]
    for net,list_node in enumerate(netlist_edge):
        for node in list_node:
            netlist_us.append(net)
            netlist_vs.append(node)
    with open("../test/test_hanna_graph.json","r") as f:
        hanna_edge = json.load(f)
    hanna_edge_us,hanna_edge_vs = [],[]
    for u,v in hanna_edge:
        hanna_edge_us.append(u)
        hanna_edge_vs.append(v)
    
    with open("../test/test_cell2hanna_graph.json","r") as f:
        cell2hanna_edge = json.load(f)
    cell2hanna_edge_us,cell2hanna_edge_vs = [],[]
    for u,v in cell2hanna_edge:
        cell2hanna_edge_us.append(u)
        cell2hanna_edge_vs.append(v)
    
    route_graph = dgl.heterograph({
        ('cell','pins','net'):(netlist_vs,netlist_us),
        ('net','pinned','cell'):(netlist_us,netlist_vs),
        ('cell','point-to','hanna'):(cell2hanna_edge_us,cell2hanna_edge_vs),
        ('hanna','point-from','cell'):(cell2hanna_edge_vs,cell2hanna_edge_us),
        ('hanna','connect','hanna'):(hanna_edge_us,hanna_edge_vs)
    },num_nodes_dict={'cell':num_cell,'net':num_net,"hanna":num_hanna})#
    route_graph.nodes['cell'].data['hv'] = torch.abs(torch.randn([num_cell,7]))
    route_graph.nodes['cell'].data['pos'] = torch.randn([num_cell,2])
    route_graph.nodes['net'].data['hv'] = torch.abs(torch.randn([num_net,7]))
    route_graph.nodes['net'].data['degree'] = torch.randn([num_net,1])
    route_graph.nodes['hanna'].data['hv'] = torch.abs(torch.randn([num_hanna,8]))

    route_graph.edges['pinned'].data['feats'] = torch.randn([num_pin,3])
    config = {
        'N_LAYER':2,
        'NODE_FEATS':32,
        'NET_FEATS':32,
        'PIN_FEATS':16,
        'HANNA_FEATS':32,
    }
    model = RouteGNN(
        in_node_feats = 7,
        in_net_feats = 7,
        in_pin_feats = 3,
        in_hanna_feats = 8,
        n_target = 1,
        config = config
    )
    with open("../data/graph.pickle","rb") as f:
        list_tuple_graph = pickle.load(f)
    for hetero_graph,route_graph in tqdm.tqdm(list_tuple_graph):
        # hetero_graph = list_tuple_graph[0][i]
        # route_graph = list_tuple_graph[1][i]
        y1,y2 = model.forward(in_node_feat=route_graph.nodes['cell'].data['hv'],
                            in_net_feat=route_graph.nodes['net'].data['hv'],
                            in_pin_feat=route_graph.edges['pinned'].data['feats'],
                            in_hanna_feat=route_graph.nodes['hanna'].data['hv'],
                            node_net_graph=route_graph)
        assert y1.size(0) == hetero_graph.nodes['cell'].data['label'].size(0),f"cell size error {y1.size()} {hetero_graph.nodes['cell'].data['label'].size()}"
        assert y2.size(0) == hetero_graph.nodes['net'].data['label'].size(0),f"net size error {y2.size()} {hetero_graph.nodes['net'].data['label'].size()}"


