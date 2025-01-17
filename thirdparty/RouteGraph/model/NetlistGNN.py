import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import NNConv, SAGEConv, GATConv, HeteroGraphConv, GraphConv, CFConv
from typing import Tuple, Dict, Any, List, Union


class NodeNetGNN(nn.Module):
    def __init__(self, hidden_node_feats: int, hidden_net_feats: int, hidden_pin_feats: int, hidden_edge_feats: int,
                 out_node_feats: int, out_net_feats: int, topo_conv_type, geom_conv_type, agg_type):
        super(NodeNetGNN, self).__init__()
        assert topo_conv_type in ['MPNN', 'SAGE', 'CFCNN', 'GCN'], f'{topo_conv_type} not in MPNN/SAGE/CFCNN/GCN'
        assert geom_conv_type in ['MPNN', 'SAGE', 'CFCNN', 'GCN'], f'{geom_conv_type} not in MPNN/SAGE/CFCNN/GCN'
        self.topo_conv_type = topo_conv_type
        self.geom_conv_type = geom_conv_type
        self.net_lin = nn.Linear(hidden_net_feats, hidden_net_feats)
        self.topo_lin = nn.Linear(hidden_pin_feats, hidden_net_feats * out_node_feats)
        self.geom_lin = nn.Linear(hidden_edge_feats, hidden_node_feats * out_node_feats)
        self.topo_weight = nn.Linear(hidden_pin_feats, 1)
        self.geom_weight = nn.Linear(hidden_edge_feats, 1)

        def topo_edge_func(efeat):
            return self.topo_lin(efeat)

        def geom_edge_func(efeat):
            return self.geom_lin(efeat)
        
        def my_agg_func(tensors, dsttype):
            new_tensors = []
            for tensor in tensors:
                if len(tensor.shape) == 3:
                    new_tensors.append(tensor[:, 0, :])
                else:
                    new_tensors.append(tensor)
            stacked = torch.stack(new_tensors, dim=0)
            return torch.max(stacked, dim=0)[0]

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
            'near':
                NNConv(in_feats=hidden_node_feats, out_feats=out_node_feats,
                       edge_func=geom_edge_func) if geom_conv_type == 'MPNN' else
                SAGEConv(in_feats=hidden_node_feats, out_feats=out_node_feats,
                         aggregator_type='pool') if geom_conv_type == 'SAGE' else
                CFConv(node_in_feats=hidden_node_feats, edge_in_feats=hidden_edge_feats,
                       hidden_feats=hidden_node_feats, out_feats=out_node_feats) if geom_conv_type == 'CFCNN' else
                GATConv(in_feats=hidden_node_feats, out_feats=out_node_feats, num_heads=1),
        }, aggregate=agg_type)

    def forward(self, g: dgl.DGLHeteroGraph, node_feat: torch.Tensor, net_feat: torch.Tensor,
                pin_feat: torch.Tensor, edge_feat: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = {
            'cell': node_feat,
            'net': net_feat,
        }

        mod_kwargs = {}
        if self.topo_conv_type == 'MPNN':
            mod_kwargs['pinned'] = {'efeat': pin_feat}
        elif self.topo_conv_type == 'SAGE':
            mod_kwargs['pinned'] = {'edge_weight': torch.sigmoid(self.topo_weight(pin_feat))}
        elif self.topo_conv_type == 'CFCNN':
            mod_kwargs['pinned'] = {'edge_feats': pin_feat}
        if self.geom_conv_type == 'MPNN':
            mod_kwargs['near'] = {'efeat': edge_feat}
        elif self.geom_conv_type == 'SAGE':
            mod_kwargs['near'] = {'edge_weight': torch.sigmoid(self.geom_weight(edge_feat))}
        elif self.geom_conv_type == 'CFCNN':
            mod_kwargs['near'] = {'edge_feats': edge_feat}

        h1 = self.hetero_conv.forward(g, h, mod_kwargs=mod_kwargs)

        return h1['cell'], h1['net'] + self.net_lin(net_feat)
#         return h1['node'], h1['net']


class NetlistGNN(nn.Module):
    def __init__(self, in_node_feats: int, in_net_feats: int, in_pin_feats: int, in_edge_feats: int,
                 n_target: int, config: Dict[str, Any],
                 activation: str = 'sig', recurrent=False,
                 topo_conv_type='CFCNN', geom_conv_type='SAGE', agg_type='max', cat_raw=True):
        super(NetlistGNN, self).__init__()
        self.recurrent = recurrent

        self.in_node_feats = in_node_feats
        self.in_net_feats = in_net_feats
        self.in_pin_feats = in_pin_feats
        self.in_edge_feats = in_edge_feats
        self.n_layer = config['N_LAYER']
        self.out_node_feats = config['NODE_FEATS']
        self.out_net_feats = config['NET_FEATS']
        self.hidden_node_feats = self.out_node_feats
        self.hidden_pin_feats = config['PIN_FEATS']
        self.hidden_edge_feats = config['EDGE_FEATS']
        self.hidden_net_feats = self.out_net_feats
        self.cat_raw = cat_raw

        self.node_lin = nn.Linear(self.in_node_feats, self.hidden_node_feats)
        self.net_lin = nn.Linear(self.in_net_feats, self.hidden_net_feats)
        self.pin_lin = nn.Linear(self.in_pin_feats, self.hidden_pin_feats)
        self.edge_lin = nn.Linear(self.in_edge_feats, self.hidden_edge_feats)
        if self.recurrent:
            self.node_net_gnn = NodeNetGNN(self.hidden_node_feats, self.hidden_net_feats,
                                           self.hidden_pin_feats, self.hidden_edge_feats,
                                           self.out_node_feats, self.out_net_feats,
                                           topo_conv_type, geom_conv_type, agg_type)
        else:
            self.list_node_net_gnn = nn.ModuleList(
                [NodeNetGNN(self.hidden_node_feats, self.hidden_net_feats,
                            self.hidden_pin_feats, self.hidden_edge_feats,
                            self.out_node_feats, self.out_net_feats,
                            topo_conv_type, geom_conv_type, agg_type) for _ in range(self.n_layer)])
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
        self.activation = activation
        self.net_readout_params = [
            self.output_layer_net_1, self.output_layer_net_2, self.output_layer_net_3,
            self.output_layer_net_x1, self.output_layer_net_x2, self.output_layer_net_x3,
        ]

    def forward(self, in_node_feat: torch.Tensor, in_net_feat: torch.Tensor,
                in_pin_feat: torch.Tensor, in_edge_feat: torch.Tensor,
                node_net_graph: dgl.DGLHeteroGraph = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_net_feat = torch.log10(in_net_feat + 1e-4)
        in_edge_feat = torch.log10(in_edge_feat + 1e-4)
        node_feat = F.leaky_relu(self.node_lin(in_node_feat))
        net_feat0 = net_feat = F.leaky_relu(self.net_lin(in_net_feat))
        pin_feat = F.leaky_relu(self.pin_lin(in_pin_feat))
        edge_feat = F.leaky_relu(self.edge_lin(in_edge_feat))

        for i in range(self.n_layer):
            if self.recurrent:
                node_feat, net_feat = self.node_net_gnn.forward(
                    node_net_graph, node_feat, net_feat, pin_feat, edge_feat)
            else:
                node_feat, net_feat = self.list_node_net_gnn[i].forward(
                    node_net_graph, node_feat, net_feat, pin_feat, edge_feat)
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
        if self.activation == 'sig':
            output_predictions = torch.sigmoid(output_predictions)
#             output_net_predictions = torch.sigmoid(output_net_predictions)
        elif self.activation == 'tanh':
            output_predictions = torch.tanh(output_predictions)
#             output_net_predictions = torch.tanh(output_net_predictions)
        else:
            assert False, f'Undefined activation {self.activation}'
        return output_predictions, output_net_predictions