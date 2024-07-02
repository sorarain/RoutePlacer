import os
from typing import List
import torch
import dgl
import pickle as pkl
import glob
import tqdm

from data.load_data_optimize import load_data,build_grid_graph

def scale_label(label):
    return torch.log(label + 1)
    # label[label>1] = torch.log(label[label>1]) + 1
    # return label


def preprocess_netlists(netlists_dir, netlists_names: List[str], args):

    directory = "./data/processed_collect"

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
   

    for netlist_name in netlists_names:
        directory = f"./data/processed_collect/{netlist_name}"

        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory + "/graphs")
            os.makedirs(directory + "/labels")
            os.makedirs(directory + "/density")
            print(f"Directory '{directory}' created successfully.")
        
        tuple_graph, batch_input = load_data(os.path.join(netlists_dir, netlist_name), args)
        hetero_graph, list_hetero_graph = tuple_graph
        for idx_batch_input, input_dict in enumerate(batch_input):
            print(f"processing batch {idx_batch_input}")
            
            h_net_density_grid = input_dict['h_net_density_grid']
            v_net_density_grid = input_dict['v_net_density_grid']
            
            filename_graph = directory + "/graphs/" + netlist_name + "_" + str(idx_batch_input) + "_graph.pkl"
            filename_label = directory + "/labels/" + netlist_name + "_" + str(idx_batch_input)  + "_label" + ".pkl"
            filename_density = directory + "/density/" + netlist_name + "_" + str(idx_batch_input) + "_density" + ".pkl"
            
            graphs = []
            labels = []
            densities = []
            
            for sub_hetero_graph in tqdm.tqdm((list_hetero_graph)):
                if not os.path.exists(filename_graph):
                    sub_hetero_graph.nodes['cell'].data['pos'] = input_dict['pos'][sub_hetero_graph.nodes['cell'].data[dgl.NID]]
                    sub_hetero_graph.nodes['cell'].data['hv'] = input_dict['hv'][sub_hetero_graph.nodes['cell'].data[dgl.NID]]
                    sub_hetero_graph.nodes['cell'].data['label'] = input_dict['label'][sub_hetero_graph.nodes['cell'].data[dgl.NID]]
                    sub_hetero_graph.nodes['net'].data['hv'] = input_dict['net_hv'][sub_hetero_graph.nodes['net'].data[dgl.NID]]
                    sub_node_pos = input_dict['pos'][sub_hetero_graph.nodes['cell'].data[dgl.NID]]
                    sub_route_graph = build_grid_graph(sub_hetero_graph,sub_node_pos,
                            h_net_density_grid,
                            v_net_density_grid,
                            input_dict['bin_x'],input_dict['bin_y'])
                    graphs.append(sub_route_graph)
                if not os.path.exists(filename_label):
                    scaled_label = scale_label(input_dict['label'][sub_hetero_graph.nodes['cell'].data[dgl.NID]])
                    labels.append(scaled_label)
                if not os.path.exists(filename_density):
                    density = torch.from_numpy(input_dict['node_density_grid'])[sub_hetero_graph.nodes['cell'].data[dgl.NID]]
                    densities.append(density)
                    
            if not os.path.exists(filename_graph):
                with open(filename_graph, "wb") as f:
                    pkl.dump(graphs, f)
                print("route graph file saved successfully as", filename_graph)
            if not os.path.exists(filename_label):
                with open(filename_label, "wb") as f:
                    pkl.dump(labels, f)
                print("label file saved successfully as", filename_label)
            if not os.path.exists(filename_density):
                with open(filename_density, "wb") as f:
                    pkl.dump(densities, f)
                print("density saved successfully as", filename_density)


class RouteGraphDataset():
    def __init__(self, netlists_names: List[str], dataset_path: str = "./data/processed_collect/") -> None:
        super().__init__()
        self.graph_paths = []
        self.label_paths = []
        self.density_paths = []
        for netlist_name in netlists_names:
            netlist_graph_paths = glob.glob(f"{dataset_path}{netlist_name}/graphs/*")
            netlist_label_paths = glob.glob(f"{dataset_path}{netlist_name}/labels/*")
            netlist_density_paths = glob.glob(f"{dataset_path}{netlist_name}/density/*")
            assert len(netlist_density_paths) == len(netlist_label_paths) == len(netlist_graph_paths)
            self.graph_paths.extend(netlist_graph_paths)
            self.label_paths.extend(netlist_label_paths)
            self.density_paths.extend(netlist_density_paths)

    def __len__(self):
        return len(self.graph_paths)
    
    def __iter__(self):
        for idx in range(len(self)):
            assert os.path.exists(self.graph_paths[idx])
            assert os.path.exists(self.label_paths[idx])
            assert os.path.exists(self.density_paths[idx])
            with open(self.graph_paths[idx], "rb") as f:
                route_graphs = pkl.load(f)
            with open(self.label_paths[idx], "rb") as f:
                labels = pkl.load(f)
            with open(self.density_paths[idx], "rb") as f:
                densities = pkl.load(f)
            assert len(route_graphs) == len(labels) == len(densities)
            for i in range(len(route_graphs)):
                yield route_graphs[i], labels[i], densities[i]