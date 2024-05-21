import os
import sys
sys.path.append(os.path.join(os.path.abspath('.'),'build'))
sys.path.append(os.path.abspath('.'))

import dgl
import networkx as nx
import json
import numpy as np
import torch
import math

def create_input_graph_file(graph,
                            input_file_name:str):
    num_nets = graph.num_nodes('net')
    num_nodes = graph.num_nodes('cell')

    output_str = f"{num_nets} {num_nodes}\n"

    lines = [[] for _ in range(num_nets)]
    cells,nets = graph.edges(etype='pins')

    for cell,net in zip(cells,nets):
        lines[net].append(int(cell) + 1)
    
    for line in lines:
        for i,cell in enumerate(line):
            output_str += str(int(cell))
            if i != len(line) - 1:
                output_str += ' '
            else:
                output_str += '\n'
    
    with open(input_file_name,'w') as f:
        f.write("".join(output_str))

def create_group_graph(
    graph:dgl.DGLHeteroGraph,
    belong,
    blocks: int
):
    num_nets = graph.num_nodes('net')
    num_nodes = graph.num_nodes('cell')
    cells,nets = graph.edges(etype='pins')

    nets_id = num_nodes + 3
    net_have_nodes = [{} for _ in range(num_nets)]
    group_graph_edges = [[] for _ in range(blocks)]

    for cell,net in zip(cells,nets):
        cell_belong = int(belong[int(cell)])
        if cell_belong == -1:
            continue
        net_have_nodes[net].setdefault(cell_belong,[])
        net_have_nodes[net][cell_belong].append(int(cell))
    
    for net in range(num_nets):
        for group in net_have_nodes[net]:
            for cell in net_have_nodes[net][group]:
                cell_belong = int(belong[int(cell)])
                if cell_belong == -1:
                    continue
                group_graph_edges[cell_belong].append((int(cell),nets_id))
                group_graph_edges[cell_belong].append((nets_id,int(cell)))
            nets_id += 1
    return group_graph_edges

def create_cluster_json(
    graph:dgl.DGLHeteroGraph,
    input_filename:str,
    output_filename:str,
    blocks: int):

    num_nets = graph.num_nodes('net')
    num_nodes = graph.num_nodes('cell')
    block_lists = []
    belong = np.ones(num_nodes) * (-1)

    with open(input_filename,'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        belong[i] = int(line)
    
    batch_graph_edges = create_group_graph(graph,belong,blocks)

    for i,group_edges in enumerate(batch_graph_edges):
        group_graph = nx.Graph(list(group_edges))
        connect_part_set_list = nx.connected_components(group_graph)
        connect_part_lists = [list(node_set) for node_set in connect_part_set_list]
        for connect_part in connect_part_lists:
            tmp_result = []
            for cell in connect_part:
                if cell > num_nodes:
                    continue
                tmp_result.append(int(cell))
            block_lists.append(tmp_result)

    print(f"total blocks {len(block_lists)}")

    json_data = json.dumps(block_lists)
    with open(output_filename,"w") as f:
        f.write(json_data)
    return block_lists

def partition_graph(graph,netlist_name,
                    output_dir:str = os.path.abspath('.'),
                    keep_cluster_file:bool = True,
                    use_kahypar="mt_strong"):
    num_cells = graph.num_nodes(ntype='cell')
    blocks = int(np.ceil(num_cells / 10000))
    input_graph_file_name = os.path.join(output_dir,f'{netlist_name}_graph.input')
    if not os.path.exists(input_graph_file_name):
        create_input_graph_file(graph,input_graph_file_name)
    cmd_path = "/root/mt-kahypar/build/mt-kahypar/application"
    preset_type = "default"#"quality"#"default_flows"
    if use_kahypar == 'mt_strong':
        cmd = f"{cmd_path}/MtKaHyPar -h {input_graph_file_name} --preset-type={preset_type}  --instance-type=hypergraph -t 64 -k {blocks} -e 0.03 -o km1 -m direct --write-partition-file=true"
    else:
        assert 1==0,"error in partition"
    # elif use_kahypar == 'mt_fast':
    #     cmd = f"{cmd_path}/MtKaHyParFast -h {input_graph_file_name} -k {blocks} -e 0.03 -o km1 -m direct -p ./thirdparty/mt-kahypar/config/fast_preset.ini -t 24"
    grouping_filename = os.path.join(output_dir,f"{netlist_name}_graph.input.part{blocks}.epsilon0.03.seed0.KaHyPar")
    if not os.path.exists(grouping_filename):
        os.system(cmd)
    cluster_json_filename = os.path.join(output_dir,f"{netlist_name}_cell_clusters.json")
    if not os.path.exists(cluster_json_filename):
        partition_list = create_cluster_json(graph,grouping_filename,cluster_json_filename,blocks)
    else:
        with open(cluster_json_filename,"r") as f:
            partition_list = json.load(f)
    partition_list = np.array(partition_list, dtype=object)
    if not keep_cluster_file:
        os.remove(grouping_filename)
        os.remove(input_graph_file_name)
    return partition_list