import sys
import os
sys.path.append("./thirdparty/RouteGraph")
sys.path.append(os.path.join(os.path.abspath("."),"build"))
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import numpy as np
import logging

import dreamplace.configure as configure
import Params
import PlaceDB
PARAM_PATH = 'test/'
DATASET_NAME = "dac2012"
RESULTS_DIR = 'results'

logging.root.name = 'DREAMPlace'
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)-7s] %(name)s - %(message)s',
                    stream=sys.stdout)
netlist_name = 'superblue19'
oinfo_path = '/home/xuyanyang/RL/hyb/DREAMPlace/oinfo/superblue19'



param_path = os.path.join(PARAM_PATH, DATASET_NAME, f"{netlist_name}.json")
params = Params.Params()
params.load(param_path)

placedb = PlaceDB.PlaceDB()
placedb(params)

oinfo_list = os.listdir(oinfo_path)
for oinfo_file_name in oinfo_list:
    oinfo_file_path = os.path.join(oinfo_path,oinfo_file_name)
    congestion_map = torch.zeros((placedb.num_routing_grids_x,
                                        placedb.num_routing_grids_y,
                                        placedb.num_routing_layers),
                                        )


    wirelength = 0
    total_overflow = 0
    max_overflow = 0
    num_via = 0
    with open(oinfo_file_path, "r") as f:
        status = 0
        for line in f:
            line = line.strip()

            if line.startswith("Total wirelength"):
                wirelength=int(line.split("=")[-1])
            if line.startswith("Total overflow"):
                total_overflow=int(line.split("=")[-1])
            if line.startswith("Max overflow"):
                max_overflow=int(line.split("=")[-1])
            if line.startswith("Via count"):
                num_via=int(line.split("=")[-1])

            if line.startswith("Overflowed grid edges :"):
                status = 1
            elif line.startswith("end") and status:
                status = 0
                break
            elif line.startswith("(") and status:
                tokens = line.split()
                start = (int(tokens[0][1:-1]), int(tokens[1][:-1]),
                            int(tokens[2][:-1]))
                end = (int(tokens[4][1:-1]), int(tokens[5][:-1]),
                        int(tokens[6][:-1]))
                overflow = int(tokens[7])
                assert start[2] == end[2]
                congestion_map[start[0], start[1], start[2]] = overflow


    horizontal_routing_capacities = torch.from_numpy(
                    placedb.unit_horizontal_capacities *
                    placedb.routing_grid_size_y)
    vertical_routing_capacities=torch.from_numpy(
                placedb.unit_vertical_capacities *
                placedb.routing_grid_size_x)
    routing_capacities = (horizontal_routing_capacities + 
                                    vertical_routing_capacities).view([
                                        1, 1,
                                        len(horizontal_routing_capacities)
                                    ])

    overflow_map = congestion_map / (routing_capacities + 1e-6) + 1

    horizontal_overflow_map = overflow_map[:, :, 0:placedb.num_routing_layers:2].sum(dim=2)
    horizontal_overflow_map /= horizontal_routing_capacities.sum()
    vertical_overflow_map = overflow_map[:, :, 1:placedb.num_routing_layers:2].sum(dim=2)
    vertical_overflow_map /= vertical_routing_capacities.sum()
    ret = torch.max(horizontal_overflow_map, vertical_overflow_map)

    print(f"--------{oinfo_file_name}------------")
    print(f"\t\t total_overflow:{total_overflow}")
    print(f"\t\t max_overflow:{max_overflow}")
    print(f"\t\t max_h_gr:{float(torch.max(horizontal_overflow_map))}")
    print(f"\t\t max_v_gr:{float(torch.max(vertical_overflow_map))}")
    print(f"\t\t num_via:{num_via}")
    print(f"\t\t wirelength:{wirelength}")


