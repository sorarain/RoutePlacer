import sys
import os
sys.path.append("./thirdparty/RouteGraph")
sys.path.append(os.path.join(os.path.abspath("."),"build"))
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import numpy as np
import dgl
import time
import tqdm
import logging
import json

import dreamplace.configure as configure
import Params
import PlaceDB
import NonLinearPlace
import PlaceObj
import Timer
from dreamplace.Args import get_args
from dreamplace.CongestionPredictor import CongestionPredictor
PARAM_PATH = 'test/'
DATASET_PATH = "/root/DREAMPlace/benchmarks"
DATASET_NAME = ""
RESULTS_DIR = 'results'

logging.root.name = 'DREAMPlace'
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)-7s] %(name)s - %(message)s',
                    stream=sys.stdout)

args = get_args()
if "dac" in args.name:
    DATASET_PATH += "/dac2012"
    DATASET_NAME = "dac2012"
elif "ispd" in args.name:
    DATASET_PATH += "/ispd2011"
    DATASET_NAME = "ispd2011"

routability_opt_flag = 1
if "^" in args.hashcode:
    routability_opt_flag = 0
    args.hashcode = args.hashcode.replace("^","")

if "/" not in args.hashcode:
    netlist_name = args.hashcode
else:
    netlist_name = args.hashcode.split("/")[0]

param_path = os.path.join(PARAM_PATH, DATASET_NAME, f"{netlist_name}.json")
params = Params.Params()
params.load(param_path)

suffix = ''
params.__dict__["timing_opt_flag"] = 0
# params.__dict__["our_route_opt"] = 1
# params.__dict__["congestion_weight"] = args.congestion_weight
params.__dict__["routability_opt_flag"] = routability_opt_flag
# params.__dict__['max_num_area_adjust'] = 4
if "/" in args.hashcode:
    param_list = args.hashcode.split("/")[1]
    params.__dict__['max_num_area_adjust'] = int(param_list.split(",")[0])
    params.__dict__['route_opt_adjust_exponent'] = float(param_list.split(",")[1])

params.args = args
placedb = PlaceDB.PlaceDB()
placedb(params)
path = "/root/DREAMPlace/%s/%s/%s" % (params.result_dir, args.name, params.design_name())
if not os.path.exists(path):
    os.system("mkdir -p %s" % (path))
if "/" in args.hashcode:
    gp_out_file = os.path.join(
        path,
        "%s_%s.gp.%s" % (params.design_name(), '_'+args.hashcode.split("/")[1].replace(',','_'), params.solution_file_suffix()))
else:
    gp_out_file = os.path.join(
        path,
        "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
if not os.path.exists(gp_out_file):
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    metrics = placer(params, placedb)
    placedb.write(params, gp_out_file)


cmd = ""
cmd += "cd /root/DREAMPlace/thirdparty/NCTUgr.ICCAD2012 && "
aux_path = os.path.join(DATASET_PATH,netlist_name,netlist_name + ".aux")
if "superblue15" in netlist_name:
    aux_path = aux_path.replace("superblue15.aux","superblue15_1.aux")
cmd += f"./NCTUgr ICCAD {aux_path} {gp_out_file} ./DAC12.set {gp_out_file.replace('.gp.pl','.gr')}"
if not os.path.exists(gp_out_file.replace(".gp.pl",".gr.ofinfo")):
    print(cmd)
    os.system(cmd)
    os.system(f"rm -f {gp_out_file.replace('.gp.pl','.gr')}")

congestion_map = torch.zeros((placedb.num_routing_grids_x,
                                      placedb.num_routing_grids_y,
                                      placedb.num_routing_layers),
                                     )
with open(gp_out_file.replace(".gp.pl",".gr.ofinfo"), "r") as f:
    status = 0
    for line in f:
        line = line.strip()
        if line.startswith("Total wirelength"):
            wirelength=int(line.split("=")[-1])
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

horizontal_routing_capacities=torch.from_numpy(
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
# ret = torch.max(horizontal_overflow_map, vertical_overflow_map)
# ret = overflow_map.max(dim=2)[0]

result = {
    "netlist":netlist_name,
    "total_overflow":int(torch.sum(congestion_map)),
    "max_overflow":int(torch.max(congestion_map)),
    "H-CR":round(float(torch.max(horizontal_overflow_map)),2),
    "V-CR":round(float(torch.max(vertical_overflow_map)),2),
    "wirelength": int(wirelength)
}
print(result)
with open(f"./results/{args.name}/{netlist_name}.json","w") as f:
    json.dump(result,f)

