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
import pandas as pd
PARAM_PATH = 'test/ours'
DATASET_NAME = "dac2012"
RESULTS_DIR = 'results'

logging.root.name = 'DREAMPlace'
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)-7s] %(name)s - %(message)s',
                    stream=sys.stdout)

netlist_names = [
    # 'superblue1',
    # 'superblue2',
    # 'superblue3',
    # 'superblue4',
    'superblue5',
    'superblue6',
    'superblue7',
    # 'superblue9',# wrong kreorder
    # 'superblue10',#fail
    'superblue11',
    'superblue12',
    'superblue14',
    # # 'superblue15',#can't read
    'superblue16',
    'superblue18',
    'superblue19',
]

args = get_args()

if "dac" in args.name:
    DATASET_NAME = "dac2012"
    netlist_names = [
    'superblue2',
    'superblue3',
    'superblue6',
    'superblue7',
    'superblue11',
    'superblue12',
    'superblue14',
    'superblue16',
    'superblue19',
    ]
elif "ispd" in args.name:
    DATASET_NAME = "ispd2011"
    netlist_names = [
    'superblue1',
    'superblue2',
    'superblue4',
    'superblue5',
    'superblue10',#fail
    'superblue12',
    'superblue15',#can't read
    'superblue18',
]

result = {}

for netlist_name in netlist_names:
    param_path = os.path.join(PARAM_PATH, DATASET_NAME, f"{netlist_name}.json")
    params = Params.Params()
    params.load(param_path)

    suffix = ''
    params.__dict__["timing_opt_flag"] = 0
    params.__dict__["our_route_opt"] = 1
    # params.__dict__["congestion_weight"] = args.congestion_weight
    params.__dict__["routability_opt_flag"] = 0
    params.__dict__['max_num_area_adjust'] = 4
    params.args = args

    os.system(f"mkdir -p ./results/{args.name}")

    os.system(f"python dreamplace/run_ours.py --name {args.name} --model {args.model} --hashcode {netlist_name} > ./results/{args.name}/{netlist_name}.log")
    if not os.path.exists(f"./results/{args.name}/{netlist_name}.json"):
        continue
    with open(f"./results/{args.name}/{netlist_name}.json","r") as f:
        tmp_result = json.load(f)
    for key in tmp_result.keys():
        result.setdefault(key,[]).append(tmp_result[key])

df = pd.DataFrame()
for key in result.keys():
    df[key] = result[key]

df.to_excel(f"./results/{args.name}/{args.name}.xlsx",index=False)



