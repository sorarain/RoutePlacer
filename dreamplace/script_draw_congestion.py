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
import cairocffi as cairo

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

def draw_congestion(congestion_map,placedb,save_path):
    width = int(placedb.xh - placedb.xl)
    height = int(placedb.yh - placedb.yl)
    num_movable_nodes = placedb.num_movable_nodes
    num_physical_nodes = placedb.num_physical_nodes
    node_xl = placedb.node_x
    node_yl = placedb.node_y
    node_xh = node_xl + placedb.node_size_x[0:len(node_xl)]
    node_yh = node_yl + placedb.node_size_y[0:len(node_yl)]

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    # draw background
    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(placedb.xl, placedb.yl, width,
                          height)
    ctx.fill()

    def draw_rect(x1, y1, x2, y2):
                ctx.move_to(x1, y1)
                ctx.line_to(x1, y2)
                ctx.line_to(x2, y2)
                ctx.line_to(x2, y1)
                ctx.close_path()
                ctx.stroke()
    # draw fixed macros
    # ctx.set_source_rgba(1, 0, 0, alpha=0.5)
    ctx.set_source_rgba(0.8, 0.8, 0.8, alpha=0.5)
    for i in range(num_movable_nodes, num_physical_nodes):
        ctx.rectangle(node_xl[i], node_yl[i], node_xh[i] - node_xl[i],
                        node_yh[i] -
                        node_yl[i])  # Rectangle(xl, yl, w, h)
        ctx.fill()
    ctx.set_source_rgba(0, 0, 0, alpha=1.0)  # Solid color
    for i in range(num_movable_nodes, num_physical_nodes):
        draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i])
    
    ctx.set_source_rgba(0.8, 0.8, 0.8, alpha=0.5)  # Solid color
    for i in range(num_movable_nodes):
        ctx.rectangle(node_xl[i], node_yl[i], node_xh[i] - node_xl[i],
                        node_yh[i] -
                        node_yl[i])  # Rectangle(xl, yl, w, h)
        ctx.fill()
    
    colors = [
        (1.0, 1.0, 0.0),  # 浅黄色
        (1.0, 0.75, 0.0),  # 黄色
        (1.0, 0.5, 0.0),  # 橙色
        (1.0, 0.0, 0.0)   # 红色
    ]

    min_value = 1
    max_value = 20
    for i in range(placedb.num_routing_grids_x):
        for j in range(placedb.num_routing_grids_y):
            value = congestion_map[i,j]
            if value < 1:
                continue
            xl = placedb.routing_grid_xl + i * placedb.routing_grid_size_x
            yl = placedb.routing_grid_yl + j * placedb.routing_grid_size_y
            xh = xl + placedb.routing_grid_size_x
            yh = yl + placedb.routing_grid_size_y
            xl = max(xl,placedb.xl)
            yl = max(yl,placedb.yl)
            xh = min(xh,placedb.xh)
            yh = min(yh,placedb.yh)

            # ratio = (value - min_value) / (max_value - min_value)
            # green = 1 - ratio
            # #(1, green, 0)
            # ctx.set_source_rgba(1, green, 0, alpha=0.8)
            value = min(value,max_value)
            value = int((value - min_value) / 5)
            color = colors[value]
            ctx.set_source_rgb(*color)
            
            ctx.rectangle(xl,yl,xh - xl,yh - yl)
            ctx.fill()
    surface.write_to_png(save_path)




netlist_names = [
    # 'superblue1',
    # 'superblue2',
    # 'superblue3',
    # 'superblue4',
    # 'superblue5',
    # 'superblue6',
    # 'superblue7',
    # # 'superblue9',# wrong kreorder
    # # 'superblue10',#fail
    # 'superblue11',
    # 'superblue12',
    # 'superblue14',
    # # # 'superblue15',#can't read
    # 'superblue16',
    # 'superblue18',
    'superblue19',
]

args = get_args()

if "dac" in args.name:
    DATASET_NAME = "dac2012"
    netlist_names = [
    # 'superblue2',
    # 'superblue3',
    # 'superblue6',
    # 'superblue7',
    # 'superblue11',
    # 'superblue12',
    # 'superblue14',
    # 'superblue16',
    'superblue19',
    ]
elif "ispd" in args.name:
    DATASET_NAME = "ispd2011"
    netlist_names = [
    # 'superblue1',
    # 'superblue2',
    # 'superblue4',
    'superblue5',
    # 'superblue10',#fail
    # 'superblue12',
    # # 'superblue15',#can't read
    # 'superblue18',
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
    params.__dict__["routability_opt_flag"] = 1
    params.__dict__['max_num_area_adjust'] = 4
    params.args = args
    path = "/root/DREAMPlace/%s/%s/%s" % (params.result_dir + "/hist_map_pl", args.name, params.design_name())
    pl_path = os.path.join(path,params.design_name() + ".gp.pl")

    placedb = PlaceDB.PlaceDB()
    placedb(params)
    placedb.read_pl(params,pl_path)
    save_path = os.path.join(path,params.design_name()+"_congestion.png")
    congestion_map = np.load(os.path.join(path,params.design_name()+"_map.npy"))
    draw_congestion(congestion_map,placedb,save_path)

    

