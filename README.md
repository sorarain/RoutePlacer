## Data Collection

To collect data to train RouteGNN, we first need to collect placement result in different epochs when running placer and save them as the structure below

~~~
data---|---collect---netlist_name--|---1---|---placement results
                                   |
                                   |---2---|---placement results
                                   |
                                   |---epochs---|---placement results
                                   ...
                                   |
                                   |---2000---|---placement results
~~~

The placement results include the file as below

* hdm.npy vdm.npy: initial horizon/vertical utilization map
* iter\_\{epochs\}\_bad_cmap_h.npy iter\_\{epochs\}\_bad_cmap_v.npy: horizon/vertical rudy map
* iter\_\{epochs\}_cmap_h.npy iter\_\{epochs\}\_cmap_v.npy: horizon/vertical overflow map
* iter\_\{epochs\}_pos.npy: all cell position, shape:[num_movable_cells, 2]
* iter\_\{epochs\}_x.npy, iter\_\{epochs\}\_y.npy: movable cell x-position/y-position
*  iter_fix_hori_cap.npy iter_fix_verti_cap.npy: horizon/vertical capacity for each routing grid
* iter_fix_mapper.npy: map each cell to its original cell in circuits
* iter_fix_nctu_cap.npy: capacities for NCTUgr
* iter_fix_pd.npy: pin direction
* iter_fix_pin2edge.npy: map each pin to its connecting net
* iter_fix_pin2node.npy: map each pin to its cell
* iter_fix_po_x.npy iter_fix_po_y.npy: pin offset for each pins
* iter_fix_sizes_x.npy iter_fix_sizes_y.npy: width/height of each cells

After collecting the data, run the command below to process the original data for training in the "RouteGraph/data" path

~~~
python script_process.py
~~~

In script_process.py, you can appoint to process which netlist in "netlist_names" variable.

## Train

Run the command to train model in the "RouteGraph" path

~~~
python script_train_congestion_optimize.py --name {your_model_name}
~~~

The model parameter will store in param/{your_model_name}.pkl

## Run RoutePlacer

### Use DREAMPlace Framework

Clone [DREAMPlace](https://github.com/limbo018/DREAMPlace) repository. Copy the file in RoutePlacer repository to DREAMPlace and replace  the file.

Complie DREAMPlace following [DREAMPlace README](https://github.com/limbo018/DREAMPlace/blob/master/README.md) 

### Run Benchmark

Copy DAC2012 and ISPD2011 benchmark to "DREAMPlace/benchmark". The file structure is as below

~~~
DREAMPlace---|---benchmark---|---ISPD2011---|---netlist_name---|---circuit_files
                             |              |
                             |              |---netlist_name---|---circuit_files
                             |
                             |
						   |---DAC2012----|---netlist_name---|---circuit_files
						   			     |
						   			     |
						   			     |---netlist_name---|---circuit_files
~~~

circuit files include the files below

~~~
netlist_name.aux
netlist_name.nets
netlist_name.nodes
netlist_name.pl
netlist_name.route
netlist_name.scl
netlist_name.shapes
netlist_name.wts
~~~

move trained model file into "DREAMPlace/param"

run command to get benchmark

~~~
python script_run_ours.py --name {dac12/ispd11} --model {model_name}
~~~

the result will store in "DREAMPlace/results"