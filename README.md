# Warehouse routing problem solved with q-learning

This repository contains my code for the assignment of the course AE4350 Bio-Inspired Intelligence and Learning for 
Aerospace Applications. It demonstrates how to solve a warehouse routing problem (related to the TSP or VRP) using 
q-learning (a reinforcement learning algorithm). Next to the basic version of the problem, I have also added capacitated 
and multi-agent versions. I have also implemented it in a parallel way (using multiple cores) with 
[Ray](https://github.com/ray-project/ray). More information about the theory and the results can be found in the report.

## Installation
Install using `pipenv` with Python 3.8.

## Usage
- [warehouse.py](warehouse.py): contains the basic `Warehouse` class and the basic `train` function
- [warehouse_multiagent.py](warehouse_multiagent.py) and [warehouse_capacitated.py](warehouse_capacitated.py): contain 
the multiagent and capacitated versions of the `Warehouse` class, respectively. The training process is not implemented 
as a separate function. As these two files now contain a lot of duplicate code, they should in the future be combined. 
Ideally, there would be only one `Warehouse` class and one `train` function containing all features.
- [warehouse_parallel.py](warehouse_parallel.py): contains the `Main` and `Worker` classes and the `train_parallel` function
of the parallel q-learning implementation. `Warehouse` from `warehouse.py` is reused.
- [sensitivity_analyses](sensitivity_analyses): contains scripts for sensitivity analyses
- [other_results](other_results): contains scripts for other results, for instance the comparison between the 
sequential and the parallel algorithm
- [tries](tries): contains script that were used to try out various programming aspects
