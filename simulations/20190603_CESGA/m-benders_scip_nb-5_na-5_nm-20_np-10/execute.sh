#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p thinnodes
#SBATCH -t 20:00:00

firedecomp_simulations -n 100 -nb 5 -na 5 -nm 20 -np 10 -m benders_scip -so ../solver_options.yaml -vvv &> simulations.log
