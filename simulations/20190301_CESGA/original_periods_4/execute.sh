#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p thinnodes
#SBATCH -t 33:00:00

firedecomp_simulations -n 25 -nb 5 20 -na 5 20 -nm 5 20 -np 20 -m benders -so ../solver_options.yaml -vvv &> simulations.log
