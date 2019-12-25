#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p thinnodes
#SBATCH -t 2:00:00

module load python/3.6.8

firedecomp_simulations -n 100 -nb 4 -na 2 -nm 2 -np 15 -m fix_work -so ../solver_options.yaml -vvv &> simulations.log
