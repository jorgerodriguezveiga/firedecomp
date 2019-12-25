#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p thinnodes
#SBATCH -t 20:00:00

module load python/3.6.8

firedecomp_simulations -n 100 -nb 5 -na 5 -nm 20 -np 10 -m fix_work -so ../solver_options.yaml -vvv &> simulations.log
