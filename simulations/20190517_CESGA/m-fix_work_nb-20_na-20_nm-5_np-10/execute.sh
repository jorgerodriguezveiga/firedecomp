#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p thinnodes
#SBATCH -t 20:00:00

firedecomp_simulations -n 100 -nb 20 -na 20 -nm 5 -np 10 -m fix_work -so ../solver_options.yaml -vvv &> simulations.log
