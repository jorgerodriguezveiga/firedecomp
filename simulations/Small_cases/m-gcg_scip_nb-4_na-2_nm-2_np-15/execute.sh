#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p thinnodes
#SBATCH -t 17:00:00

module load python/3.6.8
module load zlib
module load gmp
module load libreadline/7.0
module load gsl
module load gcc/6.4.0

firedecomp_simulations -n 100 -nb 4 -na 2 -nm 2 -np 15 -m gcg_scip -so ../solver_options.yaml -vvv &> simulations.log
