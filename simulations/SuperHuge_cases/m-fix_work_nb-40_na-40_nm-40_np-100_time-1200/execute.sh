#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p thinnodes
#SBATCH -t 06:00:00
#SBATCH --mem-per-cpu=10G

module load python imkl
export GRB_LICENSE_FILE=/home/csic/gim/dro/license/gurobi.lic

firedecomp_simulations -s 8 9 -nb 40 -na 40 -nm 40 -np 100 -m fix_work -so solver_options.yaml -ar -vvv &> simulations.log
