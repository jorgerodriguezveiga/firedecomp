#!/bin/bash

# Methods
methods="original fix_work"
times="600 1200 1800"

# Main
for m in ${methods}; do
for t in ${times}; do

folder=m-${m}_nb-40_na-40_nm-40_np-100_time-${t}
executeFile=execute.sh
solverOptionsFile=solver_options.yaml
mkdir -p ${folder}

cd ${folder}

cat > ${executeFile} << EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p thinnodes
#SBATCH -t 06:00:00
#SBATCH --mem-per-cpu=10G

module load python imkl
export GRB_LICENSE_FILE=/home/csic/gim/dro/license/gurobi.lic

firedecomp_simulations -n 10 -nb 40 -na 40 -nm 40 -np 100 -m ${m} -so ${solverOptionsFile} -vvv &> simulations.log
EOF

cat > ${solverOptionsFile} << EOF
original:
  valid_constraints: ['work', 'contention']
  solver_options:
    MIPGapAbs: 0
    MIPGap: 0
    OutputFlag: 1
    LogToConsole: 1
    TimeLimit: ${t}
original_scip:
fix_work:
  mip_gap_obj: 0
  n_start_info: 100
  max_iters: 1000
  max_time: ${t}
  start_period: 100
  step_period: 6
  valid_constraints: ['max_obj', 'work', 'contention']
  solver_options:
    MIPGapAbs: 0
    MIPGap: 0
    OutputFlag: 0
    LogToConsole: 0
    TimeLimit: ${t}
benders_scip:
  limits/time: ${t}
gcg_scip:
  limits/time: ${t}
AL:
  limits/time: ${t}
EOF

chmod u+x ${executeFile}
sbatch -o output.out ${executeFile}
cd -
done
done
