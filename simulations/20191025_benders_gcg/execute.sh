#!/bin/bash

# Maria option
# methods="original"

# Jorge option
methods="gcg_scip"

#Other options
numB="2 4"
numA="2 4"
numM="2 4"
numP="10 15"

# 48 cases

# Main
for m in ${methods}; do
for nb in ${numB}; do
for na in ${numA}; do
for nm in ${numM}; do
for np in ${numP}; do

folder=m-${m}_nb-${nb}_na-${na}_nm-${nm}_np-${np}
file=execute.sh
mkdir -p ${folder}

cd ${folder}

cat > ${file} << EOF
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

firedecomp_simulations -n 100 -nb ${nb} -na ${na} -nm ${nm} -np ${np} -m ${m} -so ../solver_options.yaml -vvv &> simulations.log
EOF

chmod u+x ${file}
sbatch -o output.out ${file}
cd -

done
done
done
done
done
