#!/bin/bash

# Maria option
# methods="original"

# Jorge option
methods="benders_scip"

#Other options
numB="5 20"
numA="5 20"
numM="5 20"
numP="10 20 30 40 50 60"

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
#SBATCH -t 20:00:00

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
