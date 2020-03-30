#!/bin/zsh
#SBATCH --account=jara0001
#SBATCH --partition=c16m
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=distrubance
#SBATCH --output=sbatch.txt
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --cpus-per-task=2

PATH_TO_MAIN_PROG=~/repos/hpc/chameleon-apps/applications/

module load chameleon-lib


make clean
make dist
make -C ${PATH_TO_MAIN_PROG}matrix_example

mkdir output-files

export ITERATION_NUM=10

#mpiexec.hydra -np 4 -genvall ./wrapper.sh

for i in {0..9}
do
    export ITERATION_NUM=$i
    mpiexec -np 4 -genvall ./wrapper.sh
done
