#!/bin/zsh

#PMI_RANK=1

#echo "Number of Ranks Requested $PMI_SIZE. Executing Rank $PMI_RANK, iteration num: $ITERATION_NUM"

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=24

export MATRIX_SIZE=300
export NUM_TASKS=2000
#kleinere größe, mehr tasks, zb 1000 tasks, size = 300
RANKS_TO_DISTURB=(1 3)

DIST_NUM_THREADS=24
DIST_TYPE=compute
#DIST_TYPE=memory
#DIST_TYPE=communication
DIST_RANDOM=false
#DIST_RANDOM=true
DIST_COMP_WINDOW=10
DIST_PAUSE_WINDOW=0
DIST_MIN_COMP_WINDOW=2
DIST_MAX_COMP_WINDOW=20
DIST_RAM_MB=30000

FILENAME=output-files/output_rank_${PMI_RANK}_${ITERATION_NUM}.txt
PATH_TO_MAIN_PROG=~/repos/hpc/chameleon-apps/applications/matrix_example/

args="--type=$DIST_TYPE --rank_number=$PMI_RANK --use_multiple_cores=$DIST_NUM_THREADS --use_random=$DIST_RANDOM --window_size_min=$DIST_MIN_COMP_WINDOW --window_size_max=$DIST_MAX_COMP_WINDOW --use_ram=$DIST_RAM_MB"

echo "This is the Outputfile for Rank $PMI_RANK" >> $FILENAME

processID=""
for i in $RANKS_TO_DISTURB
do
    if [[ $i = $PMI_RANK ]]; then
        ./dist.exe ${args} >> $FILENAME &
        processID=$!
        echo "Rank $PMI_RANK will be disturbed with PID = $processID"
        echo "Disturbance PID is $processID" >> $FILENAME
    fi
done

${PATH_TO_MAIN_PROG}main $MATRIX_SIZE $NUM_TASKS $NUM_TASKS $NUM_TASKS $NUM_TASKS >> $FILENAME

if [[ 0 = $PMI_RANK ]]; then
    grep "Computations" $FILENAME
fi

if [[ $processID != "" ]]; then
    kill -9 $processID
fi