#! /bin/bash -x
#
REPO_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf
BENCH_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf/test

#NNODES=`wc -l < $PBS_NODEFILE`
NNODES=1
NRANKS_PER_NODE=12

let NRANKS=${NNODES}*${NRANKS_PER_NODE}
echo "NUMBER_OF_NODES=${NNODES}"

#N=2
#PPN=2

#let PALS_WORLD_SIZE=${N}*${PPN}
let PALS_WORLD_SIZE=${NNODES}*${NRANKS_PER_NODE}
export PALS_WORLD_SIZE=${PALS_WORLD_SIZE}
echo "PALS_WORLD_SIZE = ${PALS_WORLD_SIZE}"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_MR_CACHE_MONITOR=userfaultfd

export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
#export CCL_LOG_LEVEL=debug

## For 1024+ nodes, try:
export CCL_KVS_MODE=mpi

#export CCL_ALLGATHERV_SCALEOUT=ring

#export CPU_AFFINITY="list:4-7"
#export CCL_WORKER_AFFINITY="42"
#export ZE_AFFINITY_MASK="0"

#export CPU_AFFINITY="list:4-7:8-11"
#export CCL_WORKER_AFFINITY="42,43"
#export ZE_AFFINITY_MASK="0,1"
#
#export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=2048 #4096
#export FI_CXI_DEFAULT_CQ_SIZE=4096
export FI_CXI_RX_MATCH_MODE=hybrid


export CPU_AFFINITY="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"
 
module add frameworks

cd ${REPO_DIR} && python -c "import ringX_attn; print(ringX_attn.__file__)"

cd ${REPO_DIR}
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}
#python -u ${BENCH_DIR}/test_fused_attention.py 2>&1 | tee ${BENCH_DIR}/test_fused_attention.txt

mpiexec -n ${PALS_WORLD_SIZE} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} \
    -env MASTER_ADDR=$(hostname).hsn.cm.aurora.alcf.anl.gov \
    -env MASTER_PORT=2345 python -u ${BENCH_DIR}/test_ringX1_attn_func.py 2>&1 | tee ${BENCH_DIR}/test_ringX1_attn_func.txt


