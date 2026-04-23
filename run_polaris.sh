#! /bin/bash -x
#
#
REPO_DIR=/lus/eagle/projects/datascience/hossainm/testing/incite_2026_ringx/ringX-attention-alcf
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}

export DEVICE_TYPE=cuda

#export CCL_PROCESS_LAUNCHER=torchrun

module use /soft/modulefiles
module add conda
conda activate base

#Quick timer validation: targets the seqlen/config that showed negative elapsed_time
#BACKEND=fused \
#BENCHMARK_MODES="backward fwd_bwd" \
#ALGOS="ringX_attn.ringX1_attn ringX_attn.ringX2_attn" \
#SEQ_LENGTHS="8192" \
#BATCH_SIZE=2 \
#NUM_HEADS=32 \
#NUM_ITER=3 \
#NPROC=4 \
#./script/run_benchmarks.sh

#BACKEND=fused BENCHMARK_MODES="forward backward fwd_bwd" ./script/run_benchmarks.sh

#BACKEND=fused ALGOS="ringX_attn.ringX1_attn" \
#    BENCHMARK_MODES="forward backward fwd_bwd" \
#    NPROC=4 ./script/run_benchmarks.sh
#
BACKEND=portable ALGOS="ringX_attn.ringX2_attn" \
    BENCHMARK_MODES="backward" BATCH_SIZE=4 \
    NPROC=1 ./script/run_benchmarks.sh

BACKEND=fused ALGOS="ringX_attn.ringX2_attn" \
    BENCHMARK_MODES="backward" BATCH_SIZE=4 \
    NPROC=1 ./script/run_benchmarks.sh
#
#BACKEND=portable ALGOS="ringX_attn.ringX2_attn" \
#    BENCHMARK_MODES="forward backward fwd_bwd" \
#    NPROC=4 ./script/run_benchmarks.sh

#BACKEND=flash_attn ALGOS="ringX_attn.ringX2_attn" \
#    BENCHMARK_MODES="forward backward fwd_bwd" \
#    NPROC=4 ./script/run_benchmarks.sh
