#! /bin/bash -x
#
#
REPO_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}

export DEVICE_TYPE=xpu

export CCL_PROCESS_LAUNCHER=torchrun

module add frameworks

#Quick timer validation: targets the seqlen/config that showed negative elapsed_time
#BACKEND=fused \
#BENCHMARK_MODES="backward fwd_bwd" \
#ALGOS="ringX_attn.ringX1_attn" \
#SEQ_LENGTHS="8192" \
#BATCH_SIZE=2 \
#NUM_HEADS=8 \
#NUM_ITER=3 \
#./script/run_benchmarks.sh

BACKEND=fused BENCHMARK_MODES="forward backward fwd_bwd" ./script/run_benchmarks.sh
#BACKEND=portable BENCHMARK_MODES="forward backward fwd_bwd" ./script/run_benchmarks.sh
