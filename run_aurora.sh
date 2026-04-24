#! /bin/bash -x
#
#
REPO_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf
export TRITON_AUTOTUNE_CACHE_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/triton_autotune_cache
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}

export DEVICE_TYPE=xpu

export CCL_PROCESS_LAUNCHER=torchrun

module add frameworks

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

BACKEND=portable ALGOS="ringX_attn.ringX2_attn" \
    BENCHMARK_MODES="backward" BATCH_SIZE=32 \
    NPROC=4 SEQ_LENGTHS="16384" ./script/run_benchmarks.sh


#BACKEND=portable ALGOS="ringX_attn.ringX2_attn" \
#    BENCHMARK_MODES="backward" BATCH_SIZE=32 \
#    NPROC=4 ./script/run_benchmarks.sh
#
#BACKEND=fused ALGOS="ringX_attn.ringX2_attn" \
#    BENCHMARK_MODES="backward" BATCH_SIZE=32 \
#    NPROC=4 ./script/run_benchmarks.sh

#BACKEND=fused ALGOS="ringX_attn.ringX1_attn" \
#    BENCHMARK_MODES="forward backward fwd_bwd" \
#    NPROC=4 ./script/run_benchmarks.sh
#
#BACKEND=portable ALGOS="ringX_attn.ringX1_attn" \
#    BENCHMARK_MODES="forward backward fwd_bwd" \
#    NPROC=4 ./script/run_benchmarks.sh

#BACKEND=fused ALGOS="ringX_attn.ringX2_attn" \
#    BENCHMARK_MODES="forward backward fwd_bwd" \
#    NPROC=4 ./script/run_benchmarks.sh
#
#BACKEND=portable ALGOS="ringX_attn.ringX2_attn" \
#    BENCHMARK_MODES="forward backward fwd_bwd" \
#    NPROC=4 ./script/run_benchmarks.sh
