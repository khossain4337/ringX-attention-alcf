#! /bin/bash -x
#
#
REPO_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}

export CCL_PROCESS_LAUNCHER=torchrun

module add frameworks
BACKEND=fused BENCHMARK_MODES="forward backward fwd_bwd" ./script/run_benchmarks.sh
