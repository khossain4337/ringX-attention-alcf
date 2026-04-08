#! /bin/bash -x 

BENCH_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf/test

module add frameworks

python ${BENCH_DIR}/bench_fused_attention.py
