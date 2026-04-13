#! /bin/bash -x
#
REPO_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf
BENCH_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf/test
 
module add frameworks

cd ${REPO_DIR} && python -c "import ringX_attn; print(ringX_attn.__file__)"

cd ${REPO_DIR}
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}
python -u ${BENCH_DIR}/test_fused_attention.py 2>&1 | tee ${BENCH_DIR}/test_fused_attention.txt


