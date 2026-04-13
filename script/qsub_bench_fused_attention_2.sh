#! /bin/bash -x
#
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=03:00:00
#PBS -q capacity
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -N FA_Triton
#PBS -o /lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf/script/outdir_aurora
#PBS -e /lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf/script/errordir_aurora
#PBS -j oe
#
tstamp() {
     date +"%Y-%m-%d-%H%M%S"
}

## Proxies to clone from a compute node
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128

BENCH_DIR=/lus/flare/projects/datasets/softwares/testing/incite_2026_ringx/ringX-attention-alcf/test

module add frameworks

#echo "========================================"
#echo "  Running ORIGINAL bench_fused_attention"
#echo "========================================"
#python ${BENCH_DIR}/bench_fused_attention.py | tee ${BENCH_DIR}/bench_original.txt

echo ""
echo "========================================"
echo "  Running NEW bench_fused_attention_2"
echo "========================================"
python ${BENCH_DIR}/bench_fused_attention_2.py 2>&1 | tee ${BENCH_DIR}/bench_new.txt

echo ""
echo "========================================"
echo "  DIFF (should be empty if identical)"
echo "========================================"
diff ${BENCH_DIR}/bench_original.txt ${BENCH_DIR}/bench_new.txt && echo "SUCCESS: outputs are identical!" || echo "DIFFERENCES FOUND — see above"
