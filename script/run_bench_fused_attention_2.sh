#! /bin/bash -x

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
python ${BENCH_DIR}/bench_fused_attention_2.py 2>&1 | tee ${BENCH_DIR}/bench_new_interactive.txt

echo ""
echo "========================================"
echo "  DIFF (should be empty if identical)"
echo "========================================"
diff ${BENCH_DIR}/bench_original.txt ${BENCH_DIR}/bench_new.txt && echo "SUCCESS: outputs are identical!" || echo "DIFFERENCES FOUND — see above"
