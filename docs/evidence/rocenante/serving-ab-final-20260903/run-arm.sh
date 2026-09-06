#!/usr/bin/env bash
# usage: run-arm.sh <label>   -- decode matrix + coding peak, then standalone cold prefill, against maxwell:8000
set -u
L="$1"; R=~/spark_vllm/benchmark-results/roce-ab-final-20260903; cd ~/spark_vllm/llm-inference-bench
COMMON="--host maxwell --port 8000 --model glm-5.3-flash-nvfp4-dflash2 --token-targeting exact --display-mode plain --no-hw-monitor --no-resume"
echo "$(date '+%F %T') start $L decode+coding" >> $R/status.txt
python3 llm_decode_bench.py $COMMON --concurrency 1,2,4,8,16 --contexts 0,16k,32k --duration 30 --max-tokens 512 \
  --cell-warmup-timeout-seconds 180 --show-capacity-limited-values --coding-peak --coding-peak-runs 5 \
  --output "$R/$L-decode.json" > "$R/$L-decode.log" 2>&1
echo "$(date '+%F %T') start $L prefill" >> $R/status.txt
python3 llm_decode_bench.py $COMMON --prefill-only --standalone-prefill --prefill-contexts 8k,32k,128k --prefill-duration 30 \
  --output "$R/$L-prefill.json" > "$R/$L-prefill.log" 2>&1
echo "$(date '+%F %T') DONE $L" >> $R/status.txt
