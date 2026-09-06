#!/usr/bin/env bash
# Waits for the ROCE-final arm, restarts the cluster with the RoCE-off override (same image), runs NCCL-final, compares.
set -u
P=~/spark_vllm/benchmark-results/roce-ab-final-20260903
until grep -q "DONE ROCE-final" $P/status.txt; do sleep 20; done
BASE='cd ~/spark_vllm && E=benchmark-results/glm53-jovian-r12-kv33g-20260901/r12-dflash62f758c-mtp5-kv33g-$(hostname).env && docker compose --env-file $E -f docker-compose-glm53-flash-dflash2-tp4-spark.yml -f benchmark-results/glm53-jovian-r12-kv33g-20260901/docker-compose-r12-dflash62f758c-mtp5-kv33g.override.yml -f docker-compose-dual-hca.override.yml -f docker-compose-chunk8192.override.yml'
for h in maxwell ampere faraday hertz; do ssh $h "$BASE -f docker-compose-roce-allreduce.override.yml down" >/dev/null 2>&1 & done; wait
for h in maxwell ampere faraday hertz; do ssh $h "$BASE -f docker-compose-roce-off.override.yml up -d" >/dev/null 2>&1 & done; wait
echo "$(date '+%F %T') NCCL arm: cluster restarted with roce-off override" >> $P/status.txt
ssh maxwell 'C=$(docker ps -q --filter name=glm53 | head -1); for i in $(seq 1 80); do docker logs $C 2>&1 | grep -q "Application startup complete" && break; sleep 5; done; docker logs $C 2>&1 | grep -E "all-reduce backends.*tp:0|RoCEnante" | head -2 | cut -c1-160; M=$(curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)[\"data\"][0][\"id\"])"); curl -s -m 120 http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python one-liner that reverses a string.\"}],\"max_tokens\":60,\"temperature\":0}" | python3 -c "import sys,json;d=json.load(sys.stdin);print(\"sanity:\",d[\"choices\"][0][\"message\"][\"content\"][:120].replace(chr(10),\" \"))"' >> $P/nccl-startup.txt 2>&1
$P/run-arm.sh NCCL-final
python3 $P/compare.py ROCE-final NCCL-final > $P/ROCE-vs-NCCL.txt 2>&1
echo "$(date '+%F %T') ALL DONE" >> $P/status.txt
