#!/bin/bash
# Usage: run_queue.sh <gpu> <jobfile>   (each line: <config> <fold>)
cd "$(dirname "$0")/.."
mkdir -p logs
while read -r cfg fold; do
  [ -z "$cfg" ] && continue
  name=$(basename "$cfg" .yaml)
  [ -f "results/$name/fold$fold/log.json" ] && continue
  CUDA_VISIBLE_DEVICES=$1 ~/env/ml/bin/python train.py --config "$cfg" --fold "$fold" \
    > "logs/${name}_fold${fold}.log" 2>&1
done < "$2"
