#!/bin/bash

# Usage:
# chmod +x sh/plainnet.sh
# nohup ./sh/plainnet.sh > all.log 2>&1 &

# export WANDB_API_KEY=

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs/plainnet/${TIMESTAMP}

echo "PID: $$" >> logs/plainnet/${TIMESTAMP}/pid.log 2>&1

BATCH_SIZE=128

USE_WANDB=true
USE_TQDM=false
CHECK=false
USE_AUX=false
MODEL_SIZES=("18" "34" "50" "101" "152")
SEEDS=(100 200 300)

NUM_GPUS=2
CURRENT_GPU=0

for SEED in "${SEEDS[@]}"; do
for SIZE in "${MODEL_SIZES[@]}"; do
	# GPUの数だけバックグラウンドプロセスが実行されていたら待つ
	while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
		sleep 1
	done

	GPU_ID=$((CURRENT_GPU % NUM_GPUS))
	CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} uv run python plainnet.py --model-size ${SIZE} --seed ${SEED} --batch-size ${BATCH_SIZE}"

	if [ "$USE_AUX" = true ]; then
		CMD+=" --aux"
	fi

	if [ "$USE_WANDB" = true ]; then
		CMD+=" --use-wandb"
	fi

	if [ "$USE_TQDM" = true ]; then
		CMD+=" --use-tqdm"
	fi

	if [ "$CHECK" = true ]; then
		CMD+=" --check"
	fi

	echo "$CMD" >> logs/plainnet/${TIMESTAMP}/main.log 2>&1
	echo "Start time: $(date '+%Y-%m-%d %H:%M:%S') on GPU ${GPU_ID}" >> logs/plainnet/${TIMESTAMP}/main.log 2>&1

	# 実行してログを残す (バックグラウンドで実行)
	eval "$CMD" >> logs/plainnet/${TIMESTAMP}/${SIZE}_${SEED}.log 2>&1 &

	echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> logs/plainnet/${TIMESTAMP}/main.log 2>&1
	echo "-" >> logs/plainnet/${TIMESTAMP}/main.log 2>&1

	CURRENT_GPU=$((CURRENT_GPU + 1))
done
done

# すべてのバックグラウンドジョブの終了を待つ
wait

echo "All runs completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> logs/plainnet/${TIMESTAMP}/main.log 2>&1