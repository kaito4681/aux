#!/bin/bash

# Usage:
# chmod +x sh/plain.sh
# nohup ./sh/plain.sh > plain.log 2>&1 &

# https://wandb.ai/authorize
# export WANDB_API_KEY=

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs/plain/${TIMESTAMP}

echo "PID: $$" >> logs/plain/${TIMESTAMP}/pid.log 2>&1

BATCH_SIZE=128

USE_WANDB=true
USE_TQDM=false
CHECK=false
# MODEL_SIZES=("152" "101" "50" "34" "18")
MODEL_SIZES=("34" "18")
# SEEDS=(100 200 300)
SEEDS=(100)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs" >> logs/plain/${TIMESTAMP}/main.log 2>&1
CURRENT_GPU=0

# 各実験設定の配列
CONFIGS=("base" "base-aux" "bn" "bn-aux" "skip" "skip-aux" "resnet" "resnet-aux")

for SEED in "${SEEDS[@]}"; do
for SIZE in "${MODEL_SIZES[@]}"; do
for CONFIG in "${CONFIGS[@]}"; do
	# GPUの数だけバックグラウンドプロセスが実行されていたら待つ
	while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
		sleep 1
	done

	GPU_ID=$((CURRENT_GPU % NUM_GPUS))
	CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} uv run plain.py --model-size ${SIZE} --seed ${SEED} --batch-size ${BATCH_SIZE}"

	# 設定に応じてオプションを追加
	case $CONFIG in
		"base")
			# 基本設定（オプション追加なし）
			;;
		"base-aux")
			CMD+=" --aux"
			;;
		"bn")
			CMD+=" --bn"
			;;
		"bn-aux")
			CMD+=" --bn --aux"
			;;
		"skip")
			CMD+=" --skip"
			;;
		"skip-aux")
			CMD+=" --skip --aux"
			;;
		"resnet")
			CMD+=" --skip --bn"
			;;
		"resnet-aux")
			CMD+=" --skip --bn --aux"
	esac

	if [ "$USE_WANDB" = true ]; then
		CMD+=" --use-wandb"
	fi

	if [ "$USE_TQDM" = true ]; then
		CMD+=" --use-tqdm"
	fi

	if [ "$CHECK" = true ]; then
		CMD+=" --check"
	fi

	echo "$CMD" >> logs/plain/${TIMESTAMP}/main.log 2>&1
	echo "Start time: $(date '+%Y-%m-%d %H:%M:%S') on GPU ${GPU_ID} [${CONFIG}]" >> logs/plain/${TIMESTAMP}/main.log 2>&1

	# 実行してログを残す (バックグラウンドで実行)
	eval "$CMD" >> logs/plain/${TIMESTAMP}/${SIZE}_${SEED}_${CONFIG}.log 2>&1 &

	echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> logs/plain/${TIMESTAMP}/main.log 2>&1
	echo "-" >> logs/plain/${TIMESTAMP}/main.log 2>&1

	CURRENT_GPU=$((CURRENT_GPU + 1))
done
done
done

# すべてのバックグラウンドジョブの終了を待つ
wait

echo "All runs completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> logs/plain/${TIMESTAMP}/main.log 2>&1