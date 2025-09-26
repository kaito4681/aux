#!/bin/bash

# Usage:
# chmod +x sh/all.sh
# nohup ./sh/all.sh > all.log 2>&1 &

# export WANDB_API_KEY=

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs/${TIMESTAMP}

echo "PID: $$" >> logs/${TIMESTAMP}/pid.log 2>&1

BATCH_SIZE=16

USE_WANDB=true
USE_TQDM=false
CHECK=false
AUX_LIST=("none" "mid" "all")

for AUX in "${AUX_LIST[@]}"; do
	for pretrained in true false; do
		CMD="uv run main.py --aux ${AUX}"
		if [ "$pretrained" = true ]; then
			CMD+=" --pretrained"
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
		CMD+=" --batch-size ${BATCH_SIZE}"

		echo "$CMD" >> logs/${TIMESTAMP}/main.log 2>&1
		echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" >> logs/${TIMESTAMP}/main.log 2>&1
		
		if [ "$pretrained" = true ]; then
			eval $CMD >> logs/${TIMESTAMP}/vit-${AUX}-pretrained.log 2>&1
		else
			eval $CMD >> logs/${TIMESTAMP}/vit-${AUX}.log 2>&1
		fi
	done
done

echo "All runs completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> logs/${TIMESTAMP}/main.log 2>&1