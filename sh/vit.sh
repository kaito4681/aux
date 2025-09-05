# nohup ./sh/vit.sh > vit.log 2>&1 &

# export WANDB_API_KEY=

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "PID: $$"
uv run main.py --use-wandb >> logs/vit_${TIMESTAMP}.log 2>&1
uv run main.py --aux --use-wandb >> logs/vit_${TIMESTAMP}.log 2>&1
