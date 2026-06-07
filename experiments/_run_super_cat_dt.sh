#!/bin/bash
# Launch the super-solution CatBoost training on deepthought (runs in repo, backgrounded).
cd /home/swatson/work/kaggle/ROGII || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle
rm -f log/super_cat_*.log
LOG="log/super_cat_$(date +%Y%m%d_%H%M%S).log"
nohup python -u experiments/super_train_cat.py > "$LOG" 2>&1 &
echo "launched PID $! -> $LOG"
