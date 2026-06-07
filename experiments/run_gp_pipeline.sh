#!/bin/bash
# Wait for the GP feature build to finish, then run the combined gate.
# Writes coordination status to log/gp_pipeline.log and the gate verdict to /tmp/gp_gate_result.txt
set -u
cd /home/swatson/work/kaggle/ROGII
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch
PL=log/gp_pipeline.log
: > "$PL"
echo "[pipeline] $(date) waiting for GP build..." >> "$PL"

# wait up to 60 min for the build to finish (DONE marker) or error
for i in $(seq 1 360); do
  if ls log/gp_build_*.log >/dev/null 2>&1; then
    if grep -q "GP FEATURE BUILD DONE" log/gp_build_*.log; then
      echo "[pipeline] $(date) GP build DONE" >> "$PL"; break
    fi
    if grep -qiE "traceback|error" log/gp_build_*.log; then
      echo "[pipeline] $(date) GP build ERROR -- aborting" >> "$PL"
      tail -30 log/gp_build_*.log >> "$PL"; exit 1
    fi
  fi
  sleep 10
done

if [ ! -f data/processed/konbu/gp_feats_train.parquet ]; then
  echo "[pipeline] $(date) gp_feats_train.parquet not found -- aborting" >> "$PL"; exit 1
fi

echo "[pipeline] $(date) launching gp_gate.py" >> "$PL"
GLOG=log/gp_gate_$(date +%Y%m%d_%H%M%S).log
python -u experiments/gp_gate.py > "$GLOG" 2>&1
echo "[pipeline] $(date) gp_gate.py exited rc=$? (log=$GLOG)" >> "$PL"
