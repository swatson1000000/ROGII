#!/bin/bash
# Validate the GP-augmented inference kernel end-to-end against a mock /kaggle/input,
# then compare its submission to the gate's ground-truth (konbu_gp/submission_local.csv).
set -u
cd /home/swatson/work/kaggle/ROGII
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch

V=/tmp/gpval
rm -rf "$V"; mkdir -p "$V/input/comp" "$V/input/art"

SAMP=$(find data -name "sample_submission.csv" 2>/dev/null | head -1)
echo "[setup] sample_submission: $SAMP"
cp "$SAMP" "$V/input/comp/sample_submission.csv"
# real dirs with symlinked files: Path.rglob does NOT descend symlinked DIRS in this py
cp -rs "$PWD/data/raw/train" "$V/input/comp/train"
cp -rs "$PWD/data/raw/test"  "$V/input/comp/test"
cp models/konbu_gp/* "$V/input/art/" 2>/dev/null
echo "[setup] art files: $(ls $V/input/art | tr '\n' ' ')"

# patched copy of the kernel: local input + local output
sed -e 's#/kaggle/input#'"$V"'/input#g' \
    -e 's#/kaggle/working/submission.csv#'"$V"'/submission.csv#g' \
    jupyter_konbu/rogii_konbu_inference.py > "$V/kernel_local.py"

echo "[run] executing patched kernel..."
python -u "$V/kernel_local.py" > "$V/run.log" 2>&1
RC=$?
echo "[run] rc=$RC"
tail -25 "$V/run.log"

echo "[compare] kernel output vs gate ground-truth"
python - <<'PY' > "$V/verdict.txt" 2>&1
import pandas as pd, numpy as np
k = pd.read_csv("/tmp/gpval/submission.csv")
g = pd.read_csv("data/processed/konbu_gp/submission_local.csv")
m = k.merge(g, on="id", suffixes=("_kernel","_gate"))
assert len(m)==len(g)==len(k), f"row mismatch k={len(k)} g={len(g)} merged={len(m)}"
d = np.abs(m["tvt_kernel"].to_numpy()-m["tvt_gate"].to_numpy())
print(f"rows={len(m)}  max|d|={d.max():.6g}  mean|d|={d.mean():.6g}  p99={np.percentile(d,99):.6g}")
print("VERDICT:", "KERNEL MATCHES gate -> safe to ship" if d.max()<0.05 else f"MISMATCH max={d.max():.4g} -> investigate")
PY
cat "$V/verdict.txt"
echo "=== VALIDATION DONE ==="
