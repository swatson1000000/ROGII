#!/bin/bash
# Poll the fle3n v5 repro kernel until it finishes, then pull its output.
source /home/swatson/miniconda3/etc/profile.d/conda.sh
conda activate kaggle-arch
KERNEL="stevewatson999/rogii-fle3n-v5-repro"
OUTDIR="/tmp/fle3n_repro_output"
while true; do
    STATUS=$(kaggle kernels status "$KERNEL" 2>&1)
    echo "$(date '+%Y-%m-%d %H:%M:%S') $STATUS"
    case "$STATUS" in
        *RUNNING*|*QUEUED*) sleep 600 ;;
        *COMPLETE*)
            echo "COMPLETE — pulling output"
            mkdir -p "$OUTDIR"
            kaggle kernels output "$KERNEL" -p "$OUTDIR" 2>&1
            ls -la "$OUTDIR"
            exit 0 ;;
        *ERROR*|*CANCEL*)
            echo "TERMINAL non-complete status — pulling log"
            mkdir -p "$OUTDIR"
            kaggle kernels output "$KERNEL" -p "$OUTDIR" 2>&1 || true
            exit 1 ;;
        *) sleep 600 ;;
    esac
done
