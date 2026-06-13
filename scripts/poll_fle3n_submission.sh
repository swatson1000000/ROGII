#!/bin/bash
# Poll the fle3n v5 repro LB submission (2026-06-13 01:46 UTC) until it scores.
source /home/swatson/miniconda3/etc/profile.d/conda.sh
conda activate kaggle-arch
while true; do
    ROW=$(kaggle competitions submissions rogii-wellbore-geology-prediction 2>&1 | grep "2026-06-13 01:46")
    echo "$(date '+%Y-%m-%d %H:%M:%S') $ROW"
    case "$ROW" in
        *PENDING*) sleep 600 ;;
        *COMPLETE*|*ERROR*)
            echo "RESOLVED: $ROW"
            exit 0 ;;
        "")
            echo "row not found — listing head:"
            kaggle competitions submissions rogii-wellbore-geology-prediction 2>&1 | head -4
            sleep 600 ;;
        *) sleep 600 ;;
    esac
done
