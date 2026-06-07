"""Reproduce the hidden-test exception locally.

The kernel ran COMPLETE on the 3 public wells but threw on the hidden ~200-well set.
The pre-GP kernel scored on the hidden set, so the bug is in the NEW GP code, triggered
by some well not represented in the 3 public ones. Closest local proxy: run the kernel's
build_hidden_features (is_train=False, WITH the GP imputer) over ALL 773 train wells and
catch the first per-well exception + traceback.
"""
from pathlib import Path
import glob, json, traceback
import numpy as np
import pandas as pd

ROOT = Path("/home/swatson/work/kaggle/ROGII")
src = (ROOT / "jupyter_konbu/rogii_konbu_inference.py").read_text()
head = src.split("# ---------------- Kaggle inference main")[0]
ns = {}
exec(head, ns)
FormationPlaneKNN = ns["FormationPlaneKNN"]; RowKNN = ns["RowKNN"]
FormationGP = ns["FormationGP"]; build_hidden_features = ns["build_hidden_features"]

train_paths = [Path(p) for p in sorted(glob.glob(str(ROOT / "data/raw/train/*__horizontal_well.csv")))]
anchor = json.load(open(ROOT / "models/konbu_gp/gp_anchor.json"))
print(f">> {len(train_paths)} wells; building imputers", flush=True)
form = FormationPlaneKNN(train_paths); row = RowKNN(train_paths); gp = FormationGP(train_paths, anchor)

throws = 0
none_count = 0
ok = 0
for i, p in enumerate(train_paths):
    wid = p.stem.replace("__horizontal_well", "")
    try:
        h = pd.read_csv(p)
        t = pd.read_csv(p.parent / f"{wid}__typewell.csv")
        feats = build_hidden_features(h, t, wid, is_train=False,
                                      formation_imputer=form, row_imputer=row, gp_imputer=gp)
        if feats is None:
            none_count += 1
        else:
            # also exercise the exact columns the kernel selects + nan audit on GP feats
            for c in ["gp_drift", "gp_std", "gp_ancc", "gp_vs_fk"]:
                nn = int(feats[c].isna().sum())
                if nn:
                    print(f"   [{wid}] {c} has {nn}/{len(feats)} NaN", flush=True)
            ok += 1
    except Exception:
        throws += 1
        print(f"\n!!! THROW on well {wid} (idx {i}):", flush=True)
        traceback.print_exc()
        if throws >= 3:
            print(">> stopping after 3 throws", flush=True)
            break
    if (i + 1) % 100 == 0:
        print(f"   {i+1}/{len(train_paths)}  ok={ok} none={none_count} throws={throws}", flush=True)

print(f"\n=== SUMMARY: ok={ok} none={none_count} throws={throws} ===", flush=True)
print("REPRO DONE", flush=True)
