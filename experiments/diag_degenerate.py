"""Hunt the hidden-rerun throw: exercise the FULL kernel path (features -> model predict)
on degenerate wells the 3 public + 773 train wells may not cover.

Cases:
  A) NaN in X/Y on some hidden rows -> does GP produce NaN -> do lgb/xgb/cat predict throw?
  B) a 1-hidden-row well
  C) a well whose typewell has <8 GR rows
  D) confirm the GP block leaves NaN (no fillna in kernel) and whether that breaks predict
Also: does the kernel `feats[feature_cols]` have any non-float/object column or all-NaN col?
"""
from pathlib import Path
import glob, json, traceback
import numpy as np
import pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor

ROOT = Path("/home/swatson/work/kaggle/ROGII")
src = (ROOT / "jupyter_konbu/rogii_konbu_inference.py").read_text()
head = src.split("# ---------------- Kaggle inference main")[0]
ns = {}; exec(head, ns)
FormationPlaneKNN = ns["FormationPlaneKNN"]; RowKNN = ns["RowKNN"]
FormationGP = ns["FormationGP"]; build_hidden_features = ns["build_hidden_features"]
MODELS = ROOT / "models/konbu_gp"; N = 5

train_paths = [Path(p) for p in sorted(glob.glob(str(ROOT / "data/raw/train/*__horizontal_well.csv")))]
anchor = json.load(open(MODELS / "gp_anchor.json"))
form = FormationPlaneKNN(train_paths); row = RowKNN(train_paths); gp = FormationGP(train_paths, anchor)
feat_cols = json.load(open(MODELS / "feature_cols.json"))
print(">> imputers ready", flush=True)

def predict_all(feats):
    X = feats[feat_cols]; Xv = X.values
    out = {}
    for k in ["lgb_42", "cat_42", "xgb_42"]:
        seed = k.split("_")[1]
        if k.startswith("lgb"):
            out[k] = lgb.Booster(model_file=str(MODELS / f"lgb_seed{seed}_fold0.txt")).predict(X)
        elif k.startswith("cat"):
            m = CatBoostRegressor(); m.load_model(str(MODELS / f"cat_seed{seed}_fold0.cbm")); out[k] = m.predict(Xv)
        else:
            b = xgb.Booster(); b.load_model(str(MODELS / f"xgb_seed{seed}_fold0.json"))
            out[k] = b.predict(xgb.DMatrix(Xv), iteration_range=(0, int(b.best_iteration)+1))
    return out

# baseline: a real public test well
tp = ROOT / "data/raw/test/000d7d20__horizontal_well.csv"
h0 = pd.read_csv(tp); t0 = pd.read_csv(tp.parent / "000d7d20__typewell.csv")

def trial(name, h, t, wid="000d7d20"):
    print(f"\n--- {name} ---", flush=True)
    try:
        feats = build_hidden_features(h, t, wid, is_train=False,
                                      formation_imputer=form, row_imputer=row, gp_imputer=gp)
        if feats is None:
            print("   build returned None (skipped by kernel)", flush=True); return
        # NaN audit on the feature columns the kernel selects
        Xsel = feats[feat_cols]
        nan_cols = {c: int(Xsel[c].isna().sum()) for c in feat_cols if Xsel[c].isna().any()}
        obj_cols = [c for c in feat_cols if Xsel[c].dtype == object]
        print(f"   feats shape={feats.shape}; NaN cols={nan_cols}; object cols={obj_cols}", flush=True)
        pr = predict_all(feats)
        for k, v in pr.items():
            print(f"   {k}: nan_in_pred={int(np.isnan(v).sum())} mean={np.nanmean(v):.3f}", flush=True)
        print("   OK", flush=True)
    except Exception:
        print("   !!! THREW:", flush=True); traceback.print_exc()

# A) NaN in X/Y on hidden rows
hA = h0.copy(); ms = int(np.flatnonzero(hA["TVT_input"].isna().to_numpy())[0])
hA.loc[hA.index[ms+10:ms+20], "X"] = np.nan
trial("A: NaN X on 10 hidden rows", hA, t0)

# B) 1 hidden row
hB = h0.iloc[:ms+1].copy()
trial("B: single hidden row", hB, t0)

# C) typewell with <8 GR rows
tC = t0.iloc[:5].copy()
trial("C: typewell <8 rows", h0, tC)

# D) NaN ALL X (whole-well coord missing)
hD = h0.copy(); hD["X"] = np.nan
trial("D: all-NaN X", hD, t0)

# E) NaN in Z hidden rows
hE = h0.copy(); hE.loc[hE.index[ms:ms+5], "Z"] = np.nan
trial("E: NaN Z on hidden rows", hE, t0)

print("\n=== DEGENERATE DIAG DONE ===", flush=True)
