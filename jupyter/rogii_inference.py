"""ROGII code-competition inference kernel (Phase-3 formation-KNN LightGBM, OOF 12.76).

Runs offline on Kaggle. Reuses the project's tested src/ inference path by shimming
three config paths:
  C.RAW    -> the competition input (train/ + test/ + sample_submission.csv)
  C.MODELS -> per-fold models from the attached artifacts dataset
  C.PROC   -> /kaggle/working

At submit time Kaggle swaps in the hidden ~200-well test/; this same code rebuilds the
train-reference imputers (FormationPlaneKNN + RowKNN) from the competition train/ and
predicts drift, then adds the per-well anchor back. Writes /kaggle/working/submission.csv.

Robust to Kaggle's archive auto-extraction: src may arrive as a folder (extracted) or as
src.zip (importable via zipimport); models may arrive as .txt (extracted) or .txt.gz.
"""
import gzip
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

INPUT = Path("/kaggle/input")


def _all_dirs(root: Path, maxdepth: int = 4):
    """All directories under root up to maxdepth (CLI-attached sources nest as
    /kaggle/input/{competitions,datasets}/<...>/<slug>/)."""
    out = [root]
    if not root.exists():
        return out

    def rec(d: Path, depth: int):
        if depth > maxdepth:
            return
        try:
            for x in sorted(d.iterdir()):
                if x.is_dir():
                    out.append(x)
                    rec(x, depth + 1)
        except Exception:
            pass

    rec(root, 1)
    return out


cands = _all_dirs(INPUT)
print("[input] dirs under /kaggle/input:", [str(p) for p in cands], flush=True)


def _has_artifacts(p: Path) -> bool:
    return bool(list(p.glob("lgb_phase3_fold*.txt*"))) or (p / "src").exists() or (p / "src.zip").exists()


COMP = next((p for p in cands if (p / "sample_submission.csv").exists()), None)
ART = next((p for p in cands if _has_artifacts(p)), None)
if COMP is None or ART is None:
    raise FileNotFoundError(f"COMP={COMP} ART={ART}; dirs={[str(p) for p in cands]}")
print(f"[locate] COMP={COMP}  ART={ART}", flush=True)


# ── locate the src package, robust to Kaggle's archive auto-extraction ──
def _find_src_parent(root: Path):
    for cand in (root, root / "src"):
        if (cand / "src" / "config.py").exists():
            return str(cand)
    if (root / "src.zip").exists():
        return str(root / "src.zip")            # Python zipimport
    for q in root.rglob("config.py"):           # last-resort search
        if q.parent.name == "src":
            return str(q.parent.parent)
    raise FileNotFoundError(f"src package not found under {root}: {list(root.iterdir())}")


sys.path.insert(0, _find_src_parent(ART))
from src import config as C  # noqa: E402

C.RAW = COMP
C.PROC = Path("/kaggle/working")

# ── stage per-fold models into working (decompress .gz if needed) ──
mdir = Path("/kaggle/working/models")
mdir.mkdir(parents=True, exist_ok=True)
for f in range(C.N_FOLDS):
    dst = mdir / f"lgb_phase3_fold{f}.txt"
    txt = ART / f"lgb_phase3_fold{f}.txt"
    gz = ART / f"lgb_phase3_fold{f}.txt.gz"
    if txt.exists():
        shutil.copy(txt, dst)
    elif gz.exists():
        with gzip.open(gz, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        raise FileNotFoundError(f"no model for fold {f} under {ART}: {list(ART.iterdir())}")
C.MODELS = mdir

# well-id lookups are lru_cached and read C.RAW; clear in case import warmed them
for fn in (C.train_well_ids, C.test_well_ids):
    try:
        fn.cache_clear()
    except Exception:
        pass

from src import features as F  # noqa: E402  (import after C.RAW is set)

print(f"[paths] RAW={C.RAW}  MODELS={C.MODELS}", flush=True)
print(f"[wells] train={len(C.train_well_ids())}  test={len(C.test_well_ids())}", flush=True)

mat = F.build_feature_matrix("test", cache=False)
feat_cols = F.feature_columns(mat)
print(f"[features] {len(mat):,} rows x {len(feat_cols)} feats", flush=True)

models = [lgb.Booster(model_file=str(mdir / f"lgb_phase3_fold{f}.txt")) for f in range(C.N_FOLDS)]
drift = np.mean([m.predict(mat[feat_cols]) for m in models], axis=0)
sub = pd.DataFrame({"id": mat["id"], "tvt": mat["anchor_tvt"].to_numpy() + drift})

ss = pd.read_csv(COMP / "sample_submission.csv")
out = ss[["id"]].merge(sub, on="id", how="left")
n_missing = int(out["tvt"].isna().sum())
if n_missing:
    print(f"[warn] {n_missing} ids unmatched -> filling with median", flush=True)
    out["tvt"] = out["tvt"].fillna(float(np.nanmedian(sub["tvt"])))

out.to_csv("/kaggle/working/submission.csv", index=False)
print(f"[submit] wrote {len(out):,} rows  tvt[{out['tvt'].min():.1f}, {out['tvt'].max():.1f}]  "
      f"mean={out['tvt'].mean():.1f}", flush=True)
