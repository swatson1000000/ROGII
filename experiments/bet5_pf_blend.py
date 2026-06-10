"""Stage B: blend the NEW +UK GBM stack with the production PF (scale-12), find OOF-optimal w,
and compare to the banked GBM-only blend (9.1732) -- the number that predicts the LB."""
from pathlib import Path
import glob, os, json
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import Ridge
ROOT=Path("/home/swatson/work/kaggle/ROGII"); RAW=ROOT/"data/raw/train"
FRU=ROOT/"data/processed/frontier_uk"; rmse=lambda e:float(np.sqrt(np.mean(e*e)))
keys=['lgb_42','lgb_7','lgb_123','cat_42','cat_7','cat_123']

def stack_resid(mdir):
    df=pd.read_parquet(FRU/"train_feats.parquet",columns=["well","id","last_known_tvt","target"])
    y=df["target"].to_numpy(np.float64)
    X=np.column_stack([np.load(f"{mdir}/oof_{k}.npy").astype(np.float64) for k in keys])
    r=Ridge(alpha=1.0,fit_intercept=False,positive=True).fit(X,y)
    df["_ri"]=df["id"].str.rsplit("_",n=1).str[-1].astype(int)
    return df, (X@r.coef_)-y   # GBM drift residual (pred-true), abs-space residual identical

df, gbm_resid = stack_resid("models/frontier_uk")
df_b, gbm_resid_b = df.copy(), None
_,gbm_resid_b = stack_resid("models/frontier")  # banked GBM (222, no UK) for apples-to-apples

paths=sorted(glob.glob(str(RAW/"*__horizontal_well.csv"))); pf_wells=[]
for p in paths:
    wid=os.path.basename(p).replace("__horizontal_well.csv","")
    hw=pd.read_csv(p,usecols=lambda c:c in("TVT","TVT_input"))
    if "TVT" not in hw.columns or hw["TVT_input"].isna().sum()==0 or hw["TVT_input"].notna().sum()<20: continue
    pf_wells.append(wid)
grp={w:g.sort_values("_ri") for w,g in df.groupby("well",sort=False)}
def good(r): return r is not None and not(isinstance(r[0],str) and r[0]=="ERR")
res=joblib.load(ROOT/"models/frontier/pf_real_results.pkl"); rg=[r for r in res if good(r)]
assert len(rg)==len(pf_wells)
pos_l,pres_l=[],[]
for i,wid in enumerate(pf_wells):
    truth,pf=rg[i]; sub=grp.get(wid)
    if sub is None or len(sub)!=len(truth): continue
    if not np.allclose(sub["target"].to_numpy(np.float64)+sub["last_known_tvt"].to_numpy(np.float64),truth,atol=1e-3): continue
    pos_l.append(sub.index.to_numpy()); pres_l.append(pf["pf_scale_12"]-truth)
pos=np.concatenate(pos_l); p_res=np.concatenate(pres_l)
rng=np.random.RandomState(42); uw=df["well"].unique().copy(); rng.shuffle(uw)
fo={w:i%5 for i,w in enumerate(uw)}; fold=np.array([fo[w] for w in df["well"].to_numpy()[pos]])
def blend_oof(g_all):
    g=g_all[pos]; gw=np.empty(len(g))
    for f in range(5):
        tr,va=fold!=f,fold==f; d=g[tr]-p_res[tr]; gw[va]=float(np.dot(g[tr],d)/np.dot(d,d))
    return rmse((1-gw)*g+gw*p_res), gw.mean()
b_new,w_new=blend_oof(gbm_resid); b_old,w_old=blend_oof(gbm_resid_b)
print(f"  banked GBM(222)  + PF: OOF={b_old:.4f}  w={w_old:.3f}")
print(f"  +UK    GBM(225)  + PF: OOF={b_new:.4f}  w={w_new:.3f}")
print(f"  delta (PF-blended OOF) = {b_new-b_old:+.4f}  -> predicts LB shift from 8.158")
print("BET5 PFBLEND DONE")
