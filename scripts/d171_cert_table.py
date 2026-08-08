"""D171 TASK 2 — the new certified table beside D158's, plus the D134 control
hash against data/capstone_pergame_D158.csv.  Read-only on the DB (never opens it)."""
import sys, csv, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nbapred.threads; nbapred.threads.pin(1)
import numpy as np
LN2 = 0.6931471805599453
ROOT = Path(__file__).resolve().parents[1]

def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r["y"] = int(r["y"]); r["p_us"] = float(r["p_us"]); r["p_mkt"] = float(r["p_mkt"])
        r["n_out_home"] = int(r["n_out_home"]); r["n_out_away"] = int(r["n_out_away"])
    return rows

def ll(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-15, 1-1e-15)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

new = load(ROOT/"data/capstone_pergame.csv")
old = load(ROOT/"data/capstone_pergame_D158.csv")

D158 = {  # registered in docs/DECISIONS.md D158 §4
 "2021-22": (1228, 0.63053, 0.60429, 0.02623, 29.52),
 "2022-23": (1230, 0.63385, 0.62437, 0.00948, 13.78),
 "2023-24": (1230, 0.59906, 0.58086, 0.01820, 16.21),
 "2024-25": (1230, 0.58857, 0.58155, 0.00703,  6.30),
 "2025-26": (1230, 0.58553, 0.57114, 0.01438, 11.79),
 "POOLED":  (6148, 0.60750, 0.59244, 0.01506, 14.95)}

def table(rows):
    out = {}
    for s in ["2021-22","2022-23","2023-24","2024-25","2025-26"]:
        sub = [r for r in rows if r["season"] == s]
        u = ll([r["y"] for r in sub], [r["p_us"] for r in sub])
        m = ll([r["y"] for r in sub], [r["p_mkt"] for r in sub])
        out[s] = (len(sub), u, m, u-m, 100*(u-m)/(LN2-m),
                  np.mean([r["n_out_home"] for r in sub]+[r["n_out_away"] for r in sub]))
    u = ll([r["y"] for r in rows], [r["p_us"] for r in rows])
    m = ll([r["y"] for r in rows], [r["p_mkt"] for r in rows])
    out["POOLED"] = (len(rows), u, m, u-m, 100*(u-m)/(LN2-m),
                     np.mean([r["n_out_home"] for r in rows]+[r["n_out_away"] for r in rows]))
    return out

NEW, OLD = table(new), table(old)
print("="*104)
print("D171 CERTIFIED TABLE (T2-HONEST, TANK_SEASON_FLOOR=2020-21, Clippers fix)  vs  D158")
print("="*104)
print(f"{'season':<9}{'n':>6}{'ll_us':>10}{'ll_mkt':>10}{'raw':>10}{'norm':>9} | "
      f"{'D158 ll_us':>11}{'D158 norm':>10}{'d(norm)':>10}{'d(ll_us)':>10}{'outs/tm':>9}")
for s in ["2021-22","2022-23","2023-24","2024-25","2025-26","POOLED"]:
    n,u,m,raw,nm,ot = NEW[s]; dn,du,dm,draw,dnm = D158[s]
    print(f"{s:<9}{n:>6}{u:>10.5f}{m:>10.5f}{raw:>+10.5f}{nm:>8.2f}% | "
          f"{du:>11.5f}{dnm:>9.2f}%{nm-dnm:>+9.2f}{u-du:>+10.5f}{ot:>9.3f}")
print()
print("re-derived D158 numbers from capstone_pergame_D158.csv (integrity check):")
for s in ["2021-22","2022-23","2023-24","2024-25","2025-26","POOLED"]:
    n,u,m,raw,nm,ot = OLD[s]; dn,du,dm,draw,dnm = D158[s]
    flag = "OK" if abs(u-du) < 1e-5 and abs(nm-dnm) < 0.011 else "MISMATCH"
    print(f"   {s:<9} n={n:<6} ll_us={u:.5f} (reg {du:.5f}) norm={nm:.2f}% (reg {dnm:.2f}%) outs/tm={ot:.3f}  {flag}")

# ---- D134 control hash -----------------------------------------------------
kn = {(r["season"], r["game_id"]): r for r in new}
ko = {(r["season"], r["game_id"]): r for r in old}
both = set(kn) & set(ko)
dp  = np.array([kn[k]["p_us"]  - ko[k]["p_us"]  for k in both])
dpm = np.array([kn[k]["p_mkt"] - ko[k]["p_mkt"] for k in both])
moved = int((np.abs(dp) > 1e-9).sum())
print()
print("="*104); print("D134 CONTROL HASH — data/capstone_pergame.csv (D171) vs data/capstone_pergame_D158.csv")
print("="*104)
print(f"  md5 new  : {hashlib.md5(open(ROOT/'data/capstone_pergame.csv','rb').read()).hexdigest()}")
print(f"  md5 D158 : {hashlib.md5(open(ROOT/'data/capstone_pergame_D158.csv','rb').read()).hexdigest()}")
print(f"  games matched      : {len(both)}/{len(new)} new, {len(ko)} old"
      f"   new-only={len(set(kn)-set(ko))} old-only={len(set(ko)-set(kn))}")
print(f"  p_us moved         : {moved} games ({100*moved/len(both):.2f}%)")
print(f"  max|dp_us|         : {np.abs(dp).max():.6f}")
print(f"  mean|dp_us|        : {np.abs(dp).mean():.6f}")
print(f"  p_mkt max|dp|      : {np.abs(dpm).max():.1e}   (must be 0)")
on = np.mean([r["n_out_home"] for r in new]+[r["n_out_away"] for r in new])
oo = np.mean([r["n_out_home"] for r in old]+[r["n_out_away"] for r in old])
print(f"  mean OUT/team-game : D171 {on:.3f}  vs D158 {oo:.3f}  ({on-oo:+.3f})")
# per-season movement
print("  per-season p_us movement:")
for s in ["2021-22","2022-23","2023-24","2024-25","2025-26"]:
    ks = [k for k in both if k[0] == s]
    d = np.array([kn[k]["p_us"] - ko[k]["p_us"] for k in ks])
    mo = np.array([kn[k]["n_out_home"]+kn[k]["n_out_away"] for k in ks], float)
    oo2 = np.array([ko[k]["n_out_home"]+ko[k]["n_out_away"] for k in ks], float)
    print(f"    {s}  n={len(ks):<5} moved={int((np.abs(d)>1e-9).sum()):<5} "
          f"({100*(np.abs(d)>1e-9).mean():5.1f}%)  max|dp|={np.abs(d).max():.6f}  "
          f"mean|dp|={np.abs(d).mean():.6f}  outs/game {oo2.mean():.3f}->{mo.mean():.3f}")

json.dump({"new": {k: list(map(float, v)) for k, v in NEW.items()},
           "d158_rederived": {k: list(map(float, v)) for k, v in OLD.items()},
           "d158_registered": {k: list(v) for k, v in D158.items()},
           "control_hash": {"matched": len(both), "moved": moved,
                            "max_dp": float(np.abs(dp).max()),
                            "mean_dp": float(np.abs(dp).mean()),
                            "pmkt_max_dp": float(np.abs(dpm).max()),
                            "outs_new": float(on), "outs_old": float(oo)}},
          open(ROOT/"data/d171_cert_table.json","w"), indent=1)
