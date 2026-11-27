"""Exact McNemar tests between checkpoints on the same 100 CPU scenes (scene seed 10000).

Usage: python scripts/paired_tests.py <repo root>; writes results/action_free/noise_sweep/paired_tests.json.
Each entry: [episodes where only the first run reached the stage, only the second, exact two-sided p]."""
import json
import sys

import csv
from scipy.stats import binomtest

root = sys.argv[1]
files = {
    "ZG": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_zg_idm_N100.csv",
    "A100": f"{root}/results/action_free/policy_N100/failure_analysis/metrics/episodes_armA_N100.csv",
    "B100": f"{root}/results/action_free/policy_N100/failure_analysis/metrics/episodes_armB_N100.csv",
    "XY25": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_xy025_N100.csv",
    "XY50": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_xy050_N100.csv",
    "SH04": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_xy_shrink04_N100.csv",
    "AR09": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_xy025_ar09_N100.csv",
    "SH07": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_xy_shrink07_N100.csv",
    "PCNN": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_pos_cnn_N100.csv",
    "FUS": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_fusion_N100.csv",
    "FTG": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_fusion_truegrip_N100.csv",
    "FGW": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_fusion_gripw_N100.csv",
    "PREC": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_posrec_truegrip_N100.csv",
    "XYZ": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_xyz_cnnlevel_N100.csv",
    "ESH": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_fusion_errshuf_N100.csv",
    "FTG8": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_fusion_truegrip_s8_N100.csv",
    "SAR": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_xyz_shrink_ar045_N100.csv",
    "XYZ2": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_noise_xyz_cnnlevel_r2_N100.csv",
    "U1s1": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed1_upper.csv",
    "FTGs1": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed1_fusion_truegrip.csv",
    "XYZs1": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed1_xyz_cnnlevel.csv",
    "SARs1": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed1_xyz_shrink_ar045.csv",
    "AR09s1": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed1_xy025_ar09.csv",
    "SH07s1": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed1_xy_shrink07.csv",
    "A100s1": f"{root}/results/action_free/policy_N100/failure_analysis/metrics/episodes_seed1_armA_N100.csv",
    "B100s1": f"{root}/results/action_free/policy_N100/failure_analysis/metrics/episodes_seed1_armB_N100.csv",
    "XY25s1": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed1_xy025.csv",
    "U1s2": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed2_upper.csv",
    "XY25s2": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed2_xy025.csv",
    "AR09s2": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed2_xy025_ar09.csv",
    "SH07s2": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed2_xy_shrink07.csv",
    "SH04s2": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed2_xy_shrink04.csv",
    "FTGs2": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed2_fusion_truegrip.csv",
    "SH04s1": f"{root}/results/action_free/noise_sweep/failure_analysis/metrics/episodes_seed1_xy_shrink04.csv",
    "U1000": f"{root}/results/action_free/upper_N1000/failure_analysis/metrics/episodes_upper_N1000.csv",
    "I60": f"{root}/results/action_free/idm_budget/failure_analysis/metrics/episodes_seed0_idm60k.csv",
    "IMO": f"{root}/results/action_free/idm_budget/failure_analysis/metrics/episodes_seed0_idm_morin.csv",
    "I60s1": f"{root}/results/action_free/idm_budget/failure_analysis/metrics/episodes_seed1_idm60k.csv",
    "IMOs1": f"{root}/results/action_free/idm_budget/failure_analysis/metrics/episodes_seed1_idm_morin.csv",
    "I60WLs2": f"{root}/results/action_free/idm_budget/failure_analysis/metrics/episodes_seed2_idm60k_wl.csv",
    "IMOs2": f"{root}/results/action_free/idm_budget/failure_analysis/metrics/episodes_seed2_idm_morin.csv",
}
stages = ["approached", "ever_grasped", "stable_lift", "reached_target", "success_once", "final_success"]
def load(f):
    rows = sorted(csv.DictReader(open(f)), key=lambda r: int(r["episode"]))
    return {s: [r[s] == "True" for r in rows] for s in stages}
d = {k: load(v) for k, v in files.items()}
counts = {k: {s: sum(df[s]) for s in stages} for k, df in d.items()}
print(json.dumps(counts, indent=1))
out = {"counts": counts}
for a, b in [("ZG", "A100"), ("ZG", "B100"), ("U1000", "ZG"), ("XY25", "A100"), ("XY25", "B100"), ("U1000", "XY25"), ("XY50", "XY25"), ("U1000", "XY50"), ("ZG", "XY25"), ("U1000", "SH04"), ("XY25", "SH04"), ("SH04", "B100"), ("A100", "SH04"), ("U1000", "AR09"), ("XY25", "AR09"), ("AR09", "B100"), ("U1000", "SH07"), ("SH07", "SH04"), ("SH07", "B100"), ("AR09", "SH07"), ("PCNN", "B100"), ("FUS", "B100"), ("PCNN", "A100"), ("FUS", "A100"), ("U1000", "PCNN"), ("U1000", "FUS"), ("FUS", "PCNN"), ("XY25", "FUS"), ("FTG", "FUS"), ("FGW", "FUS"), ("FTG", "FGW"), ("U1000", "FTG"), ("U1000", "FGW"), ("FGW", "B100"), ("U1000", "PREC"), ("U1000", "XYZ"), ("PREC", "FTG"), ("XYZ", "FTG"), ("XY25", "XYZ"), ("PREC", "XYZ"), ("XYZ", "PCNN"), ("PREC", "AR09"), ("PREC", "B100"), ("FTG8", "ESH"), ("FTG8", "FTG"), ("U1000", "FTG8"), ("U1000", "ESH"), ("XYZ", "FTG8"), ("XYZ", "ESH"), ("PREC", "FTG8"), ("XYZ", "XYZ2"), ("SAR", "FTG8"), ("XYZ", "SAR"), ("XYZ2", "SAR"), ("U1000", "SAR"), ("U1000", "XYZ2"), ("SAR", "FTG"), ("SAR", "ESH"), ("U1000", "U1s1"), ("FTGs1", "FTG"), ("FTGs1", "FTG8"), ("U1s1", "FTGs1"), ("U1000", "FTGs1"), ("XYZ", "FTGs1"), ("FTGs1", "SAR"), ("XYZs1", "SARs1"), ("XYZ", "XYZs1"), ("SAR", "SARs1"), ("U1s1", "XYZs1"), ("U1s1", "SARs1"), ("FTGs1", "XYZs1"), ("FTGs1", "SARs1"), ("AR09", "AR09s1"), ("SH07", "SH07s1"), ("U1s1", "AR09s1"), ("U1s1", "SH07s1"), ("XYZs1", "AR09s1"), ("XYZs1", "SH07s1"), ("AR09s1", "SH07s1"), ("SARs1", "AR09s1"), ("SARs1", "SH07s1"), ("A100", "A100s1"), ("B100", "B100s1"), ("A100s1", "B100s1"), ("U1s1", "A100s1"), ("U1s1", "B100s1"), ("FTGs1", "B100s1"), ("SH07s1", "B100s1"), ("XY25", "XY25s1"), ("SH04", "SH04s1"), ("U1s1", "XY25s1"), ("U1s1", "SH04s1"), ("XY25s1", "SH04s1"), ("XY25s1", "AR09s1"), ("XY25s1", "SH07s1"), ("XY25s1", "B100s1"), ("SH07s1", "SH04s1"), ("U1000", "I60"), ("U1000", "IMO"), ("IMO", "I60"), ("I60", "SH07"), ("IMO", "SH07"), ("I60", "B100"), ("IMO", "B100"), ("I60", "A100"), ("IMO", "A100"), ("U1s1", "IMOs1"), ("IMOs1", "SH07s1"), ("IMOs1", "B100s1"), ("IMOs1", "A100s1"), ("IMO", "IMOs1"), ("U1000", "U1s2"), ("U1s1", "U1s2"), ("XY25", "XY25s2"), ("XY25s1", "XY25s2"), ("U1s2", "XY25s2"), ("U1s2", "I60WLs2"), ("U1s2", "IMOs2"), ("IMOs2", "I60WLs2"), ("I60", "I60WLs2"), ("IMO", "IMOs2"), ("IMOs1", "IMOs2"), ("XY25s2", "I60WLs2"), ("XY25s2", "IMOs2"), ("SH07", "SH07s2"), ("SH07s1", "SH07s2"), ("AR09", "AR09s2"), ("AR09s1", "AR09s2"), ("SH04", "SH04s2"), ("SH04s1", "SH04s2"), ("FTGs1", "FTGs2"), ("FTG8", "FTGs2"), ("U1s2", "AR09s2"), ("U1s2", "SH07s2"), ("U1s2", "SH04s2"), ("U1s2", "FTGs2"), ("XY25s2", "AR09s2"), ("XY25s2", "SH07s2"), ("XY25s2", "SH04s2"), ("XY25s2", "FTGs2"), ("AR09s2", "SH07s2"), ("SH07s2", "SH04s2"), ("IMOs2", "SH07s2"), ("I60WLs2", "SH07s2"), ("FTGs2", "IMOs2"), ("FTGs2", "I60WLs2")]:
    for s in stages:
        x, y = d[a][s], d[b][s]
        n10 = sum(i and not j for i, j in zip(x, y)); n01 = sum(j and not i for i, j in zip(x, y))
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        out[f"{a}_vs_{b}_{s}"] = [n10, n01, p]
        print(f"{a} vs {b} {s}: {n10} vs {n01}, p={p:.3g}")
json.dump(out, open(f"{root}/results/action_free/noise_sweep/paired_tests.json", "w"), indent=1)
