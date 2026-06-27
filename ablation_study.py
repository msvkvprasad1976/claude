"""
ablation_study.py
=================
Runs the energy-term ablation, the lambda1 sensitivity sweep, and the
cotangent-vs-combinatorial comparison by launching real training subprocesses
(5 seeds each), then aggregates the per-run JSONs into a CSV grouped by config.

  python ablation_study.py --study ablation         --weighting cotangent
  python ablation_study.py --study sensitivity       --weighting cotangent
  python ablation_study.py --study baseline_compare

Each underlying run writes its own folder + train_log.txt (proof of the run).
"""
import os, glob, json, argparse, subprocess
import numpy as np, pandas as pd

ABLATION = [        # (lambda1, lambda2, ddg, weighting)
    (0.0,  0.0,  False, "none"),
    (0.05, 0.0,  True,  "cotangent"),
    (0.0,  0.01, True,  "cotangent"),
    (0.05, 0.01, True,  "cotangent")]
SENSITIVITY = [(l, 0.01, True, "cotangent") for l in (0.001, 0.01, 0.05, 0.10, 0.50)]
BASELINE_COMPARE = [
    (0.05, 0.01, True, "combinatorial"),
    (0.05, 0.01, True, "cotangent")]


def run(cfgs, data_root, out_dir, n_runs, epochs, weighting_default):
    os.makedirs(out_dir, exist_ok=True)
    for (l1, l2, ddg, wt) in cfgs:
        w = wt if wt in ("cotangent", "combinatorial") else weighting_default
        for s in range(n_runs):
            cmd = ["python", "train_modelnet40.py",
                   "--data_root", data_root, "--out_dir", out_dir,
                   "--lambda1", str(l1), "--lambda2", str(l2),
                   "--epochs", str(epochs), "--seed", str(s),
                   "--baseline", "ddg" if ddg else "none",
                   "--weighting", w]
            print("RUN:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True, timeout=21600)
            except subprocess.CalledProcessError as e:
                print("  ERROR:", e)


def aggregate(out_dir):
    rows = [json.load(open(p)) for p in
            glob.glob(os.path.join(out_dir, "**", "modelnet40_results.json"), recursive=True)]
    if not rows:
        print("No results found in", out_dir); return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["acc_pct"] = df["overall_accuracy"] * 100.0
    g = df.groupby(["lambda1", "lambda2", "weighting", "baseline"])
    out = g.agg(acc_mean=("acc_pct", "mean"),
                acc_std=("acc_pct", lambda x: x.std(ddof=0)),
                stab_mean=("stability_score", "mean"),
                conv_mean=("convergence_epoch", "mean"),
                n_seeds=("seed", "nunique")).reset_index()
    return out.sort_values(["lambda2", "lambda1"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", default="ablation",
                   choices=["ablation", "sensitivity", "baseline_compare"])
    p.add_argument("--weighting", default="cotangent",
                   choices=["cotangent", "combinatorial"])
    p.add_argument("--data_root", default="./data/ModelNet40")
    p.add_argument("--out_dir", default="./outputs/ablation")
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--aggregate_only", action="store_true",
                   help="skip training, just aggregate existing run folders")
    a = p.parse_args()
    cfgs = {"ablation": ABLATION, "sensitivity": SENSITIVITY,
            "baseline_compare": BASELINE_COMPARE}[a.study]
    study_dir = os.path.join(a.out_dir, a.study)
    if not a.aggregate_only:
        run(cfgs, a.data_root, study_dir, a.n_runs, a.epochs, a.weighting)
    df = aggregate(study_dir)
    if len(df):
        out_csv = os.path.join(a.out_dir, f"{a.study}.csv")
        df.to_csv(out_csv, index=False)
        print(df.to_string(index=False)); print("\nSaved:", out_csv)
