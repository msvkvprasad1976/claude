"""
aggregate_results.py
====================
Reads the raw per-run JSON files the trainers wrote and builds the manuscript
tables. No hand entry. Tables are computed as mean +/- std across seeds.

  python aggregate_results.py --modelnet_dir ./outputs/modelnet40 --qm9_dir ./outputs/qm9

Outputs (in --out):
  table_modelnet40.csv   OA and stability per config, mean +/- std over seeds
  table_qm9.csv          MAE per config, mean +/- std over seeds
  table_overhead.csv     mean epoch time and net wall-clock vs baseline
  summary.md             same tables in readable form
"""
import os, glob, json, argparse
import numpy as np, pandas as pd


def load_runs(folder, fname):
    rows = []
    for p in glob.glob(os.path.join(folder, "**", fname), recursive=True):
        try:
            rows.append(json.load(open(p)))
        except Exception as e:
            print("skip", p, e)
    return rows


def agg(rows, metric_key, group_keys):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    for vals, g in df.groupby(group_keys):
        vals = vals if isinstance(vals, tuple) else (vals,)
        rec = dict(zip(group_keys, vals))
        rec["n_seeds"] = len(g)
        rec[f"{metric_key}_mean"] = g[metric_key].mean()
        rec[f"{metric_key}_std"] = g[metric_key].std(ddof=0)
        if "stability_score" in g and g["stability_score"].notna().any():
            rec["stability_mean"] = g["stability_score"].mean()
            rec["stability_std"] = g["stability_score"].std(ddof=0)
        if "convergence_epoch" in g:
            rec["conv_epoch_mean"] = g["convergence_epoch"].mean()
        if "mean_epoch_time_s" in g:
            rec["epoch_time_mean_s"] = g["mean_epoch_time_s"].mean()
        out.append(rec)
    return pd.DataFrame(out)


def config_label(r):
    if r["baseline"] != "ddg":
        return r["baseline"]
    return f"DDG-{r.get('weighting','?')}"


def main(a):
    os.makedirs(a.out, exist_ok=True)
    md = ["# Aggregated results (machine-generated from raw run JSONs)\n"]

    mn = load_runs(a.modelnet_dir, "modelnet40_results.json")
    if mn:
        for r in mn:
            r["config"] = config_label(r)
            r["overall_accuracy_pct"] = r["overall_accuracy"] * 100.0
        t = agg(mn, "overall_accuracy_pct", ["config"])
        t = t.sort_values("overall_accuracy_pct_mean")
        t.to_csv(os.path.join(a.out, "table_modelnet40.csv"), index=False)
        md.append("## ModelNet40 (PointNet++), OA %\n" + t.to_markdown(index=False) + "\n")

        ov = agg(mn, "mean_epoch_time_s", ["config"])
        base = ov.loc[ov["config"] == "none", "mean_epoch_time_s_mean"]
        if len(base):
            b = float(base.iloc[0])
            ov["per_epoch_overhead_pct"] = (ov["mean_epoch_time_s_mean"] / b - 1) * 100
        ov.to_csv(os.path.join(a.out, "table_overhead.csv"), index=False)
        md.append("## Per-epoch overhead vs no-reg baseline\n" + ov.to_markdown(index=False) + "\n")

    qm = load_runs(a.qm9_dir, "qm9_results.json")
    if qm:
        for r in qm:
            r["config"] = config_label(r)
        t = agg(qm, "test_mae_debye", ["config"])
        t = t.sort_values("test_mae_debye_mean")
        t.to_csv(os.path.join(a.out, "table_qm9.csv"), index=False)
        md.append("## QM9 (GIN), MAE Debye, lower better\n" + t.to_markdown(index=False) + "\n")

    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print("\nWrote tables to", a.out)
    if not mn and not qm:
        print("\nNO RUN JSONs FOUND. Train first, then aggregate.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modelnet_dir", default="./outputs/modelnet40")
    p.add_argument("--qm9_dir", default="./outputs/qm9")
    p.add_argument("--out", default="./outputs/tables")
    main(p.parse_args())
