"""
train_qm9.py
============
GIN on QM9 dipole moment. The DDG regularizer acts on PER-ATOM features over
the bond graph using the combinatorial graph Laplacian. QM9 has no faces, so
there is no cotangent option here, by design. Metric is MAE in Debye.

Examples
  python train_qm9.py --baseline ddg  --seed 0
  python train_qm9.py --baseline l2   --seed 0
  python train_qm9.py --baseline none --seed 0
  python train_qm9.py --baseline ddg  --epochs 3 --limit 2000 --seed 0   # fast smoke
"""
import os, argparse, random, json
import numpy as np, torch, torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from ddg.models import DDGGIN
from ddg.regularizer import DDGRegularizer
from ddg.trainer import Trainer
from ddg.mesh_construction import build_graph_laplacian
from ddg.utils import start_logging

DIPOLE_IDX = 0


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def mae(pred, target):
    return (pred.squeeze() - target[:, DIPOLE_IDX]).abs().mean().item()


class MAELoss(nn.Module):
    def forward(self, pred, target):
        return (pred.squeeze() - target[:, DIPOLE_IDX]).abs().mean()


def get_loaders(root, bs, nw, seed, limit=0):
    ds = QM9(root).shuffle()
    if limit > 0:
        ds = ds[:limit]
    N = len(ds); ntr = int(0.8 * N); nva = int(0.1 * N)
    return (DataLoader(ds[:ntr], bs, shuffle=True, num_workers=nw, drop_last=True),
            DataLoader(ds[ntr:ntr + nva], bs, num_workers=nw),
            DataLoader(ds[ntr + nva:], bs, num_workers=nw),
            ds[0].x.shape[1])


def make_ddg_fn(device):
    def ddg_fn(model, batch):
        X, edge_index, _ = model.get_node_latent()
        L, areas = build_graph_laplacian(edge_index, X.shape[0], device=device)
        return X, L, areas
    return ddg_fn


def main(a):
    run_dir = os.path.join(a.out_dir, f"{a.baseline}_seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    start_logging(run_dir)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | baseline={a.baseline} seed={a.seed} "
          f"epochs={a.epochs} limit={a.limit}")

    tr, va, te, in_dim = get_loaders(a.data_root, a.batch_size, a.num_workers, a.seed, a.limit)
    model = DDGGIN(in_dim=in_dim, hidden_dim=a.hidden_dim, out_dim=1,
                   num_layers=a.num_layers, dropout=a.dropout)
    use_ddg = a.baseline == "ddg"
    weight_decay = 1e-4 if a.baseline == "l2" else 0.0

    trainer = Trainer(model, MAELoss(), DDGRegularizer(a.lambda1, a.lambda2),
                      device, lr=a.lr, patience=a.patience, max_epochs=a.epochs, use_ddg=use_ddg)
    for pg in trainer.optimizer.param_groups:
        pg["weight_decay"] = weight_decay

    ddg_fn = make_ddg_fn(device) if use_ddg else None
    save_path = os.path.join(run_dir, "model_best.pth")
    hist = trainer.fit(tr, va, mae, "MAE", ddg_fn=ddg_fn, save_path=save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    test = trainer.eval_epoch(te, mae)

    res = {"test_mae_debye": test["metric"], "stability_score": None,
           "convergence_epoch": hist["convergence_epoch"],
           "mean_epoch_time_s": hist["mean_epoch_time"],
           "weighting": "combinatorial", "baseline": a.baseline,
           "lambda1": a.lambda1, "lambda2": a.lambda2, "seed": a.seed,
           "n_epochs_ran": len(hist["epoch_time"])}
    with open(os.path.join(run_dir, "qm9_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nFINAL " + json.dumps(res))
    print(f"MAE={test['metric']:.4f} D  conv={hist['convergence_epoch']}  "
          f"t/epoch={hist['mean_epoch_time']:.1f}s  ->  {run_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data/QM9")
    p.add_argument("--out_dir", default="./outputs/qm9")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda1", type=float, default=0.05)
    p.add_argument("--lambda2", type=float, default=0.01)
    p.add_argument("--limit", type=int, default=0, help=">0 truncates data for a fast smoke run")
    p.add_argument("--baseline", default="ddg", choices=["ddg", "l2", "none"])
    main(p.parse_args())
