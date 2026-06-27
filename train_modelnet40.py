"""
train_modelnet40.py
===================
PointNet++ on ModelNet40 with the DDG regularizer on the per-point SA1 map.
Every run: makes its own folder, tees the console log to train_log.txt, and
writes modelnet40_results.json with a FIXED schema. Do not edit outputs by hand.

Examples
  python train_modelnet40.py --weighting cotangent     --seed 0          # ours
  python train_modelnet40.py --weighting combinatorial --seed 0          # baseline
  python train_modelnet40.py --baseline l2      --seed 0
  python train_modelnet40.py --baseline dropout --seed 0
  python train_modelnet40.py --baseline none    --seed 0
  python train_modelnet40.py --weighting cotangent --epochs 3 --limit 200 --seed 0   # fast smoke
"""
import os, argparse, random, json
import numpy as np, torch, torch.nn as nn
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import (SamplePoints, NormalizeScale,
                                         RandomRotate, RandomJitter, Compose)
from torch_geometric.loader import DataLoader

from ddg.models import DDGPointNet2
from ddg.regularizer import DDGRegularizer, StabilityScore
from ddg.trainer import Trainer
from ddg.mesh_construction import build_knn_laplacian
from ddg.utils import start_logging

RESULT_SCHEMA = ["overall_accuracy", "stability_score", "convergence_epoch",
                 "mean_epoch_time_s", "weighting", "baseline",
                 "lambda1", "lambda2", "seed", "n_epochs_ran"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def accuracy(pred, target):
    return (pred.argmax(-1) == target).float().mean().item()


def get_loaders(root, bs, nw, limit=0):
    pre = Compose([SamplePoints(1024), NormalizeScale()])
    aug = Compose([RandomRotate(15, axis=1), RandomJitter(0.01)])
    tr = ModelNet(root, "40", True, pre_transform=pre, transform=aug)
    te = ModelNet(root, "40", False, pre_transform=pre)
    if limit > 0:
        tr = tr[:limit]; te = te[:max(limit // 4, bs)]
    return (DataLoader(tr, bs, shuffle=True, num_workers=nw, drop_last=True),
            DataLoader(te, bs, shuffle=False, num_workers=nw))


def make_ddg_fn(k, weighting, device):
    def ddg_fn(model, batch):
        X, pos, bvec = model.get_point_latent()
        L, areas = build_knn_laplacian(pos, bvec, k=k, weighting=weighting, device=device)
        return X, L, areas
    return ddg_fn


def main(a):
    run_dir = os.path.join(a.out_dir, f"{a.baseline}_{a.weighting}_seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    start_logging(run_dir)                       # tee console -> run_dir/train_log.txt
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | baseline={a.baseline} weighting={a.weighting} "
          f"seed={a.seed} epochs={a.epochs} limit={a.limit}")

    train_loader, val_loader = get_loaders(a.data_root, a.batch_size, a.num_workers, a.limit)
    model = DDGPointNet2(num_classes=40, dropout=a.dropout)
    use_ddg = a.baseline == "ddg"
    weight_decay = 1e-4 if a.baseline == "l2" else 0.0

    trainer = Trainer(model, nn.CrossEntropyLoss(), DDGRegularizer(a.lambda1, a.lambda2),
                      device, lr=a.lr, patience=a.patience, max_epochs=a.epochs, use_ddg=use_ddg)
    for pg in trainer.optimizer.param_groups:
        pg["weight_decay"] = weight_decay

    ddg_fn = make_ddg_fn(a.k, a.weighting, device) if use_ddg else None
    save_path = os.path.join(run_dir, "model_best.pth")
    hist = trainer.fit(train_loader, val_loader, accuracy, "Acc", ddg_fn=ddg_fn, save_path=save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    final = trainer.eval_epoch(val_loader, accuracy)
    stab = StabilityScore(0.02, 5).compute(
        model, next(iter(val_loader)), device, nn.CrossEntropyLoss()) if a.compute_stability else None

    res = {"overall_accuracy": final["metric"], "stability_score": stab,
           "convergence_epoch": hist["convergence_epoch"],
           "mean_epoch_time_s": hist["mean_epoch_time"],
           "weighting": a.weighting if use_ddg else "n/a", "baseline": a.baseline,
           "lambda1": a.lambda1, "lambda2": a.lambda2, "seed": a.seed,
           "n_epochs_ran": len(hist["epoch_time"])}
    with open(os.path.join(run_dir, "modelnet40_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nFINAL " + json.dumps(res))
    print(f"OA={final['metric']*100:.2f}%  stab={stab}  conv={hist['convergence_epoch']}  "
          f"t/epoch={hist['mean_epoch_time']:.1f}s  ->  {run_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data/ModelNet40")
    p.add_argument("--out_dir", default="./outputs/modelnet40")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--lambda1", type=float, default=0.05)
    p.add_argument("--lambda2", type=float, default=0.01)
    p.add_argument("--limit", type=int, default=0, help=">0 truncates data for a fast smoke run")
    p.add_argument("--weighting", default="cotangent", choices=["cotangent", "combinatorial"])
    p.add_argument("--baseline", default="ddg", choices=["ddg", "l2", "dropout", "none"])
    p.add_argument("--compute_stability", action="store_true", default=True)
    main(p.parse_args())
