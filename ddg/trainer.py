"""
ddg/trainer.py  (CORRECTED)
===========================
The DDG step now pulls the model's per-point / per-atom latent and a Laplacian
built to match it, via a dataset-specific `ddg_fn(model, batch) -> (X, L, areas)`.
Energy and task loss share the same computation graph.

Per-epoch wall-clock is printed. Use it to report the real overhead in
Section IX. The old fixed 46 percent figure must be replaced by the value
measured here, because the regularized layer and Laplacian changed.
"""
import os, time, torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from typing import Dict, List, Optional, Callable

from .regularizer import DDGRegularizer


class Trainer:
    def __init__(self, model, criterion, regularizer, device,
                 lr=1e-3, patience=20, max_epochs=200,
                 lr_factor=0.5, lr_step=50, use_ddg=True):
        self.model = model.to(device)
        self.criterion = criterion
        self.regularizer = regularizer
        self.device = device
        self.patience = patience
        self.max_epochs = max_epochs
        self.use_ddg = use_ddg
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        # paper Section V-B: lr halved every 50 epochs
        self.scheduler = StepLR(self.optimizer, step_size=lr_step, gamma=lr_factor)
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_metric": [],
            "e_dirichlet": [], "e_willmore": [], "epoch_time": []}

    def train_epoch(self, loader, metric_fn, ddg_fn=None):
        self.model.train()
        tl = tm = ed = ew = 0.0; nb = 0
        for batch in tqdm(loader, desc="  Train", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            task_loss = self.criterion(pred, batch.y)
            if self.use_ddg and ddg_fn is not None:
                X, L, areas = ddg_fn(self.model, batch)
                total, e_d, e_w = self.regularizer(X, task_loss, L, areas)
                ed += e_d.item(); ew += e_w.item()
            else:
                total = task_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            tl += total.item(); tm += metric_fn(pred.detach(), batch.y); nb += 1
        return {"loss": tl / nb, "metric": tm / nb,
                "e_dirichlet": ed / nb, "e_willmore": ew / nb}

    @torch.no_grad()
    def eval_epoch(self, loader, metric_fn):
        self.model.eval()
        tl = tm = 0.0; nb = 0
        for batch in tqdm(loader, desc="  Eval ", leave=False):
            batch = batch.to(self.device)
            pred = self.model(batch)
            tl += self.criterion(pred, batch.y).item()
            tm += metric_fn(pred, batch.y); nb += 1
        return {"loss": tl / nb, "metric": tm / nb}

    def fit(self, train_loader, val_loader, metric_fn, metric_name="metric",
            ddg_fn=None, save_path=None):
        best = float("inf"); wait = 0; conv = self.max_epochs
        print(f"\n{'='*60}\n  DDG={'ON' if self.use_ddg else 'OFF'}  "
              f"λ1={self.regularizer.lambda1:.3f} λ2={self.regularizer.lambda2:.3f}\n{'='*60}")
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            tr = self.train_epoch(train_loader, metric_fn, ddg_fn)
            va = self.eval_epoch(val_loader, metric_fn)
            self.scheduler.step()
            dt = time.time() - t0
            for k, v in [("train_loss", tr["loss"]), ("val_loss", va["loss"]),
                         ("val_metric", va["metric"]), ("e_dirichlet", tr["e_dirichlet"]),
                         ("e_willmore", tr["e_willmore"]), ("epoch_time", dt)]:
                self.history[k].append(v)
            print(f"Epoch {epoch:3d}/{self.max_epochs} | TrainLoss={tr['loss']:.4f} "
                  f"| Val{metric_name}={va['metric']:.4f} | E_D={tr['e_dirichlet']:.3f} "
                  f"| E_W={tr['e_willmore']:.3f} | t={dt:.1f}s")
            if va["loss"] < best:
                best = va["loss"]; wait = 0; conv = epoch
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch}. "
                          f"Best val loss {best:.4f} (epoch {conv}).")
                    break
        self.history["convergence_epoch"] = conv
        self.history["mean_epoch_time"] = float(
            sum(self.history["epoch_time"]) / len(self.history["epoch_time"]))
        return self.history
