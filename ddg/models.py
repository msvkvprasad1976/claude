"""
ddg/models.py  (CORRECTED)
==========================
Models now expose a PER-POINT (PointNet++) or PER-ATOM (GIN) latent feature
map plus the geometry needed to build a matching Laplacian. The DDG energy is
applied to that map, not to the pooled per-sample vector.

ModelNet40: regularize post-SA1 features x1 at points pos1 (batch1).
            fps uses random_start=False so the sampled point set is stable
            for a given input, which lets the Laplacian cache hit.
QM9       : regularize per-atom node features (before pooling) on the bond graph.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    PointNetConv, fps, radius, global_max_pool,
    GINConv, global_add_pool, BatchNorm,
)
from torch_geometric.nn import MLP as PyGMLP
from torch_geometric.data import Data
from typing import Optional, Tuple


class SAModule(nn.Module):
    def __init__(self, ratio, r, nn_channels):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(
            local_nn=PyGMLP(nn_channels, act="relu", norm="batch_norm"),
            global_nn=None, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio, random_start=False)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x_out = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        return x_out, pos[idx], batch[idx]


class GlobalSAModule(nn.Module):
    def __init__(self, nn_channels):
        super().__init__()
        self.nn = PyGMLP(nn_channels, act="relu", norm="batch_norm")

    def forward(self, x, pos, batch):
        x_in = torch.cat([x, pos], dim=-1)
        x_out = self.nn(x_in)
        x_global = global_max_pool(x_out, batch)
        pos_out = x_global.new_zeros((x_global.size(0), 3))
        batch_out = torch.arange(x_global.size(0), device=batch.device)
        return x_global, pos_out, batch_out


class DDGPointNet2(nn.Module):
    def __init__(self, num_classes: int = 40, dropout: float = 0.5):
        super().__init__()
        self.sa1 = SAModule(0.5, 0.2, [3 + 3, 64, 64, 128])
        self.sa2 = SAModule(0.25, 0.4, [128 + 3, 128, 128, 256])
        self.sa3 = GlobalSAModule([256 + 3, 256, 512, 1024])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(dropout))
        self.out_layer = nn.Linear(256, num_classes)
        # per-point regularization hook (set during forward)
        self._reg_x: Optional[torch.Tensor] = None
        self._reg_pos: Optional[torch.Tensor] = None
        self._reg_batch: Optional[torch.Tensor] = None

    def forward(self, data: Data) -> torch.Tensor:
        x0, pos0, batch0 = None, data.pos, data.batch
        x1, pos1, batch1 = self.sa1(x0, pos0, batch0)
        # expose the per-point feature map at the SA1 resolution
        self._reg_x, self._reg_pos, self._reg_batch = x1, pos1, batch1
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, _, _ = self.sa3(x2, pos2, batch2)
        h = self.classifier(x3)
        return self.out_layer(h)

    def get_point_latent(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._reg_x is None:
            raise RuntimeError("Call forward() before get_point_latent().")
        return self._reg_x, self._reg_pos, self._reg_batch


class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                            nn.ReLU(True), nn.Linear(out_dim, out_dim))
        self.conv = GINConv(mlp, train_eps=True)
        self.bn = BatchNorm(out_dim)
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.bn(out)
        return F.relu(out + self.res(x))


class DDGGIN(nn.Module):
    def __init__(self, in_dim=11, hidden_dim=256, out_dim=1,
                 num_layers=5, dropout=0.0):
        super().__init__()
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.gin_layers = nn.ModuleList(
            [GINLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(True))
        self.out_layer = nn.Linear(hidden_dim // 4, out_dim)
        self._reg_x: Optional[torch.Tensor] = None
        self._reg_edge: Optional[torch.Tensor] = None
        self._reg_batch: Optional[torch.Tensor] = None

    def forward(self, data: Data) -> torch.Tensor:
        x = self.atom_encoder(data.x.float())
        for gin in self.gin_layers:
            x = gin(x, data.edge_index)
        # expose per-atom features + bond graph for regularization
        self._reg_x, self._reg_edge, self._reg_batch = x, data.edge_index, data.batch
        x_graph = global_add_pool(x, data.batch)
        h = self.regressor(x_graph)
        return self.out_layer(h)

    def get_node_latent(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._reg_x is None:
            raise RuntimeError("Call forward() before get_node_latent().")
        return self._reg_x, self._reg_edge, self._reg_batch
