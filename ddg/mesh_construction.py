"""
ddg/mesh_construction.py  (CORRECTED)
=====================================
Builds Laplacians that are dimensionally aligned to a PER-POINT or PER-ATOM
feature matrix, assembled block-diagonal across a batch.

Two builders:
  build_knn_laplacian(pos, batch_vec, k, weighting)
      For point clouds. weighting = "cotangent" (geometric Laplace-Beltrami)
      or "combinatorial" (L = D - A, unit weights).
  build_graph_laplacian(edge_index, num_nodes)
      For molecular / bond graphs. Combinatorial L = D - A on given edges.

A small in-memory cache keyed on a hash of each sample's coordinates avoids
rebuilding identical geometries (helps evaluation and any un-augmented pass).
"""
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

_CACHE = {}
_CACHE_MAX = 20000


def _hash_pts(arr):
    return hash(np.round(arr, 4).tobytes())


def _knn_edges(pts, k):
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(pts)),
                            algorithm="kd_tree").fit(pts)
    _, idx = nbrs.kneighbors(pts)
    idx = idx[:, 1:]
    e = set()
    for i in range(pts.shape[0]):
        for j in idx[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            e.add((a, b))
    return np.array(sorted(e), dtype=np.int64) if e else np.zeros((0, 2), np.int64)


def _combinatorial_single(pts, k):
    N = pts.shape[0]
    edges = _knn_edges(pts, k)
    rows, cols, vals = [], [], []
    deg = np.zeros(N)
    for a, b in edges:
        rows += [a, b]; cols += [b, a]; vals += [-1.0, -1.0]
        deg[a] += 1; deg[b] += 1
    for i in range(N):
        rows.append(i); cols.append(i); vals.append(float(deg[i]))
    return (np.array(rows), np.array(cols),
            np.array(vals, np.float32), np.ones(N, np.float32))


def _cotangent_single(pts, k):
    N = pts.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(pts)
    _, idx = nbrs.kneighbors(pts); idx = idx[:, 1:]
    faces = set()
    for i in range(N):
        loc = np.concatenate([[i], idx[i]])
        P = pts[loc] - pts[loc].mean(0)
        try:
            _, _, Vt = np.linalg.svd(P, full_matrices=False)
            tri = Delaunay(P @ Vt[:2].T)
        except Exception:
            continue
        for s in tri.simplices:
            f = tuple(sorted(int(x) for x in loc[s]))
            if len(set(f)) == 3:
                faces.add(f)
    L = np.zeros((N, N), np.float64)
    areas = np.zeros(N, np.float64)

    def cot(u, v):
        c = float(np.dot(u, v)); s = float(np.linalg.norm(np.cross(u, v))) + 1e-8
        return c / s

    for (i, j, k3) in faces:
        vi, vj, vk = pts[i], pts[j], pts[k3]
        ck = cot(vi - vk, vj - vk)
        ci = cot(vj - vi, vk - vi)
        cj = cot(vi - vj, vk - vj)
        area = 0.5 * float(np.linalg.norm(np.cross(vj - vi, vk - vi)))
        for v in (i, j, k3):
            areas[v] += area / 3.0
        for (a, b, c) in [(i, j, ck), (j, k3, ci), (i, k3, cj)]:
            w = 0.5 * c
            L[a, b] -= w; L[b, a] -= w; L[a, a] += w; L[b, b] += w
    r, c = np.nonzero(L)
    return (r.astype(np.int64), c.astype(np.int64),
            L[r, c].astype(np.float32),
            np.clip(areas, 1e-8, None).astype(np.float32))


def build_knn_laplacian(pos, batch_vec, k=20, weighting="cotangent",
                        device="cpu", use_cache=True):
    """
    Block-diagonal Laplacian over all points in a batch, aligned to a
    per-point feature matrix of the same row count as `pos`.
    """
    pos_np = pos.detach().cpu().numpy().astype(np.float64)
    bv = batch_vec.detach().cpu().numpy()
    M = pos_np.shape[0]
    single = _cotangent_single if weighting == "cotangent" else _combinatorial_single
    rows, cols, vals = [], [], []
    areas = np.ones(M, np.float32)

    for s in np.unique(bv):
        sel = np.where(bv == s)[0]
        if sel.shape[0] < 4:
            continue
        sub = pos_np[sel]
        key = (weighting, k, _hash_pts(sub)) if use_cache else None
        if key is not None and key in _CACHE:
            r, c, v, a = _CACHE[key]
        else:
            r, c, v, a = single(sub, k)
            if key is not None and len(_CACHE) < _CACHE_MAX:
                _CACHE[key] = (r, c, v, a)
        off = int(sel[0])  # PyG batches are contiguous per sample
        rows.append(r + off); cols.append(c + off); vals.append(v)
        areas[sel] = a

    if not rows:
        idx = torch.zeros(2, 0, dtype=torch.long, device=device)
        L = torch.sparse_coo_tensor(idx, torch.zeros(0, device=device), (M, M))
        return L.coalesce(), torch.tensor(areas, device=device)

    R = torch.tensor(np.concatenate(rows), dtype=torch.long)
    C = torch.tensor(np.concatenate(cols), dtype=torch.long)
    V = torch.tensor(np.concatenate(vals), dtype=torch.float32)
    L = torch.sparse_coo_tensor(torch.stack([R, C]), V, (M, M)).coalesce().to(device)
    return L, torch.tensor(areas, device=device)


def build_graph_laplacian(edge_index, num_nodes, device="cpu"):
    """
    Combinatorial L = D - A from an explicit edge list (bond graph for QM9).
    Aligned to a per-atom feature matrix of row count num_nodes.
    """
    N = num_nodes
    row, col = edge_index[0], edge_index[1]
    ones = torch.ones(row.shape[0], device=device)
    deg = torch.zeros(N, device=device).scatter_add_(0, row, ones)
    all_row = torch.cat([row, torch.arange(N, device=device)])
    all_col = torch.cat([col, torch.arange(N, device=device)])
    all_val = torch.cat([-ones, deg])
    L = torch.sparse_coo_tensor(torch.stack([all_row, all_col]),
                                all_val, (N, N)).coalesce().to(device)
    areas = torch.ones(N, device=device)
    return L, areas
