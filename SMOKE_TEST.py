"""
SMOKE_TEST.py
Tests the dataset-agnostic core (Laplacian builders + regularizer + hook flow)
without torch_geometric or a GPU. Run this first after dropping the package in,
to confirm the wiring before launching full training.
"""
import torch
from types import SimpleNamespace
from ddg.regularizer import DDGRegularizer
from ddg.mesh_construction import build_knn_laplacian, build_graph_laplacian

ok = True
def check(name, cond):
    global ok; ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

# ---- 1. ModelNet40-style hook: per-point latent + k-NN Laplacian ----
print("ModelNet40 path (per-point SA1 latent):")
sizes = [200, 180, 220]                       # points per object after SA1
pos = torch.cat([torch.randn(n, 3) for n in sizes])
bvec = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
M = pos.shape[0]
X = torch.randn(M, 128, requires_grad=True)   # SA1 feature dim
reg = DDGRegularizer(0.05, 0.01)

for w in ["cotangent", "combinatorial"]:
    L, areas = build_knn_laplacian(pos, bvec, k=20, weighting=w)
    check(f"{w}: L rows match X rows ({L.shape[0]}=={M})", L.shape[0] == M)
    check(f"{w}: areas align ({areas.shape[0]}=={M})", areas.shape[0] == M)
    total, e_d, e_w = reg(X, torch.tensor(2.0), L, areas)
    total.backward(retain_graph=True)
    check(f"{w}: grad flows to latent", X.grad is not None and torch.isfinite(X.grad).all())
    check(f"{w}: energies finite (E_D={e_d:.2f}, E_W={e_w:.2f})",
          torch.isfinite(e_d) and torch.isfinite(e_w))
    X.grad = None

# ---- 2. QM9-style hook: per-atom latent + bond-graph Laplacian ----
print("\nQM9 path (per-atom latent, combinatorial bond-graph Laplacian):")
n_atoms = 60
ei = torch.randint(0, n_atoms, (2, 140))      # random bond graph
Xa = torch.randn(n_atoms, 64, requires_grad=True)
L, areas = build_graph_laplacian(ei, n_atoms)
check(f"L rows match atoms ({L.shape[0]}=={n_atoms})", L.shape[0] == n_atoms)
total, e_d, e_w = reg(Xa, torch.tensor(0.5), L, areas)
total.backward()
check("grad flows to atom latent", Xa.grad is not None and torch.isfinite(Xa.grad).all())
check(f"energies finite (E_D={e_d:.2f}, E_W={e_w:.2f})",
      torch.isfinite(e_d) and torch.isfinite(e_w))

# ---- 3. Mock the full ddg_fn(model, batch) flow used by the trainer ----
print("\nTrainer hook flow (mocked model + batch):")
class MockPN2:
    def get_point_latent(self): return X, pos, bvec
def mn_ddg_fn(model, batch):
    Xp, p, b = model.get_point_latent()
    L, a = build_knn_laplacian(p, b, k=20, weighting="cotangent")
    return Xp, L, a
Xp, L, a = mn_ddg_fn(MockPN2(), None)
total, _, _ = reg(Xp, torch.tensor(1.0), L, a)
check("ModelNet40 ddg_fn returns trainable loss", total.requires_grad)

class MockGIN:
    def get_node_latent(self): return Xa2, ei, None
Xa2 = torch.randn(n_atoms, 64, requires_grad=True)
def qm9_ddg_fn(model, batch):
    Xn, e, _ = model.get_node_latent()
    L, a = build_graph_laplacian(e, Xn.shape[0])
    return Xn, L, a
Xn, L, a = qm9_ddg_fn(MockGIN(), None)
total, _, _ = reg(Xn, torch.tensor(1.0), L, a)
check("QM9 ddg_fn returns trainable loss", total.requires_grad)

# ---- 4. Smoothness sanity: constant features have ~0 Dirichlet energy ----
print("\nSmoothness sanity:")
Xc = torch.ones(M, 128)
L, areas = build_knn_laplacian(pos, bvec, k=20, weighting="cotangent")
_, e_d_const, _ = reg(Xc, torch.tensor(0.0), L, areas)
_, e_d_rand, _ = reg(torch.randn(M, 128), torch.tensor(0.0), L, areas)
check(f"constant X has lower Dirichlet than random ({e_d_const:.3f} < {e_d_rand:.3f})",
      abs(e_d_const) < abs(e_d_rand))

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
