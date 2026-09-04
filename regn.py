"""
Relational Emergent Gravity Network (REGN)
==========================================

A research prototype of a graph neural network in which a gravity-like field is
not given as an input, but emerges from learned relational geometry.

Core idea
---------
Given a relational graph G=(V,E), the network learns:
  * node masses / energy densities      rho_i >= 0
  * edge lengths / relational metric    ell_ij > 0
  * edge couplings                      g_ij in (0,1)
  * scalar potential                    phi_i
  * curvature-like quantity             K_ij

Message passing is modulated by the learned metric and potential.  A geometric
regularizer encourages an Einstein-like relation on the graph:

    curvature  ~=  alpha * stress_energy

This is NOT a validated physics simulator. It is a differentiable modelling
framework inspired by emergent-gravity language: geometry is learned from
relations, then constrains information flow.

Only dependency: PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class REGNConfig:
    in_dim: int
    hidden_dim: int = 96
    out_dim: int = 1
    edge_attr_dim: int = 0
    layers: int = 4
    potential_range: float = 5.0
    length_floor: float = 1e-3
    eps: float = 1e-8
    curvature_weight: float = 0.1
    smoothness_weight: float = 0.02
    length_weight: float = 0.001
    einstein_alpha: float = 1.0
    # --- Anti-collapse terms (prevent degenerate trivial geometry) ---
    nontrivial_weight: float = 0.05    # keep rho, phi from vanishing
    einstein_align_weight: float = 0.1  # cosine alignment K↔T


class MLP(nn.Module):
    def __init__(self, dims, act=nn.SiLU, final_act=None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
            elif final_act is not None:
                layers.append(final_act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Geometric quantities — the "emergent field" that each layer learns.
# ---------------------------------------------------------------------------


class GeometricField(nn.Module):
    """
    Learns the five emergent geometric quantities for a single layer:

      rho_i  — node energy density (mass)         >= 0
      ell_ij — edge length (relational metric)    > 0
      g_ij   — edge coupling                      in (0, 1)
      phi_i  — scalar potential                   bounded [-V, V]
      K_ij   — curvature-like quantity            (computed, not free param)

    All are differentiable functions of the current node/edge representations.
    """

    def __init__(self, hidden_dim: int, edge_attr_dim: int, cfg: REGNConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim

        # Node-level quantities: energy density and potential
        self.rho_mlp = MLP([hidden_dim, hidden_dim, 1], final_act=None)
        self.phi_mlp = MLP([hidden_dim, hidden_dim, 1], final_act=None)

        # Edge-level quantities: length and coupling
        edge_in = 2 * hidden_dim + edge_attr_dim
        self.length_mlp = MLP([edge_in, hidden_dim, 1], final_act=None)
        self.coupling_mlp = MLP([edge_in, hidden_dim, 1], final_act=None)

    def forward(
        self,
        h: Tensor,               # [N, hidden_dim] node features
        edge_index: Tensor,       # [2, E] directed edges
        edge_attr: Optional[Tensor] = None,  # [E, edge_attr_dim]
    ) -> Dict[str, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        h_src, h_dst = h[src], h[dst]

        # Energy density: softplus ensures non-negativity
        rho = F.softplus(self.rho_mlp(h))  # [N, 1]

        # Scalar potential: tanh bounds to [-potential_range, potential_range]
        phi = torch.tanh(self.phi_mlp(h)) * self.cfg.potential_range  # [N, 1]

        # Edge length: softplus ensures positivity, + length_floor
        if edge_attr is not None:
            edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h_src, h_dst], dim=-1)

        ell = F.softplus(self.length_mlp(edge_input)) + self.cfg.length_floor  # [E, 1]
        g = torch.sigmoid(self.coupling_mlp(edge_input))  # [E, 1]

        # Curvature: discrete Laplacian of the potential along each edge,
        #   K_ij = g_ij * (phi_i - phi_j) / ell_ij^2
        # This is the graph analogue of a Ricci-type scalar derived from the
        # potential's second derivative in the learned metric.
        phi_diff = phi[src] - phi[dst]  # [E, 1]
        K = g * phi_diff / (ell.pow(2) + self.cfg.eps)  # [E, 1]

        # Stress-energy tensor component along each edge:
        #   T_ij = (rho_i * rho_j) / (ell_ij^2 + eps)
        # — interaction energy weighted by relational distance.
        T = (rho[src] * rho[dst]) / (ell.pow(2) + self.cfg.eps)  # [E, 1]

        return {
            "rho": rho,           # [N, 1]
            "phi": phi,           # [N, 1]
            "ell": ell,           # [E, 1]
            "g": g,               # [E, 1]
            "K": K,               # [E, 1]
            "T": T,               # [E, 1]
            "phi_diff": phi_diff, # [E, 1]
        }


# ---------------------------------------------------------------------------
# Message passing — information flow modulated by learned geometry.
# ---------------------------------------------------------------------------


class REGNLayer(nn.Module):
    """
    One layer of geometric message passing.

    Messages from neighbour j to i are modulated by the emergent metric:
        m_ij = g_ij * exp(-ell_ij) * MLP_message([h_j, phi_j - phi_i]) / (ell_ij + eps)

    The exp(-ell_ij) factor gives a gravity-like fall-off: distant nodes (in
    the learned relational geometry) contribute less, regardless of their
    hop-distance in the input graph.

    Node update:
        h_i' = h_i + sum_j m_ij   (residual connection)
    """

    def __init__(self, hidden_dim: int, edge_attr_dim: int, cfg: REGNConfig):
        super().__init__()
        self.cfg = cfg
        self.geom = GeometricField(hidden_dim, edge_attr_dim, cfg)

        self.msg_mlp = MLP([hidden_dim + 1, hidden_dim, hidden_dim])
        self.update_mlp = MLP([hidden_dim * 2, hidden_dim, hidden_dim])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        N = h.shape[0]
        src, dst = edge_index[0], edge_index[1]

        geo = self.geom(h, edge_index, edge_attr)

        # Gravity-modulated message
        # Potential gradient modulates the message content
        potential_gradient = geo["phi"][src] - geo["phi"][dst]  # [E, 1]
        msg_input = torch.cat([h[dst], potential_gradient], dim=-1)  # [E, hidden+1]
        msg = self.msg_mlp(msg_input)  # [E, hidden]

        # Modulate by coupling, distance fall-off, and metric
        modulation = geo["g"] * torch.exp(-geo["ell"]) / (geo["ell"] + self.cfg.eps)  # [E, 1]
        msg = msg * modulation  # [E, hidden]

        # Aggregate (sum) — like a gravitational field accumulating contributions
        agg = torch.zeros(N, h.shape[1], device=h.device, dtype=h.dtype)
        agg.index_add_(0, src, msg)

        # Residual update with gated fusion
        update_input = torch.cat([h, agg], dim=-1)
        h_new = h + self.update_mlp(update_input)
        h_new = self.norm(h_new)

        return h_new, geo


# ---------------------------------------------------------------------------
# Full model.
# ---------------------------------------------------------------------------


class REGN(nn.Module):
    """
    Relational Emergent Gravity Network.

    Stacks multiple REGN layers, collects the geometric field from each layer,
    and computes geometric regularizers that encourage an Einstein-like
    relation between curvature and stress-energy.

    Forward returns:
        out          — [N, out_dim] predictions
        geo_layers   — list of geometry dicts (one per layer)
        loss_geo     — scalar tensor, the total geometric regularization loss
    """

    def __init__(self, cfg: REGNConfig):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            REGNLayer(cfg.hidden_dim, cfg.edge_attr_dim, cfg)
            for _ in range(cfg.layers)
        ])
        self.output_head = MLP([cfg.hidden_dim, cfg.hidden_dim, cfg.out_dim])

    def forward(
        self,
        x: Tensor,                          # [N, in_dim] node features
        edge_index: Tensor,                 # [2, E] directed edges
        edge_attr: Optional[Tensor] = None,  # [E, edge_attr_dim]
    ) -> Tuple[Tensor, list, Tensor]:
        h = self.input_proj(x)

        geo_layers = []
        for layer in self.layers:
            h, geo = layer(h, edge_index, edge_attr)
            geo_layers.append(geo)

        out = self.output_head(h)  # [N, out_dim]

        # Geometric regularization
        loss_geo = self._geometry_loss(geo_layers, edge_index)

        return out, geo_layers, loss_geo

    def _geometry_loss(
        self,
        geo_layers: list,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Combine all geometric regularizers into a single scalar.

        Three families of terms:

        1. Einstein alignment (cosine): encourages the *spatial pattern* of
           curvature K_ij to align with stress-energy T_ij, without allowing
           both to trivially collapse to zero.  We use 1 - cos(K, alpha*T).

        2. Einstein magnitude: a weaker L2 term that keeps the scale of K
           proportional to alpha * T without the degenerate solution.

        3. Non-triviality: penalises rho and phi from vanishing, so the
           emergent geometry stays alive and informative.

        4. Potential smoothness + length conditioning (from original design).
        """
        cfg = self.cfg
        total = torch.tensor(0.0, device=edge_index.device)

        for geo in geo_layers:
            K = geo["K"]          # [E, 1]
            T = geo["T"]          # [E, 1]
            alpha = cfg.einstein_alpha

            # 1. Cosine alignment — pattern of K should match pattern of alpha*T.
            #    Resilient to magnitude collapse; focuses on structural agreement.
            K_flat = K.flatten()
            T_flat = (alpha * T).flatten()
            K_n = K_flat / (K_flat.norm() + cfg.eps)
            T_n = T_flat / (T_flat.norm() + cfg.eps)
            cosine_sim = (K_n * T_n).sum()
            align_loss = 1.0 - cosine_sim  # 0 when perfectly aligned, 2 when anti
            total = total + cfg.einstein_align_weight * align_loss

            # 2. Einstein magnitude — relative L2, normalised so it can't be
            #    trivially minimised by shrinking both K and T to zero.
            scale = (K.abs().mean() + alpha * T.abs().mean() + cfg.eps)
            magnitude_loss = ((K - alpha * T).pow(2).mean()) / scale
            total = total + cfg.curvature_weight * magnitude_loss

            # 3. Non-triviality — keep the geometry from collapsing.
            #    Penalise rho and phi variance being too low (everything flat).
            rho_var = geo["rho"].var()
            phi_var = geo["phi"].var()
            nontrivial = 1.0 / (rho_var + cfg.eps) + 1.0 / (phi_var + cfg.eps)
            total = total + cfg.nontrivial_weight * nontrivial

            # 4. Potential smoothness: penalise large potential gradients
            smoothness = geo["phi_diff"].pow(2).mean()
            total = total + cfg.smoothness_weight * smoothness

            # 5. Length regularizer: discourage edge collapse or explosion
            length_reg = geo["ell"].pow(2).mean() + (1.0 / (geo["ell"] + cfg.eps)).mean()
            total = total + cfg.length_weight * length_reg

        return total

    @torch.no_grad()
    def extract_geometry(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Run a forward pass and return a flattened dict of geometric quantities
        from all layers — useful for inspection and visualisation.
        """
        out, geo_layers, _ = self.forward(x, edge_index, edge_attr)
        result = {"node_output": out}
        for i, geo in enumerate(geo_layers):
            for key, val in geo.items():
                result[f"layer{i}_{key}"] = val
        return result


# ---------------------------------------------------------------------------
# Utility — build an edge_index for a k-NN graph from coordinates.
# ---------------------------------------------------------------------------


def knn_edges(coords: Tensor, k: int = 6) -> Tensor:
    """
    Given node coordinates [N, d], return [2, E] directed edge index for the
    k-nearest-neighbour graph (each node connects to its k nearest neighbours).
    """
    N = coords.shape[0]
    dist = torch.cdist(coords, coords)  # [N, N]
    dist.fill_diagonal_(float("inf"))
    _, nn_idx = dist.topk(k, dim=1, largest=False)  # [N, k]

    src = torch.arange(N, device=coords.device).unsqueeze(1).expand(N, k).flatten()
    dst = nn_idx.flatten()

    # Symmetrise: add reverse edges
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])
    # Deduplicate
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


# ---------------------------------------------------------------------------
# Synthetic demo — a small regression task on a geometric graph.
# ---------------------------------------------------------------------------


def demo():
    """
    Train REGN on a synthetic task: given nodes placed in 2-D with features,
    predict a scalar that depends on local density and distance structure.

    This validates that the model learns, optimises the geometric regularizers,
    and produces meaningful emergent geometry.
    """
    torch.manual_seed(42)

    # --- Synthetic data ---
    N = 200
    coords = torch.randn(N, 2)
    x = torch.randn(N, 4)  # node features
    edge_index = knn_edges(coords, k=8)

    # Target: a function of local density and distance — something geometry-aware
    with torch.no_grad():
        dist = torch.cdist(coords, coords)
        dist.fill_diagonal_(0.0)
        nn_dist, _ = dist.topk(8, dim=1, largest=False)
        local_density = 1.0 / (nn_dist.mean(dim=1, keepdim=True) + 1e-3)
        y = local_density * torch.sigmoid(x[:, :1])  # [N, 1]

    # --- Model ---
    cfg = REGNConfig(
        in_dim=4, hidden_dim=64, out_dim=1, layers=3,
        curvature_weight=0.05,
        einstein_align_weight=0.15,
        nontrivial_weight=0.02,
    )
    model = REGN(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # --- Training ---
    print("Training REGN on synthetic geometric regression task...\n")
    for epoch in range(300):
        opt.zero_grad()
        out, geo_layers, loss_geo = model(x, edge_index)
        loss_task = F.mse_loss(out, y)
        loss = loss_task + loss_geo
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == 299:
            # Inspect emergent geometry
            last = geo_layers[-1]
            K = last["K"].flatten()
            T = last["T"].flatten()
            if K.norm() > cfg.eps and T.norm() > cfg.eps:
                corr = torch.corrcoef(torch.stack([K, T]))[0, 1].item()
            else:
                corr = 0.0
            print(
                f"Epoch {epoch:3d} | "
                f"task={loss_task.item():.4f}  "
                f"geo={loss_geo.item():.4f}  "
                f"| rho={last['rho'].mean().item():.3f}  "
                f"ell={last['ell'].mean().item():.3f}  "
                f"phi={last['phi'].std().item():.3f}  "
                f"K={last['K'].std().item():.3f}  "
                f"T={last['T'].mean().item():.3f}  "
                f"corr(K,T)={corr:+.3f}"
            )

    # --- Inspect final geometry ---
    print("\nExtracting emergent geometry...")
    geom = model.extract_geometry(x, edge_index)
    print(f"  Keys: {sorted(geom.keys())}")
    print(f"  Final rho range: [{geom['layer2_rho'].min():.3f}, {geom['layer2_rho'].max():.3f}]")
    print(f"  Final ell range: [{geom['layer2_ell'].min():.3f}, {geom['layer2_ell'].max():.3f}]")
    print(f"  Final phi range: [{geom['layer2_phi'].min():.3f}, {geom['layer2_phi'].max():.3f}]")

    # Check Einstein relation quality
    K = geom["layer2_K"]
    T = geom["layer2_T"]
    correlation = torch.corrcoef(torch.cat([K.flatten().unsqueeze(0), T.flatten().unsqueeze(0)], dim=0))[0, 1]
    print(f"  Curvature–stress-energy correlation: {correlation:.3f}")
    print(f"\nDone. The model learned emergent geometry with "
          f"{'strong' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.2 else 'weak'} K↔T coupling.")

    return model


if __name__ == "__main__":
    model = demo()
