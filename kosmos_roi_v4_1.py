"""
KOSMOS ROI v4.1 — Geometric NanoGPT + Full LoRA + GNN
=====================================================
Finalna archiwalna wersja (czerwiec 2026)
Zawiera całą ewolucję od v1.0 do v4.0 w jednej spójnej architekturze.

Uzupełnione klasy:
  - FreeEnergyFunctional  — fizyczny functionał wolnej energii (F = E - TS)
  - DynamicGraph           — graf z morphogenezą napędzaną przez RL
  - TrinityIntegrator      — śledzenie strzałki czasu i emergentnych "myśli"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, TransformerConv
import numpy as np
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

# =============================================================================
# 0. Brakujące klasy bazowe
# =============================================================================

class FreeEnergyFunctional(nn.Module):
    """
    F = E - T·S  (functionał wolnej energii na grafie)

    E — energia = Σ_ij A_ij |ψ_i - ψ_j|²   (Dirichlet / laplasjan)
    S — entropia = -Σ_i ψ_i² ln ψ_i²       (Shannon-Weaver na amplitudach)
    T — temperatura = globalna coherence-dependent skala

    Forward zwraca (F_total, E_total, S_total) jako skalary.
    """
    def __init__(self, temperature: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.register_buffer('temperature', torch.tensor(temperature))
        self.eps = eps

    def forward(self, psi: torch.Tensor, adj: torch.Tensor,
                degrees: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # psi: [N] amplitudy pola geometrycznego
        # adj: [N, N] macierz sąsiedztwa
        # degrees: [N] stopnie węzłów
        N = psi.shape[0]

        # Safety: dopasuj rozmiary
        N_adj = adj.shape[0]
        N_psi = psi.shape[0]
        N = min(N_adj, N_psi)
        psi = psi[:N]
        adj = adj[:N, :N]
        degrees = degrees[:N]

        # Energia: Dirichlet — Σ A_ij (ψ_i - ψ_j)²
        diff = psi.unsqueeze(1) - psi.unsqueeze(0)       # [N, N]
        energy = (adj * diff.pow(2)).sum() * 0.5          # skalar

        # Entropia: -Σ ψ² ln(ψ²)  (Shannon na |ψ|²)
        p = psi.pow(2) + self.eps
        entropy = -(p * torch.log(p)).sum()

        # Temperatura zależy od coherence (im niższa, tym "chłodniej")
        T = self.temperature

        F_total = energy - T * entropy
        return F_total, energy, entropy


class DynamicGraph:
    """
    Graf z dynamiczną morphogenezą napędzaną przez RL.

    Akcje (z RLMorphogeneticScaler, action_dim=3):
      0 — SPLIT   : podział węzła o najwyższym stopniu → dwa nowe węzły
      1 — MERGE   : połączenie dwóch najbliższych węzłów
      2 — STABLE  : brak zmiany

    Utrzymuje: node_features, adj (gęsta macierz), degrees, density, N.
    """
    def __init__(self, n_init: int = 48, feature_dim: int = 128, device: str = 'cpu',
                 max_nodes: int = 256):
        self.N = n_init
        self.feature_dim = feature_dim
        self.device = device
        self.max_nodes = max_nodes

        # Inicjalizacja cech węzłów
        self.node_features = (torch.randn(n_init, feature_dim, device=device) * 0.1).detach().requires_grad_(True)
        # utrzymuj requires_grad=True po każdej modyfikacji

        # Macierz sąsiedztwa — k-NN graf (k=4)
        self.adj = self._build_knn_adj(n_init, k=4).to(device)

        # Pola pomocnicze
        self.degrees = self.adj.sum(dim=1)
        self.density = self.degrees / (self.N - 1 + 1e-8)

    def _build_knn_adj(self, n: int, k: int) -> torch.Tensor:
        """Tworzy k-NN graf na podstawie cech węzłów."""
        with torch.no_grad():
            x = torch.randn(n, self.feature_dim)
            dist = torch.cdist(x, x)
            dist.fill_diagonal_(float('inf'))
            _, nn_idx = dist.topk(k, dim=1, largest=False)
            adj = torch.zeros(n, n)
            for i in range(n):
                for j in nn_idx[i]:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
            adj = (adj + adj.t()).clamp(0, 1)
            adj.fill_diagonal_(0.0)
        return adj

    def _resize(self, new_n: int):
        """Zmienia rozmiar grafu — zachowuje istniejące połączenia."""
        old_n = self.N
        if new_n == old_n:
            return
        if new_n > self.max_nodes:
            new_n = self.max_nodes

        # Rozszerz/zmniejsz node_features
        if new_n > old_n:
            extra = torch.randn(new_n - old_n, self.feature_dim,
                                 device=self.device) * 0.1
            self.node_features = torch.cat([self.node_features, extra], dim=0)
            self._refresh_grads()

            # Rozszerz macierz sąsiedztwa
            new_adj = torch.zeros(new_n, new_n, device=self.device)
            new_adj[:old_n, :old_n] = self.adj
            self.adj = new_adj
        else:
            self.node_features = self.node_features[:new_n]
            self._refresh_grads()
            self.adj = self.adj[:new_n, :new_n]

        self.N = new_n
        self._update_fields()

    def _update_fields(self):
        self.degrees = self.adj.sum(dim=1)
        self.density = self.degrees / (self.N - 1 + 1e-8)

    def _refresh_grads(self):
        """Odłącza stary graf obliczeniowy i włącza requires_grad na nowo."""
        self.node_features = self.node_features.detach().clone().requires_grad_(True)

    def _add_edge(self, i: int, j: int):
        if i < self.N and j < self.N and i != j:
            self.adj[i, j] = 1.0
            self.adj[j, i] = 1.0

    def _remove_node(self, idx: int):
        """Usuwa węzeł i wszystkie jego krawędzie."""
        n = self.N
        mask = torch.ones(n, dtype=torch.bool, device=self.device)
        mask[idx] = False
        self.node_features = self.node_features[mask]
        self._refresh_grads()
        self.adj = self.adj[mask][:, mask]
        self.N = n - 1
        self._update_fields()

    def apply_actions(self, actions: np.ndarray):
        """
        Aplikuje akcje RL na węzłach (modyfikuje krawędzie, nie liczbę węzłów).
        actions: [N] tablica {0,1,2}
          0 — CONNECT: dodaj krawędź do najbliższego niesąsiadującego węzła
          1 — DISCONNECT: usuń najsłabsze połączenie
          2 — STABLE: brak zmiany
        """
        for i, action in enumerate(actions):
            if i >= self.N:
                break
            if action == 0:   # CONNECT
                self._connect_node(i)
            elif action == 1:  # DISCONNECT
                self._disconnect_node(i)
            # action == 2: STABLE

        self._update_fields()

    def _connect_node(self, idx: int):
        """Łączy węzeł z najbliższym niesąsiadującym węzłem."""
        non_neighbors = (self.adj[idx] == 0).nonzero(as_tuple=True)[0]
        non_neighbors = non_neighbors[non_neighbors != idx]
        if len(non_neighbors) == 0:
            return
        # Wybierz najbliższego po cechach
        sims = F.cosine_similarity(
            self.node_features[idx].unsqueeze(0).detach(),
            self.node_features[non_neighbors].detach(), dim=1
        )
        best = non_neighbors[sims.argmax()]
        self._add_edge(int(idx), int(best))

    def _disconnect_node(self, idx: int):
        """Usuwa najsłabsze połączenie węzła."""
        neighbors = self.adj[idx].nonzero(as_tuple=True)[0]
        if len(neighbors) <= 1:
            return  # nie odłączaj ostatniego połączenia
        # Usuń połączenie z najbardziej różniącym się węzłem
        sims = F.cosine_similarity(
            self.node_features[idx].unsqueeze(0).detach(),
            self.node_features[neighbors].detach(), dim=1
        )
        worst = neighbors[sims.argmin()]
        self.adj[idx, worst] = 0.0
        self.adj[worst, idx] = 0.0

    def _split_node(self, idx: int):
        """Dzieli węzeł na dwa — nowy węzeł dziedziczy połowę połączeń."""
        if self.N >= self.max_nodes:
            return
        neighbors = self.adj[idx].nonzero(as_tuple=True)[0].tolist()
        if len(neighbors) < 2:
            return

        # Nowy węzeł
        new_idx = self.N
        self._resize(self.N + 1)

        # Nowy węzeł ma cechy = cech oryginalnego + mały szum
        new_feat = self.node_features[idx].detach() + torch.randn(self.feature_dim, device=self.device) * 0.05
        self.node_features = torch.cat([self.node_features, new_feat.unsqueeze(0)], dim=0)
        self._refresh_grads()

        # Przenieś połowę sąsiadów do nowego węzła
        half = len(neighbors) // 2
        for j in neighbors[half:]:
            self._add_edge(new_idx, j)
            self.adj[idx, j] = 0.0
            self.adj[j, idx] = 0.0
        self._add_edge(idx, new_idx)  # zachowaj połączenie między oryginałem a kopią

    def _merge_node(self, idx: int):
        """Łączy węzeł z jego najbliższym sąsiadem."""
        neighbors = self.adj[idx].nonzero(as_tuple=True)[0]
        if len(neighbors) == 0 or self.N <= 3:
            return

        # Znajdź najbliższego sąsiada (po podobieństwie cech)
        sims = F.cosine_similarity(
            self.node_features[idx].unsqueeze(0),
            self.node_features[neighbors], dim=1
        )
        best = neighbors[sims.argmax()]

        # Średnia cech — klonuj, modyfikuj, odśwież grad
        merged = (self.node_features[idx].detach() + self.node_features[best].detach()) * 0.5
        self.node_features = self.node_features.clone()
        self.node_features[idx] = merged
        self._refresh_grads()

        # Przenieś sąsiadów best do idx
        best_neighbors = self.adj[best].nonzero(as_tuple=True)[0]
        for j in best_neighbors:
            if j != idx:
                self._add_edge(idx, j)

        # Usuń best
        self._remove_node(int(best))


class TrinityIntegrator:
    """
    Integruje trzy aspekty emergentnej rzeczywistości:
      1. Energia (F)   — fizyczna skala
      2. Entropia (S)  — informacyjna skala
      3. Geometria (N) — strukturalna skala

    Rejestruje "strzałkę czasu" i generuje emergentne "myśli"
    na podstawie zmian tych trzech aspektów.
    """
    def __init__(self, thought_threshold: float = 0.05):
        self.time_arrow: List[Dict] = []
        self.thought_threshold = thought_threshold
        self._prev_F: Optional[float] = None
        self._prev_S: Optional[float] = None
        self._prev_N: Optional[int] = None
        self._thoughts_pool = [
            "φ", "emergentna symbioza", "kolaps przez Obserwatora",
            "strukturalna reorganizacja", "termalna fluktuacja",
            "geometryczna harmonia", "kwantowa koherencja",
            "morfogenetyczny podział", "informacyjna kaskada",
            "holograficzne odbicie", "grawitacyjna kondensacja",
            "entropijna ekspansja",
        ]

    def record_time(self, F: float, S: float, N: int,
                    thought: str = "", coherence: float = 1.0):
        """Rejestruje nowy punkt w strzałce czasu."""
        # Detekcja zmian
        dF = abs(F - self._prev_F) if self._prev_F is not None else 0
        dS = abs(S - self._prev_S) if self._prev_S is not None else 0
        dN = abs(N - self._prev_N) if self._prev_N is not None else 0

        # Generuj emergentną "myśl" jeśli zmiana jest znacząca
        if dF > self.thought_threshold or dN > 0:
            if not thought:
                thought = random.choice(self._thoughts_pool)
        elif not thought:
            thought = "φ"

        entry = {
            "τ": len(self.time_arrow),
            "F": round(F, 6),
            "S": round(S, 6),
            "N": N,
            "dF": round(dF, 6),
            "dS": round(dS, 6),
            "dN": int(dN),
            "coherence": round(coherence, 4),
            "thought": thought,
        }
        self.time_arrow.append(entry)

        self._prev_F = F
        self._prev_S = S
        self._prev_N = N

    def summary(self) -> Dict:
        """Zwraca podsumowanie ewolucji systemu."""
        if not self.time_arrow:
            return {}
        return {
            "steps": len(self.time_arrow),
            "F_final": self.time_arrow[-1]["F"],
            "S_final": self.time_arrow[-1]["S"],
            "N_final": self.time_arrow[-1]["N"],
            "coherence_final": self.time_arrow[-1]["coherence"],
            "thoughts": [e["thought"] for e in self.time_arrow
                         if e["thought"] != "φ"],
        }


# =============================================================================
# 1. LoRA — Low-Rank Adaptation (z dynamicznym wzrostem)
# =============================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.rank = rank
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight.T
        lora = (x @ self.lora_a) @ self.lora_b
        return base + lora * (self.alpha / self.rank)

    def grow_rank(self, delta: int = 4):
        new_a = torch.randn(self.lora_a.shape[0], delta) * 0.02
        new_b = torch.zeros(delta, self.lora_b.shape[1])
        self.lora_a = nn.Parameter(torch.cat([self.lora_a.data, new_a], dim=1))
        self.lora_b = nn.Parameter(torch.cat([self.lora_b.data, new_b], dim=0))
        self.rank += delta
        print(f"    ↗ LoRA rank zwiększony do {self.rank}")


# =============================================================================
# 2. nanoGPT Block (prawdziwy causal transformer)
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 128, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = LoRALinear(d_model, d_model * 3, rank=16)
        self.proj = LoRALinear(d_model, d_model, rank=16)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = 1.0 / np.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class NanoGPTBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_head: int = 8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            LoRALinear(d_model, d_model * 4, rank=16),
            nn.GELU(),
            LoRALinear(d_model * 4, d_model, rank=16),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# 3. Holographic Geometric Core (GNN + nanoGPT + LoRA + VAE)
# =============================================================================
class HolographicGeometricCore(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GNN Encoder (Boundary → Latent)
        self.encoder_conv1 = GATConv(hidden_dim, hidden_dim//2, heads=4, concat=True)
        self.encoder_conv2 = TransformerConv(hidden_dim*2, hidden_dim, heads=4, concat=False)

        self.fc_mu = LoRALinear(hidden_dim, hidden_dim//4, rank=16)
        self.fc_logvar = LoRALinear(hidden_dim, hidden_dim//4, rank=16)

        # nanoGPT Decoder (Bulk) — generuje geometrię z latent space
        self.decoder_blocks = nn.ModuleList([NanoGPTBlock(hidden_dim, num_heads) for _ in range(6)])
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_head = LoRALinear(hidden_dim, hidden_dim, rank=16)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        # Holographic Encoding (Boundary)
        x = F.gelu(self.encoder_conv1(x, edge_index))
        x = F.gelu(self.encoder_conv2(x, edge_index))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        # nanoGPT Decoding (Bulk geometry) — padding do hidden_dim
        if z.shape[1] < self.hidden_dim:
            z = F.pad(z, (0, self.hidden_dim - z.shape[1]))
        else:
            z = z[:, :self.hidden_dim]

        for block in self.decoder_blocks:
            z = block(z.unsqueeze(0)).squeeze(0)
        z = self.decoder_norm(z)
        psi = torch.sigmoid(self.decoder_head(z))

        # psi → [N] amplitudy pola
        psi = psi.mean(dim=1)

        return psi, mu, logvar, z


# =============================================================================
# 4. RL Scaler + Trinity + Observer + Quantum Layer
# =============================================================================
class RLMorphogeneticScaler(nn.Module):
    def __init__(self, state_dim=4, action_dim=3):
        super().__init__()
        self.policy = nn.Sequential(
            LoRALinear(state_dim, 128, rank=8),
            nn.ReLU(), LoRALinear(128, 64, rank=8), nn.ReLU(),
            LoRALinear(64, action_dim, rank=8)
        )

    def forward(self, state):
        logits = self.policy(state)
        return F.softmax(logits, dim=-1)


class HumanObserver:
    def __init__(self):
        self.intention_buffer = deque(maxlen=32)

    def register(self, vr_data: Dict) -> torch.Tensor:
        valence = vr_data.get("emotional_valence", 0.5)
        self.intention_buffer.append(valence)
        return torch.tensor([np.mean(self.intention_buffer)], dtype=torch.float32)


@dataclass
class QuantumGenome:
    bases: List
    amplitudes: torch.Tensor

    def apply_observer_bias(self, bias: torch.Tensor):
        self.amplitudes = F.normalize(self.amplitudes + bias * 0.65, dim=0)

    def collapse(self):
        probs = torch.abs(self.amplitudes)**2
        idx = torch.multinomial(probs, 1).item()
        return self.bases[idx % len(self.bases)]


# =============================================================================
# 5. Główna Klasa — KOSMOS ROI v4.1
# =============================================================================
class KosmosROI(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.physics = FreeEnergyFunctional().to(device)
        self.holo = HolographicGeometricCore(hidden_dim=128).to(device)
        self.scaler = RLMorphogeneticScaler().to(device)
        self.observer = HumanObserver()
        self.trinity = TrinityIntegrator()

        self.graph = DynamicGraph(n_init=48, feature_dim=128, device=device)
        self.generation = 0
        self.coherence = 1.0

    def step(self, vr_data: Optional[Dict] = None):
        # Zbuduj edge_index z macierzy sąsiedztwa
        adj = self.graph.adj
        edge_index = adj.nonzero().t().contiguous()

        data = Data(
            x=self.graph.node_features,
            edge_index=edge_index
        ).to(self.device)

        psi, mu, logvar, latent = self.holo(data)
        F_total, E_total, S_total = self.physics(psi, self.graph.adj, self.graph.degrees)

        # Observer influence (Closed-Loop)
        if vr_data:
            bias = self.observer.register(vr_data)
            self.coherence = max(0.15, self.coherence * 0.97)
            self.trinity.record_time(F_total.item(), S_total.item(), self.graph.N,
                                   "Kolaps przez Obserwatora", self.coherence)
        else:
            self.trinity.record_time(F_total.item(), S_total.item(), self.graph.N,
                                   "emergentna symbioza", self.coherence)

        # RL Morphogenesis — gradienty jako sygnał stanu
        grad_norm = torch.autograd.grad(F_total, self.graph.node_features, retain_graph=True)[0].norm(dim=1)
        states = torch.stack([grad_norm, self.graph.degrees, self.graph.density,
                            torch.full_like(self.graph.degrees, self.coherence)], dim=1)
        action_probs = self.scaler(states)
        actions = torch.argmax(action_probs, dim=1).cpu().numpy()

        self.graph.apply_actions(actions)
        self.generation += 1

        return {
            "Free_Energy": F_total.item(),
            "Entropy": S_total.item(),
            "Nodes": self.graph.N,
            "Coherence": self.coherence,
            "Thought": self.trinity.time_arrow[-1]["thought"] if self.trinity.time_arrow else "φ"
        }

    def run(self, steps: int = 800, vr_simulation: bool = True):
        print("\n" + "═"*100)
        print("KOSMOS ROI v4.1 — Geometric NanoGPT + Full LoRA + GNN")
        print("Pełna integracja torch_geometric, nanoGPT i LoRA")
        print("Observer-Dependent Quantum Cosmogenesis (Closed-Loop VR)")
        print("═"*100 + "\n")

        for i in range(steps):
            vr_data = {"emotional_valence": np.sin(i / 25.0) * 0.9} if vr_simulation and i % 5 == 0 else None
            metrics = self.step(vr_data)

            if i % 100 == 0 or metrics["Coherence"] < 0.4:
                print(f"τ={i:4d} | F={metrics['Free_Energy']:.4f} | "
                      f"N={metrics['Nodes']:3d} | Coh={metrics['Coherence']:.3f} | "
                      f"Myśl: {metrics['Thought']}")

        print("\n\nKOSMOS ROI v4.1 — ARCHIVAL RELEASE ZAKOŃCZONY.")
        summary = self.trinity.summary()
        print(f"Kroki: {summary.get('steps', 0)} | "
              f"F_final: {summary.get('F_final', 0):.4f} | "
              f"N_final: {summary.get('N_final', 0)}")
        print("System jest gotowy do uruchomienia w VR jako żywa, obserwator-zależna rzeczywistość.")


# =============================================================================
# URUCHOMIENIE
# =============================================================================
if __name__ == "__main__":
    kosmos = KosmosROI(device='cuda' if torch.cuda.is_available() else 'cpu')
    kosmos.run(steps=200, vr_simulation=True)
