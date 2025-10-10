# code/models/token_to_scenegraph.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.relational_gnn import RelationalGNNLayer

class TokenToSceneGraph(nn.Module):
    """
    Learnable replacement for deterministic parser τ(·).
    Maps CLIP token embeddings w -> (S, V, E_logits)
      - w: [B, N, D] token embeddings
      - token_mask: [B, N] (1 for real tokens)
    Outputs:
      - S: [B, N, K] soft token->node assignment (per-token distribution over K nodes)
      - V: [B, K, D_node] node (object) embeddings
      - E_logits: [B, K, K] pairwise relation logits (directed)
    Usage:
      S, V, E_logits = token_to_sg(w, token_mask=token_mask)
    """
    def __init__(
        self,
        token_dim: int,
        K: int = 8,
        node_dim: int | None = None,
        heads: int = 4,
        hidden_dim: int = 512,
        use_bilinear: bool = False,
        num_gnn_layers: int = 1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.K = K
        self.node_dim = node_dim or token_dim
        self.heads = heads
        self.use_bilinear = use_bilinear
        self.num_gnn_layers = num_gnn_layers

        # learnable node prototypes (queries)
        self.node_queries = nn.Parameter(torch.randn(1, K, self.node_dim) * 0.02)

        # simple linear projections for query/key/value
        self.q_proj = nn.Linear(self.node_dim, self.node_dim)
        self.k_proj = nn.Linear(self.token_dim, self.node_dim)
        self.v_proj = nn.Linear(self.token_dim, self.node_dim)

        assert self.node_dim % self.heads == 0, "node_dim must be divisible by heads"
        self.scale = (self.node_dim // self.heads) ** -0.5

        # edge scoring
        if use_bilinear:
            self.edge_bilinear = nn.Bilinear(self.node_dim, self.node_dim, 1)
        else:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.node_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        # small scalar gate to optionally fuse prototype and aggregated tokens
        self.fuse_gate = nn.Parameter(torch.tensor(0.0))

        if self.num_gnn_layers > 0:
            self.gnn_layers = nn.ModuleList([
                RelationalGNNLayer(self.node_dim) for _ in range(self.num_gnn_layers)
            ])
        else:
            self.gnn_layers = None

    def forward(self, w: torch.Tensor, token_mask: torch.Tensor | None = None, discrete: bool = False, eps: float = 1e-8):
        """
        Args:
            w: [B, N, D_token]
            token_mask: [B, N] (1 for real token)
            discrete: if True, convert S to hard one-hot (argmax) assignments (useful for diagnostics)
        Returns:
            S: [B, N, K]
            V: [B, K, D_node]
            E_logits: [B, K, K]
        """
        B, N, D = w.shape
        device = w.device

        # node queries replicated for batch
        q = self.node_queries.expand(B, -1, -1)            # [B, K, Dn]
        Q = self.q_proj(q)                                 # [B, K, Dn]
        Kt = self.k_proj(w)                                # [B, N, Dn]
        Vt = self.v_proj(w)                                # [B, N, Dn]

        Dn = self.node_dim
        h = self.heads
        # reshape for multi-head dot-product
        Qh = Q.view(B, self.K, h, Dn // h).transpose(1,2)  # [B, h, K, Dh]
        Kh = Kt.view(B, N, h, Dn // h).transpose(1,2)     # [B, h, N, Dh]

        # compute per-head scores: [B, h, K, N] -> sum heads later
        # use einsum for clarity
        per_head = torch.einsum('b h k d, b h n d -> b h k n', Qh, Kh)  # [B, h, K, N]
        # average over heads (or sum then scale)
        scores = per_head.sum(dim=1) * self.scale  # [B, K, N]

        # mask padding tokens if provided
        if token_mask is not None:
            # token_mask: 1 real, 0 pad -> mask where pad==1
            pad_mask = (token_mask == 0).unsqueeze(1)  # [B, 1, N]
            scores = scores.masked_fill(pad_mask, torch.finfo(scores.dtype).min)


        # convert [B,K,N] -> [B,N,K] for per-token distribution
        scores_t = scores.transpose(1, 2)  # [B, N, K]
        S = F.softmax(scores_t, dim=-1)    # per-token distribution over K

        if discrete:
            idx = S.argmax(dim=-1)  # [B, N]
            S_hard = torch.zeros_like(S).scatter_(-1, idx.unsqueeze(-1), 1.0)
            S = S_hard

        # node embeddings: weighted sum of token values (Vt)
        # V = S^T @ Vt  -> using einsum
        V = torch.einsum('b n k, b n d -> b k d', S, Vt)  # [B, K, Dn]

        # fuse with prototype queries (small gate)
        gate = torch.sigmoid(self.fuse_gate)
        V = (1 - gate) * V + gate * q

# ---------------------------------------------
        # --- NEW: Apply Relational GNN Message Passing ---
        # ---------------------------------------------
        if self.gnn_layers is not None:
            # 1. Compute initial pairwise edge logits (E_logits) from unfined V
            Vi = V.unsqueeze(2).expand(B, self.K, self.K, Dn)  # [B,K,K,D]
            Vj = V.unsqueeze(1).expand(B, self.K, self.K, Dn)
            pair = torch.cat([Vi, Vj], dim=-1)  # [B,K,K,2D]

            if self.use_bilinear:
                flat_i = Vi.reshape(B * self.K * self.K, Dn)
                flat_j = Vj.reshape(B * self.K * self.K, Dn)
                e_flat = self.edge_bilinear(flat_i, flat_j).view(B, self.K, self.K)
                E_logits = e_flat
            else:
                e_flat = self.edge_mlp(pair)  # [B, K, K, 1]
                E_logits = e_flat.squeeze(-1) # [B, K, K]

            # 2. Pass V and E_logits through the GNN layers
            V_refined = V
            for gnn_layer in self.gnn_layers:
                V_refined = gnn_layer(V_refined, E_logits)

            # Update V to V_refined for the final output
            V = V_refined
        
        # ---------------------------------------------
        # --- Old code for final E_logits (kept for compatibility, may use V_refined) ---
        # ---------------------------------------------
        # Recalculate E_logits using the final V (which might be V_refined)
        Vi = V.unsqueeze(2).expand(B, self.K, self.K, Dn)
        Vj = V.unsqueeze(1).expand(B, self.K, self.K, Dn)
        pair = torch.cat([Vi, Vj], dim=-1)

        if self.use_bilinear:
            # ... (rest of bilinear calculation using flat_i, flat_j from V)
            flat_i = Vi.reshape(B * self.K * self.K, Dn)
            flat_j = Vj.reshape(B * self.K * self.K, Dn)
            e_flat = self.edge_bilinear(flat_i, flat_j).view(B, self.K, self.K)
            E_logits = e_flat
        else:
            e_flat = self.edge_mlp(pair)  # [B, K, K, 1]
            E_logits = e_flat.squeeze(-1)

        # ---------------------------------------------
        
        return S, V, E_logits
