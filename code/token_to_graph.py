import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenToGraph(nn.Module):
    """
    Map token embeddings from CLIP into a graph representation.
    - sequence encoder (optional transformer)
    - slot-based node queries (like DETR)
    - node existence head
    - edge type head
    """

    def __init__(self, token_dim, hidden_dim=512,
                 num_slots=12, num_edge_types=6, n_transformer_layers=1):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.num_edge_types = num_edge_types

        self.proj_in = nn.Linear(token_dim, hidden_dim)

        # transformer encoder for token refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # slot queries
        self.slot_queries = nn.Parameter(torch.randn(num_slots, hidden_dim))

        # cross-attn to extract node features
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        # node head
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # edge head
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_edge_types)
        )

    def forward(self, token_embeddings):
        """
        token_embeddings: (B, T, D)
        Returns dict with:
          node_logits: (B, num_slots)
          node_feats: (B, num_slots, H)
          edge_logits: (B, num_slots, num_slots, num_edge_types)
        """
        x = self.proj_in(token_embeddings)
        x = self.encoder(x)

        B = x.size(0)
        slots = self.slot_queries.unsqueeze(0).expand(B, -1, -1)

        attn_out, _ = self.cross_attn(query=slots, key=x, value=x)
        node_feats = attn_out  # (B, K, H)

        node_logits = self.node_mlp(node_feats).squeeze(-1)  # (B, K)

        # edges
        f_i = node_feats.unsqueeze(2).expand(B, self.num_slots, self.num_slots, self.hidden_dim)
        f_j = node_feats.unsqueeze(1).expand(B, self.num_slots, self.num_slots, self.hidden_dim)
        edge_in = torch.cat([f_i, f_j], dim=-1)  # (B, K, K, 2H)

        edge_logits = self.edge_mlp(edge_in)  # (B, K, K, E)

        return {
            "node_logits": node_logits,
            "node_feats": node_feats,
            "edge_logits": edge_logits,
        }
