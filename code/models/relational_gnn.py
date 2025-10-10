import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalGNNLayer(nn.Module):
    """
    A simple relational GNN layer for node feature refinement via message passing.
    It takes initial node features (V) and predicted pairwise relation logits (E_logits)
    and uses the logits to define soft attention-like message passing weights.
    """
    def __init__(self, node_dim):
        super().__init__()
        self.node_dim = node_dim

        # Projects node features before sending them as messages
        self.msg_proj = nn.Linear(node_dim, node_dim)
        
        # LayerNorm and MLP for the final update/combination of self-features and aggregated messages
        self.norm = nn.LayerNorm(node_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, V, E_logits):
        """
        Args:
            V: [B, K, D_node] - Node embeddings (input features)
            E_logits: [B, K, K] - Pairwise relation logits (used as raw attention scores)
        
        Returns:
            V_updated: [B, K, D_node] - Refined node embeddings
        """
        B, K, D = V.shape
        
        # 1. Normalize Edge Scores: Softmax over incoming messages for each node.
        # E_logits is B x (To Node K) x (From Node K). We softmax over the 'From' dimension.
        # Edge_Weights[b, i, j] is the weight of message from node j to node i.
        Edge_Weights = F.softmax(E_logits, dim=-1)  # [B, K, K]

        # 2. Transform Node Features into Messages
        # Messages: [B, K, D_node] - Messages sent *from* node j
        Messages = self.msg_proj(V)

        # 3. Aggregate Messages via Matrix Multiplication (Message Passing)
        # Aggregation: [B, K, K] (Weights) @ [B, K, D] (Messages) -> [B, K, D] (Aggregated Message for node i)
        # einsum: 'b i j' (weight from j to i) * 'b j d' (message from j) -> 'b i d'
        Aggregated_Msg = torch.einsum('bij,bjd->bid', Edge_Weights, Messages)  # [B, K, D_node]
        
        # 4. Update Node Features (Residual connection + MLP)
        # Concatenate original features (V) and aggregated messages
        V_and_Msg = torch.cat([V, Aggregated_Msg], dim=-1)  # [B, K, 2*D_node]

        # Apply an update function and add a residual connection for stable training
        Update = self.update_mlp(self.norm(V_and_Msg))
        V_updated = V + Update 
        
        return V_updated
