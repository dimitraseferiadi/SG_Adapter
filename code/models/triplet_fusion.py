import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalTripletFusion(nn.Module):
    """
    Synthesizes a new set of K contextual node features by performing a weighted 
    aggregation of all KxK potential triplet features, thereby forcing the 
    feature to contain explicit Subject-Relation-Object context.
    The output dimension matches the input dimension (D_node).
    """
    def __init__(self, node_dim: int, edge_logit_dim: int = 1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_logit_dim = edge_logit_dim
        
        # Concatenated Dimension: 2 * D_node (for Subj + Obj) + Edge_Logit_Dim
        triplet_input_dim = 2 * node_dim + edge_logit_dim
        
        # MLP to map the raw concatenated triplet feature back down to D_node (3080)
        self.triplet_map_mlp = nn.Sequential(
            nn.Linear(triplet_input_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        
        self.norm = nn.LayerNorm(node_dim)
        
        # Learnable gate to interpolate between the GNN-refined node feature (V') 
        # and the newly calculated triplet-contextualized feature (V_fused_triplet)
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, V_refined: torch.Tensor, E_logits: torch.Tensor) -> torch.Tensor:
        """
        V_refined: [B, K, D_node] (Output from GNN layer - Subject/Object features)
        E_logits: [B, K, K] (Output from TokenToSceneGraph - Relation weights/scores)
        
        Returns:
        V_fused: [B, K, D_node] (The final Triplet-Contextualized Feature to be used by the adapter)
        """
        B, K, D = V_refined.shape
        
        # 1. Generate KxK Triplet Candidate Features (Subject i -> Object j)
        V_i = V_refined.unsqueeze(2)           # Subject features [B, K_sub, 1, D]
        V_j = V_refined.unsqueeze(1)           # Object features [B, 1, K_obj, D]
        E_logits_expanded = E_logits.unsqueeze(-1) # Edge features [B, K_sub, K_obj, 1]
        
        # Concatenate Subject, Object, and Edge features to form all KxK triplets
        Triplet_Feature_Candidates = torch.cat([
            V_i.expand(-1, -1, K, -1),            # Subject (i)
            V_j.expand(-1, K, -1, -1),            # Object (j)
            E_logits_expanded
        ], dim=-1) # [B, K_sub, K_obj, 2*D + 1]

        # 2. Map Triplet Candidates down to the D_node dimension (still KxK candidates)
        T_flat = Triplet_Feature_Candidates.view(B * K * K, -1)
        T_mapped_flat = self.triplet_map_mlp(T_flat)
        T_mapped = T_mapped_flat.view(B, K, K, D) # [B, K_sub, K_obj, D]

        # 3. Aggregate: Sum over the Subject dimension, weighted by soft attention.
        # We assume the feature is related to the Object (K_obj dimension).
        
        # Calculate soft attention weights over the Subject dimension (K_sub) for each Object (K_obj).
        # We softmax the E_logits transposed to normalize over all possible Subjects (K_sub) pointing to a single Object (K_obj).
        # Edge_Weights_Subj is [B, K_obj, K_sub]
        Edge_Weights_Subj = F.softmax(E_logits.transpose(1, 2), dim=-1) 
        
        # Weighted sum: V_fused_triplet[b, k_obj, d] = sum_{k_sub} [Weight[b, k_obj, k_sub] * T_mapped[b, k_sub, k_obj, d]]
        # Result: [B, K_obj, D] -> Contextualized Feature for each Object slot
        V_fused_triplet = torch.einsum('boj,bijd->bod', Edge_Weights_Subj, T_mapped) 
        
        # 4. Final Interpolation with GNN-Refined Feature
        gate = torch.sigmoid(self.fusion_gate)
        
        # Normalize the computed triplet feature before fusion (optional, but good practice)
        V_fused_triplet = self.norm(V_fused_triplet)
        
        # Interpolate the original GNN-refined node feature (V_refined) with the new triplet context
        V_fused = (1.0 - gate) * V_refined + gate * V_fused_triplet
        
        return V_fused