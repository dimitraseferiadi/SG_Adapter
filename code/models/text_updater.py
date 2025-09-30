import math
import torch
import torch.nn as nn
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention


class RelationAttentionWithSelfAttention(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        query_dim,
        sg_emb_dim,
        n_heads,
        d_head,
    ):
        super().__init__()

        # project incoming CLIP token features into query_dim
        self.proj_in = nn.Linear(768, query_dim)

        # project node embeddings (V) into the same dimension as queries
        # Keep API consistent with prior code by using sg_emb_dim as input dim for linear
        self.linear = nn.Sequential(
            nn.Linear(sg_emb_dim, query_dim),
            nn.SiLU(),
        )

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.self_attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        # learnable scalars controlling residual contributions
        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_self", nn.Parameter(torch.tensor(0.0)))

    def forward(self, x, node_embeddings, token_node_assign=None, self_attention_mask=None):
        """
        Args:
            x: [B, N, C_in]  token embeddings (CLIP token features)
            node_embeddings: [B, K, sg_emb_dim]  learned object/node embeddings (V)
            token_node_assign: [B, N, K] soft token->node assignments S (optional)
            self_attention_mask: optional mask for self-attention
        Returns:
            x: [B, N, query_dim] updated token embeddings
        """
        # project tokens into query_dim
        x = self.proj_in(x)  # [B, N, query_dim]

        # project node embeddings into query_dim
        # node_embeddings may have shape [B, K, sg_emb_dim]
        sg_proj = self.linear(node_embeddings)  # [B, K, query_dim]

        # build attention bias from soft assignments if provided
        # Attention in diffusers expects attention_mask to be an additive mask to logits.
        # We provide log(S + eps) which will bias tokens toward their assigned nodes.
        bias = None
        if token_node_assign is not None:
            # token_node_assign: [B, N, K] -> bias: same shape
            bias = torch.log(token_node_assign + 1e-8)

        # cross-attention: tokens (queries) attend over node embeddings (keys/values)
        x = x + self.alpha_attn.tanh() * self.attn(
            self.norm1(x),
            encoder_hidden_states=self.norm1(sg_proj),
            attention_mask=bias,
        )

        # feed-forward residual
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        # self-attention over token positions (keep same mask semantics as before)
        x = x + self.alpha_self.tanh() * self.self_attn(x, attention_mask=self_attention_mask)

        return x


class RelationAttention(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        query_dim,
        sg_emb_dim,
        n_heads,
        d_head,
        pooling=False,
    ):
        super().__init__()

        self.proj_in = nn.Linear(768, query_dim)

        # note: original code used 2312 as input dim for sg; keep flexible by using sg_emb_dim
        self.linear = nn.Sequential(
            nn.Linear(sg_emb_dim, query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
        )

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.pooling = pooling
        if self.pooling:
            # gate controlling strength of relation-aware pooling interpolation
            self.register_parameter("alpha_pool", nn.Parameter(torch.tensor(-5.0)))

    def forward(self, x, node_embeddings, token_node_assign=None, self_attention_mask=None):
        """
        Args:
            x: [B, N, C_in] token embeddings
            node_embeddings: [B, K, sg_emb_dim]
            token_node_assign: [B, N, K] (soft assignments). If None, attention bias is not applied.
            self_attention_mask: optional mask for self-attention
        """
        x = self.proj_in(x)  # [B, N, query_dim]

        sg_proj = self.linear(node_embeddings)  # [B, K, query_dim]

        # attention bias built from token->node assignments
        bias = None
        if token_node_assign is not None:
            bias = torch.log(token_node_assign + 1e-8)  # [B, N, K]

        # cross-attention where tokens query node embeddings
        x = x + self.alpha_attn.tanh() * self.attn(
            self.norm1(x),
            encoder_hidden_states=self.norm1(sg_proj),
            attention_mask=bias,
        )
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        # optional relation-aware pooling using soft assignments
        if token_node_assign is not None and self.pooling:
            # token_node_assign: [B, N, K] -- weights of tokens per relation/node
            # x: [B, N, D] token features after projection/attn
            # Produce relation-level embeddings by aggregating token embeddings per node:
            # relation_embedding_sums: [B, K, D] = token_node_assign^T @ x
            relation_embedding_sums = torch.einsum("bnk,bnd->bkd", token_node_assign, x)

            # counts per relation (soft), shape [B, K]
            token_counts_per_relation = token_node_assign.sum(dim=1) + 1e-5  # [B, K]

            # average to get relation embeddings
            relation_embedding_averages = relation_embedding_sums / token_counts_per_relation.unsqueeze(-1)  # [B,K,D]

            # reconstruct updated token embeddings as weighted sum of their relations' averages
            updated_token_embeddings = torch.einsum("bnk,bkd->bnd", token_node_assign, relation_embedding_averages)  # [B,N,D]

            # gate controlling interpolation strength
            update_gate = torch.sigmoid(self.alpha_pool)

            # interpolate original token embeddings with updated ones
            x = (1 - update_gate) * x + update_gate * updated_token_embeddings

        return x
