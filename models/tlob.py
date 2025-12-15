from torch import nn
import torch
import torch.nn.functional as F
import math
from einops import rearrange
import constants as cst
from models.bin import BiN
from models.mlplob import MLP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Norms for Post-Norm architecture
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Integrated QKV projections
        self.q = nn.Linear(hidden_dim, hidden_dim*num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim*num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim*num_heads)
        
        # Standard MLP/FFN block
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)
        
        self.w0 = nn.Linear(hidden_dim*num_heads, hidden_dim)
        
    def forward(self, x, need_weights: bool = False, pos_emb=None):
        res = x
        v = self.v(x)

        # Q and K are computed from pos_emb if available and dimensions match
        # This creates "static" attention maps based on relative positions for temporal layers
        if pos_emb is not None and pos_emb.shape[-1] == self.hidden_dim:
            q = self.q(pos_emb.unsqueeze(0))
            k = self.k(pos_emb.unsqueeze(0))
        else:
            q = self.q(x)
            k = self.k(x)

        head_dim = self.hidden_dim
        # Rearrange to (b, h, s, d)
        # Note: if q/k came from pos_emb with batch=1, rearrange works fine 
        # because we start with 'b' (which is 1)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads, d=head_dim)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads, d=head_dim)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads, d=head_dim)

        # SDPA supports broadcasting if query/key have batch_size=1 and value has batch_size=B
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

        att = None
        if need_weights:
            scale = math.sqrt(head_dim)
            att = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)

        x = rearrange(attn_out, "b h s d -> b s (h d)")
        x = self.w0(x)
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        
        if x.shape == res.shape:
            x = x + res
            
        return x, att


class TLOB(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 num_heads: int,
                 is_sin_emb: bool,
                 dataset_type: str,
                 use_pos_in_attn: bool = False
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_sin_emb = is_sin_emb
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.dataset_type = dataset_type
        self.use_pos_in_attn = use_pos_in_attn
        self.layers = nn.ModuleList()
        self.first_branch = nn.ModuleList()
        self.second_branch = nn.ModuleList()
        # self.order_type_embedder = nn.Embedding(3, 1) # Removed as part of LOBSTER removal
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        if is_sin_emb:
            self.pos_encoder = sinusoidal_positional_embedding(seq_size, hidden_dim)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))
        
        for i in range(num_layers):
            if i != num_layers-1:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))
            else:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim//4))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size//4))
        self.att_temporal = []
        self.att_feature = []
        self.mean_att_distance_temporal = []
        total_dim = (hidden_dim//4)*(seq_size//4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim//4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim//4
        self.final_layers.append(nn.Linear(total_dim, 3))
        
    
    def forward(self, input, store_att=False):
        x = input
        x = rearrange(x, 'b s f -> b f s')
        x = self.norm_layer(x)
        x = rearrange(x, 'b f s -> b s f')
        x = self.emb_layer(x)
        x = x[:] + self.pos_encoder
        att_list = []
        for i in range(len(self.layers)):
            if self.use_pos_in_attn:
                x, att = self.layers[i](x, need_weights=store_att, pos_emb=self.pos_encoder)
            else:
                x, att = self.layers[i](x, need_weights=store_att)
            if store_att and att is not None:
                att_list.append(att.detach().cpu())
            x = x.permute(0, 2, 1)
        x = rearrange(x, 'b s f -> b (f s) 1')              
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
            
        if store_att:
            return x, att_list
        return x
    
    
def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings.to(cst.DEVICE, non_blocking=True)


def count_parameters(layer):
    print(f"Number of parameters: {sum(p.numel() for p in layer.parameters() if p.requires_grad)}")
    

def compute_mean_att_distance(att):
    att_distances = np.zeros((att.shape[0], att.shape[1]))
    for h in range(att.shape[0]):
        for key in range(att.shape[2]):
            for query in range(att.shape[1]):
                distance = abs(query-key)
                att_distances[h, key] += torch.abs(att[h, query, key]).cpu().item()*distance
    mean_distances = att_distances.mean(axis=1)
    return mean_distances
    
    
class TransformerBlockCLS(nn.Module):
    """
    Standard Transformer block for CLS-token flow (no axis swapping / flatten).
    """
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            device=cst.DEVICE
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, need_weights: bool = False):
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, attn = self.attn(normed, normed, normed, need_weights=need_weights, average_attn_weights=False)
        x = x + self.dropout(attn_out)

        # Feed-forward
        x = x + self.mlp(self.norm2(x))

        if need_weights:
            return x, attn
        return x, None


class TLOBClsToken(nn.Module):
    """
    ViT-style variant: prepend CLS token, keep sequence length, classify from CLS without
    flatten/down-projection of all tokens.
    """
    def __init__(self,
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 num_heads: int,
                 is_sin_emb: bool,
                 dataset_type: str,
                 dropout: float = 0.1,
                 mlp_ratio: float = 4.0,
                 use_pos_in_attn: bool = False
                 ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.is_sin_emb = is_sin_emb
        self.dataset_type = dataset_type
        self.use_pos_in_attn = use_pos_in_attn

        # self.order_type_embedder = nn.Embedding(3, 1) # Removed as part of LOBSTER removal
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        if is_sin_emb:
            pos = sinusoidal_positional_embedding(seq_size + 1, hidden_dim).unsqueeze(0)
            self.register_buffer("pos_encoder", pos, persistent=False)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size + 1, hidden_dim))

        self.layers = nn.ModuleList(
            [
                TransformerBlockCLS(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, input, store_att: bool = False):
        x = input

        x = rearrange(x, 'b s f -> b f s')
        x = self.norm_layer(x)
        x = rearrange(x, 'b f s -> b s f')
        x = self.emb_layer(x)

        bsz = x.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_encoder

        att_list = []
        for layer in self.layers:
            x, att = layer(x, need_weights=store_att)
            if store_att:
                att_list.append(att.detach())

        cls_out = x[:, 0]
        logits = self.head(cls_out)

        if store_att:
            return logits, att_list
        return logits
