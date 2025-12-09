from torch import nn
import torch
from einops import rearrange
import constants as cst
from models.bin import BiN
from models.mlplob import MLP
from fla.layers import GatedLinearAttention

class GLALayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = GatedLinearAttention(hidden_size=hidden_dim, num_heads=num_heads)
        self.mlp = MLP(hidden_dim, hidden_dim*4, final_dim)
        self.w0 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        res = x
        x, *_ = self.attention(x)
        x = self.w0(x)
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x, None

class GLABlockCLS(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = GatedLinearAttention(
            hidden_size=hidden_dim,
            num_heads=num_heads
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
        normed = self.norm1(x)
        attn_out, *_ = self.attn(normed)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x, None

class GLALOBClsToken(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 num_heads: int,
                 is_sin_emb: bool,
                 dataset_type: str,
                 dropout: float = 0.1,
                 mlp_ratio: float = 4.0
                 ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.is_sin_emb = is_sin_emb
        self.dataset_type = dataset_type

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
                GLABlockCLS(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, input, store_att: bool = False):
        # if self.dataset_type == "LOBSTER":
        #     continuous_features = torch.cat([input[:, :, :41], input[:, :, 42:]], dim=2)
        #     order_type = input[:, :, 41].long()
        #     order_type_emb = self.order_type_embedder(order_type).detach()
        #     x = torch.cat([continuous_features, order_type_emb], dim=2)
        # else:
        x = input

        x = rearrange(x, 'b s f -> b f s')
        x = self.norm_layer(x)
        x = rearrange(x, 'b f s -> b s f')
        x = self.emb_layer(x)

        bsz = x.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([x, cls], dim=1)
        x = x + self.pos_encoder

        att_list = []
        for layer in self.layers:
            x, att = layer(x, need_weights=store_att)
            if store_att:
                att_list.append(att)

        cls_out = x[:, -1]
        logits = self.head(cls_out)

        if store_att:
            return logits, att_list
        return logits

class GLALOB(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 num_heads: int,
                 is_sin_emb: bool,
                 dataset_type: str
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_sin_emb = is_sin_emb
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.dataset_type = dataset_type
        self.layers = nn.ModuleList()
        # self.order_type_embedder = nn.Embedding(3, 1) # Removed as part of LOBSTER removal
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        
        if is_sin_emb:
            self.pos_encoder = sinusoidal_positional_embedding(seq_size, hidden_dim)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))
        
        for i in range(num_layers):
            if i != num_layers-1:
                self.layers.append(GLALayer(hidden_dim, num_heads, hidden_dim))
                self.layers.append(GLALayer(seq_size, num_heads, seq_size))
            else:
                self.layers.append(GLALayer(hidden_dim, num_heads, hidden_dim//4))
                self.layers.append(GLALayer(seq_size, num_heads, seq_size//4))
            
        total_dim = (hidden_dim//4) * (seq_size//4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim//4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim//4
        self.final_layers.append(nn.Linear(total_dim, 3))
    
    def forward(self, input):
        # if self.dataset_type == "LOBSTER":
        #     continuous_features = torch.cat([input[:, :, :41], input[:, :, 42:]], dim=2)
        #     order_type = input[:, :, 41].long()
        #     order_type_emb = self.order_type_embedder(order_type).detach()
        #     x = torch.cat([continuous_features, order_type_emb], dim=2)
        # else:
        x = input
        x = rearrange(x, 'b s f -> b f s')
        x = self.norm_layer(x)
        x = rearrange(x, 'b f s -> b s f')
        x = self.emb_layer(x)
        x = x[:] + self.pos_encoder
        for i in range(len(self.layers)):
            x, _ = self.layers[i](x)
            x = x.permute(0, 2, 1)
        x = rearrange(x, 'b s f -> b (f s) 1')              
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x

def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):
    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim")

    T = token_sequence_size
    d = token_embedding_dim

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d)
    embeddings[:, 0::2] = torch.sin(positions/denominators)
    embeddings[:, 1::2] = torch.cos(positions/denominators)

    return embeddings.to(cst.DEVICE, non_blocking=True)
