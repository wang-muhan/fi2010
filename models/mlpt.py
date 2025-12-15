import torch
import torch.nn as nn
from einops import rearrange

from models.adaln_mlplob import AdaLNBlock
from models.bin import BiN
from models.tlob import TransformerLayer, sinusoidal_positional_embedding


class MLPT(nn.Module):
    """
    MLP with AdaLN conditioning followed by Transformer blocks.
    The MLP mirrors AdalnMLPLOB but keeps feature/temporal dims intact
    (no downscaling) before handing off to the Transformer stack.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_mlp_layers: int,
        num_trans_layers: int,
        seq_size: int,
        num_features: int,
        num_heads: int,
        is_sin_emb: bool,
        dataset_type: str,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_mlp_layers = num_mlp_layers
        self.num_trans_layers = num_trans_layers
        self.seq_size = seq_size
        self.dataset_type = dataset_type

        self.cond_dim = hidden_dim
        self.stock_embed = nn.Embedding(5, self.cond_dim)

        self.norm_layer = BiN(num_features, seq_size)
        self.first_layer = nn.Linear(num_features, hidden_dim)
        self.first_act = nn.SiLU()

        # AdaLN MLP blocks (no dimensionality reduction at the tail)
        self.mlp_feature_blocks = nn.ModuleList()
        self.mlp_temporal_blocks = nn.ModuleList()
        for _ in range(num_mlp_layers):
            self.mlp_feature_blocks.append(
                AdaLNBlock(hidden_dim, hidden_dim * 4, hidden_dim, self.cond_dim)
            )
            self.mlp_temporal_blocks.append(
                AdaLNBlock(seq_size, seq_size * 4, seq_size, self.cond_dim)
            )

        # Positional encoding for the Transformer stack
        if is_sin_emb:
            pos = sinusoidal_positional_embedding(seq_size, hidden_dim).unsqueeze(0)
            self.register_buffer("pos_encoder", pos, persistent=False)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))

        # Transformer blocks (feature axis then temporal axis), final pair reduces dims
        self.trans_feature_blocks = nn.ModuleList()
        self.trans_temporal_blocks = nn.ModuleList()
        for i in range(num_trans_layers):
            if i != num_trans_layers - 1:
                self.trans_feature_blocks.append(
                    TransformerLayer(hidden_dim, num_heads, hidden_dim)
                )
                self.trans_temporal_blocks.append(
                    TransformerLayer(seq_size, num_heads, seq_size)
                )
            else:
                self.trans_feature_blocks.append(
                    TransformerLayer(hidden_dim, num_heads, hidden_dim // 4)
                )
                self.trans_temporal_blocks.append(
                    TransformerLayer(seq_size, num_heads, seq_size // 4)
                )

        total_dim = (hidden_dim // 4) * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim //= 4
        self.final_layers.append(nn.Linear(total_dim, 3))

    def forward(self, input, stock_id=None):
        # Fallback to zeros if stock ids are not provided
        if stock_id is None:
            stock_id = torch.zeros(input.shape[0], dtype=torch.long, device=input.device)

        c = self.stock_embed(stock_id)

        # BiN normalization
        x = input
        x = x.permute(0, 2, 1)
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)

        # AdaLN MLP stack
        x = self.first_act(self.first_layer(x))
        for feat_block, temp_block in zip(self.mlp_feature_blocks, self.mlp_temporal_blocks):
            x = feat_block(x, c)
            x = x.permute(0, 2, 1)  # swap to (batch, hidden, seq) for temporal block
            x = temp_block(x, c)
            x = x.permute(0, 2, 1)  # back to (batch, seq, hidden)
        x = x[:, 1:, :] - x[:, :-1, :]
        x = torch.cat([x[:, :1, :], x], dim=1)
        # Transformer stack
        x = x + self.pos_encoder
        for feat_block, temp_block in zip(self.trans_feature_blocks, self.trans_temporal_blocks):
            x, _ = feat_block(x)
            x = x.permute(0, 2, 1)  # (batch, hidden, seq)
            x, _ = temp_block(x)
            x = x.permute(0, 2, 1)  # (batch, seq, hidden)

        x = rearrange(x, "b s f -> b (f s)")
        for layer in self.final_layers:
            x = layer(x)
        return x

