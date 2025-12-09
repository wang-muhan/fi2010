import torch
import torch.nn as nn
from kan import KAN
from models.bin import BiN


class KANBlock(nn.Module):
    def __init__(
        self,
        start_dim: int,
        hidden_dim: int,
        final_dim: int,
        grid: int = 3,
        k: int = 3,
        device=None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kan = KAN(
            width=[start_dim, hidden_dim, final_dim],
            grid=grid,
            k=k,
            seed=0,
            device=self.device,
            auto_save=False,  # disable auto-save to avoid stray folders
        )
        self.norm = nn.LayerNorm(final_dim)
        self.act = nn.GELU()

    def forward(self, x):
        if x.device != self.device:
            self.kan = self.kan.to(x.device)
            self.device = x.device
        residual = x if x.shape[-1] == self.kan.width[0] else None
        b, s, d = x.shape
        x = x.reshape(-1, d)
        x = self.kan(x)
        x = x.reshape(b, s, -1)
        if residual is not None and x.shape == residual.shape:
            x = x + residual
        x = self.norm(x)
        x = self.act(x)
        return x


class KANLOB(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 dataset_type: str
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dataset_type = dataset_type

        self.norm_layer = BiN(num_features, seq_size)
        self.layers = nn.ModuleList()
        self.layers.append(KANBlock(num_features, hidden_dim, hidden_dim))
        for i in range(num_layers):
            if i != num_layers - 1:
                # keep width stable to reduce compute
                self.layers.append(KANBlock(hidden_dim, hidden_dim, hidden_dim))
            else:
                self.layers.append(KANBlock(hidden_dim, hidden_dim, max(hidden_dim // 2, 1)))
        
        total_dim = max(hidden_dim // 2, 1) * seq_size
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, 3))
    
    def forward(self, input):
        x = input
        x = x.permute(0, 2, 1)
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x)

        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x
