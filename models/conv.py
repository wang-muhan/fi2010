import torch
import torch.nn as nn
from models.bin import BiN

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=32, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_chan, in_chan, kernel_size=kernel_size, 
                                   groups=in_chan, padding='same') 
        self.pointwise = nn.Conv1d(in_chan, out_chan, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_chan)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class ConvLOB(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 dataset_type: str,
                 kernel_size: int = 32,
                 dropout: float = 0.0
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dataset_type = dataset_type
        self.order_type_embedder = nn.Embedding(3, 1)
        
        self.norm_layer = BiN(num_features, seq_size)
        self.projection = nn.Conv1d(num_features, hidden_dim, kernel_size=1)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DepthwiseSeparableConv(hidden_dim, hidden_dim, kernel_size=kernel_size, dropout=dropout))
            
        total_dim = hidden_dim * seq_size
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, 3))

    def forward(self, input):
        if self.dataset_type == "LOBSTER":
            continuous_features = torch.cat([input[:, :, :41], input[:, :, 42:]], dim=2)
            order_type = input[:, :, 41].long()
            order_type_emb = self.order_type_embedder(order_type).detach()
            x = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            x = input
            
        # Input: (Batch, Seq, Features)
        x = x.permute(0, 2, 1) # (Batch, Features, Seq)
        x = self.norm_layer(x)
        
        x = self.projection(x) # (Batch, Hidden, Seq)
        
        for layer in self.layers:
            x = layer(x)
            
        x = x.reshape(x.shape[0], -1) # Flatten
        
        for layer in self.final_layers:
            x = layer(x)
        return x
