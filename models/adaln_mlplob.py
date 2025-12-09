import torch
import torch.nn as nn
from models.bin import BiN


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class AdaLNBlock(nn.Module):
    def __init__(self, start_dim, hidden_dim, final_dim, cond_dim):
        super().__init__()
        self.start_dim = start_dim
        self.final_dim = final_dim

        self.norm = nn.LayerNorm(start_dim, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(start_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, final_dim)

        if start_dim == final_dim:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 3 * start_dim, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * start_dim, bias=True),
            )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        if self.start_dim == self.final_dim:
            shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
            shift, scale, gate = shift.unsqueeze(1), scale.unsqueeze(1), gate.unsqueeze(1)

            h = modulate(self.norm(x), shift, scale)
            h = self.fc2(self.act(self.fc1(h)))
            return x + gate * h

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)

        h = modulate(self.norm(x), shift, scale)
        h = self.fc2(self.act(self.fc1(h)))
        return h


class AdalnMLPLOB(nn.Module):
    def __init__(self, hidden_dim, num_layers, seq_size, num_features, dataset_type):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dataset_type = dataset_type

        self.cond_dim = hidden_dim
        self.stock_embed = nn.Embedding(5, self.cond_dim)

        self.layers = nn.ModuleList()
        self.first_layer = nn.Linear(num_features, hidden_dim)
        self.norm_layer = BiN(num_features, seq_size)

        self.layers.append(self.first_layer)
        self.layers.append(nn.SiLU())

        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(AdaLNBlock(hidden_dim, hidden_dim * 4, hidden_dim, self.cond_dim))
                self.layers.append(AdaLNBlock(seq_size, seq_size * 4, seq_size, self.cond_dim))
            else:
                self.layers.append(AdaLNBlock(hidden_dim, hidden_dim * 2, hidden_dim // 4, self.cond_dim))
                self.layers.append(AdaLNBlock(seq_size, seq_size * 2, seq_size // 4, self.cond_dim))

        total_dim = (hidden_dim // 4) * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.SiLU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, 3))

    def forward(self, input, stock_id):
        c = self.stock_embed(stock_id)

        # First-order difference along sequence dimension; keep length with prepend
        # x = torch.diff(input, dim=1, prepend=input[:, :1, :])
        x = input
        x = x.permute(0, 2, 1)
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)

        for layer in self.layers:
            if isinstance(layer, AdaLNBlock):
                x = layer(x, c)
            else:
                x = layer(x)
            x = x.permute(0, 2, 1)

        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x
