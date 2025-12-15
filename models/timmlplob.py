from functools import partial
from typing import Optional, Type, Union, Tuple
import collections.abc

import torch
from torch import nn

from models.bin import BiN

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x
    return tuple(x for _ in range(2))

class GlobalResponseNorm(nn.Module):
    """ Global Response Normalization layer
    """
    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        self.eps = eps
        self.channels_last = channels_last
        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        if self.channels_last:
            # Expects (N, ..., C)
            # GRN computes norm over spatial dimensions.
            # If x is (N, C), there are no spatial dimensions.
            # If x is (N, L, C), spatial dim is L (dim 1).
            if x.ndim == 3:
                gx = torch.norm(x, p=2, dim=1, keepdim=True)
                nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
                return self.weight * (x * nx) + self.bias + x
            else:
                 # Fallback for 2D or other shapes: treat as identity or handle appropriately
                 return x 
        else:
            # Expects (N, C, H, W)
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=(2, 3), keepdim=True) + self.eps)
            return self.weight.view(1, -1, 1, 1) * (x * nx) + self.bias.view(1, -1, 1, 1) + x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Optional[Type[nn.Module]] = None,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: Union[float, Tuple[float, float]] = 0.,
            use_conv: bool = False,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features, **dd) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1], **dd)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.Sigmoid,
            norm_layer: Optional[Type[nn.Module]] = None,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: Union[float, Tuple[float, float]] = 0.,
            use_conv: bool = False,
            gate_last: bool = True,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features // 2, **dd) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1], **dd)
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1.bias is not None:
            nn.init.ones_(self.fc1.bias[self.fc1.bias.shape[0] // 2:])
        nn.init.normal_(self.fc1.weight[self.fc1.weight.shape[0] // 2:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


SwiGLUPacked = partial(GluMlp, act_layer=nn.SiLU, gate_last=False)


class SwiGLU(nn.Module):
    """ SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.SiLU,
            norm_layer: Optional[Type[nn.Module]] = None,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: Union[float, Tuple[float, float]] = 0.,
            align_to: int = 0,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        if align_to:
            hidden_features = hidden_features + (-hidden_features % align_to)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0], **dd)
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0], **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features, **dd) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1], **dd)
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Optional[Type[nn.Module]] = None,
            gate_layer: Optional[Type[nn.Module]] = None,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: Union[float, Tuple[float, float]] = 0.,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0], **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features, **dd)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.norm = norm_layer(hidden_features, **dd) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1], **dd)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims (for 2D NCHW tensors)
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Optional[Type[nn.Module]] = None,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: float = 0.,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0], **dd)
        self.norm = norm_layer(hidden_features, **dd) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1], **dd)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class GlobalResponseNormMlp(nn.Module):
    """ MLP w/ Global Response Norm (see grn.py), nn.Linear or 1x1 Conv2d

    NOTE: Intended for '2D' NCHW (use_conv=True) or NHWC (use_conv=False, channels-last) tensor layouts
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: Union[float, Tuple[float, float]] = 0.,
            use_conv: bool = False,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.grn = GlobalResponseNorm(hidden_features, channels_last=not use_conv, **dd)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1], **dd)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

MLP_VARIANTS = {
    "Mlp": Mlp,
    "GluMlp": GluMlp,
    "SwiGLU": SwiGLU,
    "GatedMlp": GatedMlp,
    "ConvMlp": ConvMlp,
    "GlobalResponseNormMlp": GlobalResponseNormMlp,
}

class TimMlpBlock(nn.Module):
    def __init__(self, start_dim, hidden_dim, final_dim, mlp_class, **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(start_dim)
        # timm mlps typically don't include the first projection or pre-norm inside.
        # But mlp_class provided takes in_features, hidden_features, out_features.
        # This matches what we need.
        self.mlp = mlp_class(in_features=start_dim, hidden_features=hidden_dim, out_features=final_dim, **kwargs)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.mlp(x)
        if x.shape == residual.shape:
            x = x + residual
        return x

class TIMMLPLOB(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 dataset_type: str,
                 variant: str = "Mlp"
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dataset_type = dataset_type
        self.variant = variant
        self.layers = nn.ModuleList()
        
        if variant not in MLP_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(MLP_VARIANTS.keys())}")
        
        mlp_class = MLP_VARIANTS[variant]
        
        self.first_layer = nn.Linear(num_features, hidden_dim)
        self.norm_layer = BiN(num_features, seq_size)
        self.layers.append(self.first_layer)
        self.layers.append(nn.GELU())
        
        # Determine if we should use conv mode for MLP (default False)
        # If using ConvMlp, we might need use_conv=True but that expects 2D NCHW.
        # Here we have 1D sequence (N, S, C).
        # We will use standard Linear mode (use_conv=False).
        
        for i in range(num_layers):
            if i != num_layers-1:
                self.layers.append(TimMlpBlock(hidden_dim, hidden_dim*4, hidden_dim, mlp_class))
                self.layers.append(TimMlpBlock(seq_size, seq_size*4, seq_size, mlp_class))
            else:
                self.layers.append(TimMlpBlock(hidden_dim, hidden_dim*2, hidden_dim//4, mlp_class))
                self.layers.append(TimMlpBlock(seq_size, seq_size*2, seq_size//4, mlp_class))
                
        total_dim = (hidden_dim//4)*(seq_size//4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim//4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim//4
        self.final_layers.append(nn.Linear(total_dim, 3))
    
    def forward(self, input):
        x = input
        x = x.permute(0, 2, 1)
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)
        
        # Apply layers
        # The logic in MLPLOB.forward iterates layers.
        # Layer 0: Linear (applies to last dim C).
        # Layer 1: GELU.
        # Layer 2: TimMlpBlock (C-mixing). Input (N, S, C). Linear applies to C.
        #   BUT wait, in MLPLOB:
        #   x = layer(x)
        #   x = x.permute(0, 2, 1)
        # 
        # MLPLOB logic:
        # Linear(C -> H). x: (N, S, C) -> (N, S, H).
        # Permute -> (N, H, S).
        # GELU.
        # Permute -> (N, S, H).
        # MLP(H -> 4H -> H). Input (N, S, H). Linear on H.
        # Permute -> (N, H, S).
        # MLP(S -> 4S -> S). Input (N, H, S). Linear on S.
        # Permute -> (N, S, H).
        #
        # So it seems MLPLOB explicitly manages permutations between layers to alternate dimensions.
        
        for layer in self.layers:
            x = layer(x)
            x = x.permute(0, 2, 1)
            
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x

