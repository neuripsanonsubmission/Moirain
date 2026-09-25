import torch
import torch.nn as nn
import math


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2.0
    b = 2.0
    std = math.sqrt(scale)
    torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=a, b=b)
    

def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)

def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)

def bietti_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0, fan="fan_out")

def glorot_uniform_init_(weights):
    torch.nn.init.xavier_uniform_(weights, gain=1)

def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)

def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)

def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.scale
    


class Linear(torch.nn.Linear):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
    ):
        super().__init__(in_dim, out_dim, bias=bias)
        
        if init == "default":
            lecun_normal_init_(self.weight)
        elif init == "relu":
            he_normal_init_(self.weight)
        elif init == "glorot":
            glorot_uniform_init_(self.weight)
        elif init == "gating":
            gating_init_(self.weight)
            if bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)
        elif init == "normal":
            normal_init_(self.weight)
        elif init == "final":
            final_init_(self.weight)
        elif init == "bietti":
            bietti_normal_init_(self.weight)
        else:
            raise ValueError("Invalid init string.")
            

class Embedding(torch.nn.Embedding):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        padding_idx: int = None, 
        init: str = "default",
    ):
        self.init = init
        super().__init__(in_dim, out_dim, padding_idx)


    def reset_parameters(self):

        if self.init == "default":
            lecun_normal_init_(self.weight)
        elif self.init == "relu":
            he_normal_init_(self.weight)
        elif self.init == "glorot":
            glorot_uniform_init_(self.weight)
        elif self.init == "gating":
            gating_init_(self.weight)
        elif self.init == "normal":
            normal_init_(self.weight)
        elif self.init == "final":
            final_init_(self.weight)
        elif self.init == "bietti":
            bietti_normal_init_(self.weight)
        else:
            raise ValueError("Invalid init string.")
        
        self._fill_padding_idx_with_zero()


    def forward(self, input):
        if (input < 0).any() or (input >= self.num_embeddings).any():
            raise ValueError(f"Indices out of bounds: must be in [0, {self.num_embeddings - 1}]")
        return super().forward(input)