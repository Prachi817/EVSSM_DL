import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -------- LayerNorm --------
class WithBias_LN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / (var + 1e-5).sqrt() * self.weight + self.bias

class LN2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = WithBias_LN(dim)

    def forward(self, x):
        B,C,H,W = x.shape
        t = rearrange(x, "b c h w -> b (h w) c")
        t = self.ln(t)
        return rearrange(t, "b (h w) c -> b c h w", h=H, w=W)

# -------- EDFFN Block --------
class EDFFN(nn.Module):
    def __init__(self, dim, exp=3):
        super().__init__()
        hid = int(dim * exp)
        self.inp = nn.Conv2d(dim, hid * 2, 1)
        self.dw = nn.Conv2d(hid * 2, hid * 2, 3, padding=1, groups=hid * 2)
        self.out = nn.Conv2d(hid, dim, 1)

    def forward(self, x):
        x = self.inp(x)
        x1, x2 = self.dw(x).chunk(2,1)
        return self.out(F.gelu(x1) * x2)

# -------- TinySS2D Block (Pure PyTorch SSM-like) --------
class TinySS2D(nn.Module):
    def __init__(self, dim, exp=2):
        super().__init__()
        hid = int(dim * exp)
        self.inp = nn.Linear(dim, hid)
        self.dw = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)
        self.ln = nn.LayerNorm(hid)
        self.out = nn.Linear(hid, dim)

    def forward(self, x):
        B,C,H,W = x.shape
        t = rearrange(x, "b c h w -> b h w c")
        t = self.inp(t)
        t2 = rearrange(t, "b h w c -> b c h w")
        t2 = F.gelu(self.dw(t2))
        t2 = rearrange(t2, "b c h w -> b h w c")
        t2 = self.ln(t2)
        t = self.out(t2)
        return rearrange(t, "b h w c -> b c h w")

# -------- EVS Block --------
class EVS(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = LN2D(dim)
        self.ssm = TinySS2D(dim)
        self.ln2 = LN2D(dim)
        self.ffn = EDFFN(dim)

    def forward(self, x):
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# -------- Full EVSSM No-Mamba Model --------
class EVSSM_NoMamba(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            EVS(dim),
            EVS(dim)
        )
        self.dec = nn.Sequential(
            EVS(dim),
            EVS(dim),
            nn.Conv2d(dim, 3, 3, padding=1)
        )

    def forward(self, x):
        y = self.enc(x)
        y = self.dec(y)
        return y + x  # residual connection
