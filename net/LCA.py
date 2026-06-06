import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from net.transformer_utils import LayerNorm


# Spatial Attention
class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat( [avg_out, max_out], dim=1)
        attn = self.compress(attn)

        return x * self.sigmoid(attn)


# Channel Attention
class ChannelGate(nn.Module):

    def __init__(self, dim, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(dim, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, dim, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        weight = self.fc(self.avg_pool(x))
        return x * weight


# Cross Attention Block
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# IEL
class IEL(nn.Module):

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = torch.tanh(self.dwconv1(x1)) + x1
        x2 = torch.tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Region Refinement Block
class RegionRefinementBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.mask_predictor = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, 1, 1), nn.GELU(), nn.Conv2d(dim // 2, 1, 1), nn.Sigmoid())
        self.conv_branch = nn.Sequential( nn.Conv2d(dim, dim, 3, 1, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1) )
        self.attn_branch = CAB(dim, num_heads=4)

    def forward(self, x):
        mask = self.mask_predictor(x)
        high = x * mask
        low = x * (1 - mask)
        high = self.attn_branch(high, high)
        low = self.conv_branch(low)
        out = high + low + x
        return out


# Enhanced HV-LCA

class EnhancedHV_LCA(nn.Module):

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.cab = CAB(dim, num_heads, bias)
        self.channel_gate = ChannelGate(dim)
        self.spatial_gate = SpatialGate()
        self.iel = IEL(dim)

    def forward(self, x, y):
        x = x + self.cab(self.norm(x), self.norm(y))
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        x = x + self.iel(self.norm(x))
        return x


# Enhanced I-LCA
class EnhancedI_LCA(nn.Module):

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.cab = CAB(dim, num_heads, bias)
        self.channel_gate = ChannelGate(dim)
        self.spatial_gate = SpatialGate()
        self.iel = IEL(dim)

    def forward(self, x, y):
        x = x + self.cab(self.norm(x), self.norm(y))
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        x = x + self.iel(self.norm(x))
        return x