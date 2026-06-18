import torch
import math
import torch.nn as nn
from numpy.distutils.system_info import x11_info
from numpy.random import weibull
from twisted.web.html import output

from common import *
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.modules.mamba_simple_optimized import Mamba
from Vmamba.Myvmamba import *
from pytorch_wavelets import DWTForward, DWTInverse
from kan.kan import *

class WESSP_Mamba(nn.Module):

    def __init__(self, n_colors, scale, n_feats, conv=default_conv):
        super( WESSP_Mamba, self).__init__()
        self.scale = scale
        self.shallow = Shallow(n_colors, n_feats)
        self.sap = SpaPrior(n_colors, n_feats)
        self.body = nn.ModuleList()
        for i in range(4):
            self.body.append(RMM(n_feats, n_feats))
        self.upsample = Upsampler(conv, scale, n_feats)
        self.skip_conv = nn.Conv2d(n_colors, n_feats, 3, padding=1)
        self.tail = nn.Conv2d(n_feats, n_colors, 3, padding=1)

    def forward(self, x, lms):
        sap_x = self.sap(x)
        shallow_x = self.shallow(x)
        xi = shallow_x
        for i in range(4):
            xi = self.body[i](xi, sap_x)
        body_x = xi

        up_x = self.upsample(body_x + shallow_x)
        output = self.tail(up_x + self.skip_conv(lms))
        return output


class Shallow(nn.Module):

    def __init__(self, in_channels, out_channels, conv=default_conv):
        super().__init__()
        self.head = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv0 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x0 = self.head(x)
        skip = x0
        y = self.conv0(x0)
        return y + skip


class SRM(nn.Module):

    def __init__(self, dim):
        super(SRM, self).__init__()
        self.C = dim
        self.mamba = SR_SS2D(dim, expand=1, d_state=8)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = rearrange(input, 'b c h w -> b h w c')
        output = self.mamba(input)
        output = rearrange(output, 'b h w c -> b c h w')
        return output


class PGSSMB(nn.Module):
    def __init__(self, dim, shift_size=0, drop_path=0.0, mlp_ratio=4., act_layer=nn.GELU, drop=0.1):
        super(PGSSMB, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = SRM(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))
        self.alpha3 = nn.Parameter(torch.zeros(1))
        self.cross = PMC(dim)
        self.ca = ChannelAttention(dim)

    def forward(self, input, SpaP):
        b, c, h, w = input.shape

        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c', h=h, w=w)
        input = self.norm1(input)
        input = rearrange(input, 'b (h w) c -> b c h w', h=h, w=w)
        output = self.mamba(input)
        output1 = self.ca(input)
        output = output + output1
        output = output + skip * self.alpha1

        skip = output
        output = rearrange(output, 'b c h w -> b (h w) c', h=h, w=w)
        output = self.norm2(output)
        output = self.drop_path(self.mlp(output))
        output = rearrange(output, 'b (h w) c -> b c h w', h=h, w=w)
        output = output + skip * self.alpha2

        skip = output
        output = self.cross(output, SpaP)
        output = output + skip * self.alpha3

        return output


class RMM(nn.Module):
    """Residual Mamba Module"""
    def __init__(self, in_channels, out_channels, N=3):
        super().__init__()
        self.body = nn.ModuleList()
        self.N = N
        for i in range(self.N):
            self.body.append(PGSSMB(in_channels))
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, input, SpaP):
        skip = input
        xi = input
        for i in range(self.N):
            xi = self.body[i](xi, SpaP)
        output = self.conv(xi)
        return output + skip


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, type='value'):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.type = type

    def forward(self, x):
        attention = self.avg_pool(x)
        attention = self.ca(attention)
        if self.type == 'value':
            return attention * x
        else:
            return attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, type='value'):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.spa = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.type = type

    def forward(self, x):
        b, c, h, w = x.shape
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.spa(concat)
        attention = self.sigmoid(attention)
        if self.type == 'value':
            return attention * x
        else:
            return attention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpaPrior(nn.Module):
    def __init__(self, dim1, dim2):
        super(SpaPrior, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim1, dim1, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim1, dim1, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim1, dim1, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim1, dim1, 3, 1, 1)
        )
        self.tail = nn.Conv2d(dim1, dim2, 3, 1, 1)
        self.ca = ChannelAttention(dim1, type='coffic')
        self.sa = SpatialAttention(kernel_size=3)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        b, c, h, w = input.shape

        xfm = DWTForward(J=1, mode='zero', wave='db1').cuda(device_id0)
        ifm = DWTInverse(mode='zero', wave='db1').cuda(device_id0)

        Yl, Yh = xfm(input)
        LL = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)
        HL = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)
        LH = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)
        HH = torch.zeros((b, c, h // 2, w // 2)).float().cuda(device_id0)

        LL[:, :, :, :] = Yl
        HL[:, :, :, :] = Yh[0][:, :, 0, :, :]
        LH[:, :, :, :] = Yh[0][:, :, 1, :, :]
        HH[:, :, :, :] = Yh[0][:, :, 2, :, :]

        HL = self.conv1(HL) + HL
        LH = self.conv1(LH) + LH
        HH = self.conv2(HH) + HH
        LL = self.sa(LL) + LL

        Yl = LL[:, :, :, :]
        Yh[0][:, :, 0, :, :] = HL[:, :, :, :]
        Yh[0][:, :, 1, :, :] = LH[:, :, :, :]
        Yh[0][:, :, 2, :, :] = HH[:, :, :, :]

        output = ifm((Yl, Yh))
        output = self.ca(skip) * output

        return self.tail(output + skip)


class PMC(nn.Module):
    """Prior Mamba Cross"""
    def __init__(self, dim):
        super(PMC, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x1, x2):
        skip = x1
        attn1 = self.conv1(x1)
        attn2 = self.conv2(x2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)

        return skip * attn
