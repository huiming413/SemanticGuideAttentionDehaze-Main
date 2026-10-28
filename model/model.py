import torch.nn as nn
import torch.nn.functional as F
from .deconv import FastDeconv
import torch


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels, out_channels, kernel_size, bias=True, groups=1):
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=groups)

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.LeakyReLU(0.05, True)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + x
        return x

class LightweightSemanticExtractor(nn.Module):
    def __init__(self, in_channels=3, semantic_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(64, semantic_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.05, True)
        )

    def forward(self, x):
        return self.encoder(x)

class SemanticGuidedAttention(nn.Module):
    def __init__(self, esa_channels, n_feats, semantic_channels, conv):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.sem_proj = conv(semantic_channels, f, kernel_size=1)
        self.fuse = conv(f * 2, f, kernel_size=1)

    def forward(self, x, sem_feat=None):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)

        if sem_feat is not None:
            sem = F.interpolate(sem_feat, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            sem = self.sem_proj(sem)
            spatial_feat = c3 + cf
            fused_feat = self.fuse(torch.cat([spatial_feat, sem], dim=1))
        else:
            fused_feat = c3 + cf

        c4 = self.conv4(fused_feat)
        m = self.sigmoid(c4)
        return x * m


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SemanticRefineBlock(nn.Module):
    def __init__(self, dim, semantic_channels=64, esa_channels=32):
        super().__init__()
        self.rfblock = Block(dim)
        self.eca = eca_layer(dim, k_size=5)  # 始终初始化 ECA
        self.sga = SemanticGuidedAttention(esa_channels, dim, semantic_channels, nn.Conv2d)
        self.c1 = conv_layer(dim, dim, 1)

    def forward(self, x, sem_feat=None, use_eca=False):
        x1 = self.rfblock(x)
        x2 = self.rfblock(x1)
        x3 = self.rfblock(x2)
        out = x + x3

        # 动态控制 ECA 是否参与前向传播
        if use_eca:
            out = self.eca(out)

        out = self.sga(self.c1(out), sem_feat)
        return out

class SPRBS(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_channels=52, semantic_channels=64):
        super(SPRBS, self).__init__()
        self.conv_1 = conv_layer(in_channels, feature_channels, kernel_size=3)
        self.block1 = SemanticRefineBlock(feature_channels, semantic_channels)
        self.block2 = SemanticRefineBlock(feature_channels, semantic_channels)
        self.block3 = SemanticRefineBlock(feature_channels, semantic_channels)
        self.block4 = SemanticRefineBlock(feature_channels, semantic_channels)
        self.block5 = SemanticRefineBlock(feature_channels, semantic_channels)
        self.block6 = SemanticRefineBlock(feature_channels, semantic_channels)
        self.conv_2 = conv_layer(feature_channels, out_channels, kernel_size=3)

    def forward(self, x, sem_feat=None, use_eca=False):
        out_feature = self.conv_1(x)
        out_b1 = self.block1(out_feature, sem_feat, use_eca)
        out_b2 = self.block2(out_b1, sem_feat, use_eca)
        out_b3 = self.block3(out_b2, sem_feat, use_eca)
        out_b4 = self.block4(out_b3, sem_feat, use_eca)
        out_b5 = self.block5(out_b4, sem_feat, use_eca)
        out_b6 = self.block6(out_b5, sem_feat, use_eca)
        out_low_resolution = out_b6 + out_feature
        output = self.conv_2(out_low_resolution)
        return output


class Dehaze(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, use_dropout=False, padding_type='reflect'):
        super(Dehaze, self).__init__()
        self.semantic_extractor = LightweightSemanticExtractor(in_channels=input_nc, semantic_channels=ngf * 2)
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        self.block = SPRBS(ngf * 4, ngf * 4, ngf * 4, semantic_channels=ngf * 2)
        self.up1 = nn.Sequential(conv_layer(ngf * 4, ngf * 8, 1), nn.PixelShuffle(2), nn.ReLU(True))
        self.up2 = nn.Sequential(conv_layer(ngf * 2, ngf * 4, 1), nn.PixelShuffle(2), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())
        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, input, use_eca=False):
        x_deconv = self.deconv(input)
        sem_feat = self.semantic_extractor(input)
        x_down1 = self.down1(x_deconv)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x = self.block(x_down3, sem_feat, use_eca)
        x_up1 = self.up1(x)
        x_up2 = self.up2(x_up1 + x_down2)
        out = self.up3(x_up2)
        return out