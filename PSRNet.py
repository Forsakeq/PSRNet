# PSRNet.py
# PSRNet with:
#   1) GT-guided physical slice routing
#   2) Tri-modal hyperedge fusion
#   3) Euler-Elastica Boundary Evolution Module (scheme A)

import os
import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from swin.transformer import SwinTransformerBackbone

try:
    from timm.layers import trunc_normal_
except Exception:
    from timm.models.layers import trunc_normal_


def PatchToImage(feature: torch.Tensor) -> torch.Tensor:
    assert feature.dim() == 3, f"PatchToImage expects (B,L,C), got {feature.shape}"
    b, l, c = feature.shape
    h = int(round(math.sqrt(l)))
    assert h * h == l, f"PatchToImage expects square L, got L={l}"
    return feature.permute(0, 2, 1).contiguous().view(b, c, h, h)


def ImageToPatch(feature: torch.Tensor) -> torch.Tensor:
    assert feature.dim() == 4, f"ImageToPatch expects (B,C,H,W), got {feature.shape}"
    return feature.flatten(-2).permute(0, 2, 1).contiguous()


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(in_dim, 4 * out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, _ = x.shape
        assert L == H * W, f"PatchExpand mismatch: {L} vs {H}*{W}"
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, _ = x.shape
        assert L == H * W, f"FinalPatchExpand mismatch: {L} vs {H}*{W}"
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(
            x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


class ScoreModule(nn.Module):
    def __init__(self, channels, image_size=None):
        super().__init__()
        self.extra_model = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1)
        self.image_size = image_size

    def forward(self, x):
        x = self.extra_model(x)
        x = self.conv_1(x)
        if self.image_size is not None:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=True)
        return x


class Conv3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.extra_model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.extra_model(x)


class MorphologicalGradient(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        assert k % 2 == 1
        self.k = int(k)
        self.pad = k // 2

    def forward(self, x):
        dil = F.max_pool2d(x, kernel_size=self.k, stride=1, padding=self.pad)
        ero = -F.max_pool2d(-x, kernel_size=self.k, stride=1, padding=self.pad)
        return (dil - ero).clamp(0.0, 1.0)


class SpatialGradient(nn.Module):
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]) / 8.0
        gy = gx.t().contiguous()
        self.register_buffer('gx', gx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer('gy', gy.view(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        C = x.shape[1]
        gx = self.gx.to(x.device, x.dtype).repeat(C, 1, 1, 1)
        gy = self.gy.to(x.device, x.dtype).repeat(C, 1, 1, 1)
        dx = F.conv2d(x, gx, padding=1, groups=C)
        dy = F.conv2d(x, gy, padding=1, groups=C)
        return dx, dy


class LearnableMonotonicSliceCoords(nn.Module):
    def __init__(self, num_slices: int = 12, init_step: float = 1.0):
        super().__init__()
        self.S = int(num_slices)
        inv_sp = math.log(math.exp(init_step) - 1.0)
        self.delta_raw = nn.Parameter(torch.ones(self.S, dtype=torch.float32) * inv_sp)

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        delta = F.softplus(self.delta_raw)
        phi = torch.cumsum(delta, dim=0)
        phi = (phi - phi[0]) / (phi[-1] - phi[0] + 1e-6) * (self.S - 1)
        if device is not None:
            phi = phi.to(device)
        if dtype is not None:
            phi = phi.to(dtype)
        return phi


class Slice1DPhysicalManhattanAttention(nn.Module):
    def __init__(self, dim: int, num_slices: int = 12, num_heads: int = None, init_gamma: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.S = int(num_slices)
        if num_heads is None:
            num_heads = max(1, dim // 64)
        if dim % num_heads != 0:
            num_heads = 1
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        self.gamma_raw = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, fs_slices: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = fs_slices.shape
        assert S == self.S and C == self.dim
        gamma = F.softplus(self.gamma_raw)
        dist = (phi.view(S, 1) - phi.view(1, S)).abs()
        bias = -gamma * dist

        x = fs_slices.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, S, C)
        qkv = self.qkv(x).view(B * H * W, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + bias.view(1, 1, S, S).to(attn.dtype).to(attn.device)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B * H * W, S, C)
        out = self.proj(out)
        x_ref = x + out
        return x_ref.view(B, H, W, S, C).permute(0, 3, 4, 1, 2).contiguous()


class GTGuidedPhysicalSliceSelector(nn.Module):
    def __init__(self, dim, num_slices=12, hidden_ratio=0.5,
                 init_tau=1.5, tau_min=0.3, tau_max=5.0,
                 attn_heads=None, init_gamma=1.0,
                 region_temperature=0.7,
                 train_gt_prior_strength=0.85,
                 infer_prior_strength=0.35,
                 prior_radius=1.0,
                 topk=3,
                 use_neighbor_mask=True):
        super().__init__()
        self.S = int(num_slices)
        hidden = max(8, int(dim * hidden_ratio))

        self.slice_coords = LearnableMonotonicSliceCoords(num_slices=self.S, init_step=1.0)
        self.slice_attn = Slice1DPhysicalManhattanAttention(
            dim=dim, num_slices=self.S, num_heads=attn_heads, init_gamma=init_gamma
        )

        self.mu_head = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True)
        )
        self.tau_head = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True)
        )
        self.region_head = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True)
        )

        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.region_temperature = float(region_temperature)
        self.topk = int(topk) if topk is not None else 0
        self.use_neighbor_mask = bool(use_neighbor_mask)

        self.gt_prior_logit = nn.Parameter(torch.tensor(self._inv_sigmoid(train_gt_prior_strength), dtype=torch.float32))
        self.infer_prior_logit = nn.Parameter(torch.tensor(self._inv_sigmoid(infer_prior_strength), dtype=torch.float32))
        self.log_radius = nn.Parameter(torch.tensor(float(prior_radius)).log())

        with torch.no_grad():
            for m in self.mu_head.modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                    m.bias.zero_()
            for m in self.region_head.modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                    m.bias.zero_()
            for m in self.tau_head.modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                    v = max(1e-4, float(init_tau) - self.tau_min)
                    m.bias.fill_(math.log(math.exp(v) - 1.0))

    @staticmethod
    def _inv_sigmoid(p: float) -> float:
        eps = 1e-4
        p = max(eps, min(1.0 - eps, float(p)))
        return math.log(p / (1.0 - p))

    def _radius(self, device, dtype):
        return self.log_radius.exp().to(device=device, dtype=dtype).clamp(min=1e-3, max=float(self.S))

    def _resize_mask(self, mask: torch.Tensor, size_hw):
        if mask is None:
            return None
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        assert mask.dim() == 4 and mask.size(1) == 1
        mask = mask.float()
        if mask.shape[-2:] != size_hw:
            mask = F.interpolate(mask, size=size_hw, mode='nearest')
        return mask.clamp(0.0, 1.0)

    def _pool_slice_distribution(self, fs_ref: torch.Tensor, x_rgb: torch.Tensor, region_mask: torch.Tensor):
        rgb_n = F.normalize(x_rgb, dim=1, eps=1e-6).unsqueeze(1)
        fs_n = F.normalize(fs_ref, dim=2, eps=1e-6)
        sim = (fs_n * rgb_n).sum(dim=2)
        m_hw = region_mask.clamp(0.0, 1.0).squeeze(1)
        denom = m_hw.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        scores = (sim * m_hw.unsqueeze(1)).sum(dim=(2, 3), keepdim=True) / denom.unsqueeze(1)
        scores = scores.squeeze(-1).squeeze(-1)
        alpha_region = torch.softmax(scores / self.region_temperature, dim=1)
        return alpha_region, sim

    def _center_phi_from_alpha(self, alpha_region: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        center = (alpha_region * phi.view(1, -1)).sum(dim=1, keepdim=True)
        return center.view(-1, 1, 1, 1)

    def _neighbor_mask_from_center(self, center_phi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        B = center_phi.shape[0]
        S = phi.numel()
        if self.topk <= 0 or self.topk >= S:
            return torch.ones(B, S, 1, 1, 1, device=center_phi.device, dtype=center_phi.dtype)
        dist = (phi.view(1, S) - center_phi.view(B, 1)).abs()
        idx = torch.topk(dist, k=self.topk, dim=1, largest=False).indices
        mask = torch.zeros(B, S, device=center_phi.device, dtype=center_phi.dtype)
        mask.scatter_(1, idx, 1.0)
        return mask.view(B, S, 1, 1, 1)

    def forward(self, fs_slices, x_rgb, gt_mask=None, return_aux=False):
        if fs_slices.shape[0] == self.S and fs_slices.shape[1] != self.S:
            fs_slices = fs_slices.permute(1, 0, 2, 3, 4).contiguous()

        B, S, C, H, W = fs_slices.shape
        phi = self.slice_coords(device=fs_slices.device, dtype=fs_slices.dtype)
        fs_ref = self.slice_attn(fs_slices, phi=phi)
        fs_sum = fs_ref.mean(dim=1)
        ctx = torch.cat([fs_sum, x_rgb], dim=1)

        mu_logits = self.mu_head(ctx)
        mu_01 = torch.sigmoid(mu_logits)
        phi_min = phi[0].view(1, 1, 1, 1)
        phi_max = phi[-1].view(1, 1, 1, 1)
        mu_phi = phi_min + mu_01 * (phi_max - phi_min)

        tau_logits = self.tau_head(ctx)
        tau_map = F.softplus(tau_logits) + self.tau_min
        tau_map = torch.clamp(tau_map, self.tau_min, self.tau_max)

        phi_view = phi.view(1, S, 1, 1)
        logits_data = -(phi_view - mu_phi).abs() / (tau_map + 1e-6)

        region_pred = torch.sigmoid(self.region_head(ctx))
        alpha_region_pred, sim_map = self._pool_slice_distribution(fs_ref, x_rgb, region_pred)
        phi_region_pred = self._center_phi_from_alpha(alpha_region_pred, phi)

        route_loss = torch.zeros((), device=fs_slices.device, dtype=fs_slices.dtype)
        alpha_region_gt = None
        phi_region_gt = None
        gt_ds = self._resize_mask(gt_mask, size_hw=(H, W))

        if gt_ds is not None:
            with torch.no_grad():
                alpha_region_gt, _ = self._pool_slice_distribution(fs_ref.detach(), x_rgb.detach(), gt_ds)
                phi_region_gt = self._center_phi_from_alpha(alpha_region_gt, phi)
            route_loss = F.kl_div((alpha_region_pred.clamp_min(1e-8)).log(), alpha_region_gt, reduction='batchmean') + \
                         0.5 * F.binary_cross_entropy(region_pred, gt_ds)
            center_phi = phi_region_gt
            prior_strength = torch.sigmoid(self.gt_prior_logit).to(fs_slices.device).to(fs_slices.dtype)
        else:
            center_phi = phi_region_pred
            prior_strength = torch.sigmoid(self.infer_prior_logit).to(fs_slices.device).to(fs_slices.dtype)

        radius = self._radius(device=fs_slices.device, dtype=fs_slices.dtype)
        prior_logits = -(phi_view - center_phi).abs() / (radius + 1e-6)
        logits_total = logits_data + prior_strength * prior_logits
        alpha = torch.softmax(logits_total, dim=1).unsqueeze(2)

        if self.use_neighbor_mask and self.topk > 0 and self.topk < S:
            nb_mask = self._neighbor_mask_from_center(center_phi, phi)
            alpha = alpha * nb_mask
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-6)
        else:
            nb_mask = None

        x_focal = torch.sum(alpha * fs_ref, dim=1)
        if return_aux:
            aux = {
                'alpha': alpha,
                'mu_phi': mu_phi.detach(),
                'tau_map': tau_map.detach(),
                'phi_coords': phi.detach(),
                'region_pred': region_pred,
                'alpha_region_pred': alpha_region_pred.detach(),
                'phi_region_pred': phi_region_pred.detach(),
                'alpha_region_gt': alpha_region_gt.detach() if alpha_region_gt is not None else None,
                'phi_region_gt': phi_region_gt.detach() if phi_region_gt is not None else None,
                'prior_strength': prior_strength.detach(),
                'prior_radius': radius.detach(),
                'route_loss': route_loss,
                'sim_map_mean': sim_map.mean().detach(),
                'neighbor_mask': nb_mask.detach() if nb_mask is not None else None,
            }
            return x_focal, alpha, aux
        return x_focal, alpha


class TokenMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(dim, int(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TriSimplicialHyperedgeBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = None, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        if num_heads is None:
            num_heads = max(1, dim // 64)
        if dim % num_heads != 0:
            num_heads = 1
        self.num_heads = int(num_heads)

        hidden_gate = max(16, dim // 2)
        self.hyperedge_proj = nn.Linear(dim * 6 + 1, dim)
        self.hyperedge_gate = nn.Sequential(
            nn.Linear(dim * 6 + 2, hidden_gate),
            nn.GELU(),
            nn.Linear(hidden_gate, 1)
        )

        self.norm_h1 = nn.LayerNorm(dim)
        self.hyper_self_attn = nn.MultiheadAttention(dim, self.num_heads, dropout=dropout, batch_first=True)
        self.norm_h2 = nn.LayerNorm(dim)
        self.ffn_h = TokenMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        self.norm_r1 = nn.LayerNorm(dim)
        self.norm_f1 = nn.LayerNorm(dim)
        self.norm_d1 = nn.LayerNorm(dim)
        self.norm_hk = nn.LayerNorm(dim)
        self.rgb_to_h = nn.MultiheadAttention(dim, self.num_heads, dropout=dropout, batch_first=True)
        self.focal_to_h = nn.MultiheadAttention(dim, self.num_heads, dropout=dropout, batch_first=True)
        self.depth_to_h = nn.MultiheadAttention(dim, self.num_heads, dropout=dropout, batch_first=True)

        self.norm_r2 = nn.LayerNorm(dim)
        self.norm_f2 = nn.LayerNorm(dim)
        self.norm_d2 = nn.LayerNorm(dim)
        self.ffn_r = TokenMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.ffn_f = TokenMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.ffn_d = TokenMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, r, f, d, focus_conf):
        B, N, C = r.shape
        global_conf = focus_conf.mean(dim=1, keepdim=True).expand(B, N, 1)
        base = torch.cat([r, f, d, (r - f).abs(), (r - d).abs(), (f - d).abs()], dim=-1)
        h_seed = self.hyperedge_proj(torch.cat([base, focus_conf], dim=-1))
        h_gate = torch.sigmoid(self.hyperedge_gate(torch.cat([base, focus_conf, global_conf], dim=-1)))
        h = h_seed * h_gate

        h = h + self.hyper_self_attn(self.norm_h1(h), self.norm_h1(h), self.norm_h1(h), need_weights=False)[0]
        h = h + self.ffn_h(self.norm_h2(h))

        hk = self.norm_hk(h)
        r = r + self.rgb_to_h(self.norm_r1(r), hk, hk, need_weights=False)[0]
        f = f + self.focal_to_h(self.norm_f1(f), hk, hk, need_weights=False)[0]
        d = d + self.depth_to_h(self.norm_d1(d), hk, hk, need_weights=False)[0]

        r = r + self.ffn_r(self.norm_r2(r))
        f = f + self.ffn_f(self.norm_f2(f))
        d = d + self.ffn_d(self.norm_d2(d))

        aux = {
            'hyperedge_gate_mean': h_gate.mean().detach(),
            'hyperedge_gate_max': h_gate.max().detach(),
            'focus_conf_mean': focus_conf.mean().detach(),
        }
        return r, f, d, h, aux


class TriModalHyperedgeAggregator(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(16, dim // 2)
        self.fuse_logits = nn.Sequential(
            nn.Linear(dim * 4 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3)
        )
        self.out_proj = nn.Linear(dim * 3, dim)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn_out = TokenMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, r, f, d, h, focus_conf):
        tri_consensus = (r + f + d) / 3.0
        logits = self.fuse_logits(torch.cat([r, f, d, h, focus_conf], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        mix = weights[..., 0:1] * r + weights[..., 1:2] * f + weights[..., 2:3] * d
        y = self.out_proj(torch.cat([mix, h, tri_consensus], dim=-1))
        y = y + self.ffn_out(self.norm_out(y))
        return y, {'modal_weights_mean': weights.mean(dim=(0, 1)).detach()}


class PSRTriModalFusionStage(nn.Module):
    def __init__(self, dim, fea_reso, num_slices=12, topk=3, region_temperature=0.7,
                 hyper_blocks=1, hyper_heads=None, hyper_mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.dim = int(dim)
        self.fea_reso = int(fea_reso)
        self.S = int(num_slices)

        self.selector = GTGuidedPhysicalSliceSelector(
            dim=dim, num_slices=num_slices, region_temperature=region_temperature, topk=topk
        )
        self.rgb_proj = nn.Linear(dim, dim)
        self.focal_proj = nn.Linear(dim, dim)
        self.depth_proj = nn.Linear(dim, dim)

        self.blocks = nn.ModuleList([
            TriSimplicialHyperedgeBlock(dim, num_heads=hyper_heads, mlp_ratio=hyper_mlp_ratio, dropout=dropout)
            for _ in range(int(hyper_blocks))
        ])
        self.aggregator = TriModalHyperedgeAggregator(dim, mlp_ratio=hyper_mlp_ratio, dropout=dropout)

    def _fsseq_to_slices(self, fs_seq: torch.Tensor, B: int, N: int, C: int, H: int, W: int):
        BS = fs_seq.shape[0]
        if BS == B:
            return PatchToImage(fs_seq).unsqueeze(1)
        assert BS % B == 0
        S = BS // B
        assert S == self.S
        return PatchToImage(fs_seq).view(B, S, C, H, W)

    def _alpha_to_focus_conf(self, alpha: torch.Tensor):
        p = alpha.clamp_min(1e-8)
        ent = -(p * p.log()).sum(dim=1) / math.log(self.S)
        conf = 1.0 - ent
        return conf.clamp(0.0, 1.0)

    def forward(self, rgb_seq, fs_seq, depth_seq, gt=None, return_aux=False):
        B, N, C = rgb_seq.shape
        H = W = self.fea_reso
        x_rgb = PatchToImage(rgb_seq)
        fs_slices = self._fsseq_to_slices(fs_seq, B, N, C, H, W)

        x_focal, alpha, selector_aux = self.selector(fs_slices, x_rgb, gt_mask=gt, return_aux=True)
        focus_conf = self._alpha_to_focus_conf(alpha)
        focus_conf_seq = ImageToPatch(focus_conf)

        r = self.rgb_proj(rgb_seq)
        f = self.focal_proj(ImageToPatch(x_focal))
        d = self.depth_proj(depth_seq)

        stage_block_aux = []
        h = (r + f + d) / 3.0
        for blk in self.blocks:
            r, f, d, h, blk_aux = blk(r, f, d, focus_conf_seq)
            stage_block_aux.append(blk_aux)

        fused_seq, agg_aux = self.aggregator(r, f, d, h, focus_conf_seq)
        if return_aux:
            aux = {
                'alpha': alpha,
                'focus_conf': focus_conf.detach(),
                'route_loss': selector_aux['route_loss'],
                'selector': selector_aux,
                'hyperedge_blocks': stage_block_aux,
                'aggregator': agg_aux,
            }
            return fused_seq, aux
        return fused_seq


class EulerElasticaResidualBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        if self.dim % max(1, num_heads) != 0:
            num_heads = 1
        self.num_heads = int(max(1, num_heads))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, self.num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = TokenMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        self.scalar_proj = nn.Conv2d(dim, 1, kernel_size=1)
        self.geom_res = nn.Sequential(
            nn.Conv2d(dim + 2, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        self.grad = SpatialGradient()
        self.attn_scale = nn.Parameter(torch.tensor(1.0))
        self.ffn_scale = nn.Parameter(torch.tensor(1.0))
        self.geom_scale = nn.Parameter(torch.tensor(0.5))

    def _geometry(self, b: torch.Tensor):
        gx, gy = self.grad(b)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        nx = gx / (grad_mag + 1e-6)
        ny = gy / (grad_mag + 1e-6)
        dnx_dx, _ = self.grad(nx)
        _, dny_dy = self.grad(ny)
        curv = dnx_dx + dny_dy
        return grad_mag, curv

    def forward(self, x: torch.Tensor):
        x = x + self.attn_scale * self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.ffn_scale * self.ffn(self.norm2(x))
        img = PatchToImage(x)
        boundary = torch.sigmoid(self.scalar_proj(img))
        grad_mag, curv = self._geometry(boundary)
        geom = self.geom_res(torch.cat([img, grad_mag, curv.abs()], dim=1))
        img = img + self.geom_scale * geom
        x = ImageToPatch(img)
        elastica = ((0.5 + 1.0 * curv.pow(2)) * grad_mag).mean()
        aux = {
            'grad_mean': grad_mag.mean().detach(),
            'curvature_mean': curv.abs().mean().detach(),
            'elastica_mean': elastica.detach(),
            'geom_scale': self.geom_scale.detach(),
        }
        return x, aux


class EulerElasticaEvolutionStep(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 1.0):
        super().__init__()
        self.channels = int(channels)
        hidden = max(self.channels, int(self.channels * hidden_ratio))
        self.grad = SpatialGradient()

        # feature + boundary + grad + |curv| + energy + prior residual
        in_ch = self.channels + 5
        self.delta_head = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        self.feat_update = nn.Sequential(
            nn.Conv2d(self.channels + 2, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, self.channels, kernel_size=3, padding=1),
        )
        self.step_scale = nn.Parameter(torch.tensor(0.5))
        self.feat_scale = nn.Parameter(torch.tensor(0.5))

    def _geometry(self, b: torch.Tensor):
        gx, gy = self.grad(b)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        nx = gx / (grad_mag + 1e-6)
        ny = gy / (grad_mag + 1e-6)
        dnx_dx, _ = self.grad(nx)
        _, dny_dy = self.grad(ny)
        curv = dnx_dx + dny_dy
        return grad_mag, curv

    def forward(self, feat: torch.Tensor, boundary: torch.Tensor, prior: torch.Tensor):
        grad_mag, curv = self._geometry(boundary)
        energy = (0.5 + curv.pow(2)) * grad_mag
        prior_res = prior - boundary

        state = torch.cat([feat, boundary, grad_mag, curv.abs(), energy, prior_res], dim=1)
        delta = torch.tanh(self.delta_head(state))
        gate = torch.sigmoid(self.gate_head(state))
        eta = F.softplus(self.step_scale)

        boundary_next = torch.clamp(boundary + eta * gate * delta, 0.0, 1.0)
        feedback = self.feat_update(torch.cat([feat, boundary_next, energy], dim=1))
        feat_next = feat + F.softplus(self.feat_scale) * feedback

        elastica = ((0.5 + curv.pow(2)) * grad_mag).mean()
        update_mean = (eta * gate * delta).abs().mean()
        aux = {
            'grad_mean': grad_mag.mean().detach(),
            'curvature_mean': curv.abs().mean().detach(),
            'elastica_mean': elastica.detach(),
            'update_mean': update_mean.detach(),
            'step_scale': eta.detach(),
            'gate_mean': gate.mean().detach(),
            'elastica_loss': elastica,
        }
        return feat_next, boundary_next, aux


class EulerElasticaBoundaryTransformer(nn.Module):
    def __init__(self, channels: int, token_hw: int = 14, boundary_blocks: int = 2,
                 boundary_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0,
                 morph_ks: int = 3, mask_smooth_ks: int = 5, mask_dilate_ks: int = 7,
                 evolution_steps: int = 3, evolution_hidden_ratio: float = 1.0):
        super().__init__()
        self.channels = int(channels)
        self.token_hw = int(token_hw)
        self.morph = MorphologicalGradient(k=morph_ks)
        self.grad = SpatialGradient()
        self.mask_smooth_ks = int(mask_smooth_ks)
        self.mask_dilate_ks = int(mask_dilate_ks)
        self.evolution_steps = int(evolution_steps)

        self.rgb_ctx = Conv3(3, channels)
        self.depth_ctx = Conv3(1, channels)
        self.seed_fuse = Conv3(channels * 3 + 1, channels)
        self.token_pool = nn.AdaptiveAvgPool2d((self.token_hw, self.token_hw))

        self.blocks = nn.ModuleList([
            EulerElasticaResidualBlock(channels, num_heads=boundary_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(int(boundary_blocks))
        ])

        self.boundary_score = nn.Conv2d(channels, 1, kernel_size=1)
        self.prior_to_feat = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.fuse_context_region = Conv3(2 * channels, channels)
        self.contour = ScoreModule(channels)
        self.out_scale = nn.Parameter(torch.tensor(0.5))

        self.evolution = nn.ModuleList([
            EulerElasticaEvolutionStep(channels, hidden_ratio=evolution_hidden_ratio)
            for _ in range(self.evolution_steps)
        ])

    def _soft_coarse_mask(self, coarse_prob):
        x = coarse_prob
        if self.mask_smooth_ks > 1:
            k = self.mask_smooth_ks
            x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        if self.mask_dilate_ks > 1:
            k = self.mask_dilate_ks
            x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        return x.clamp(0.0, 1.0)

    def _geometry(self, b):
        gx, gy = self.grad(b)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        nx = gx / (grad_mag + 1e-6)
        ny = gy / (grad_mag + 1e-6)
        dnx_dx, _ = self.grad(nx)
        _, dny_dy = self.grad(ny)
        curv = dnx_dx + dny_dy
        return grad_mag, curv

    def forward(self, fused_fea_img, rgb, depth, coarse_logits):
        coarse_prob = torch.sigmoid(coarse_logits)
        coarse_mask = self._soft_coarse_mask(coarse_prob)

        edge0 = self.morph(coarse_mask)
        rgb_ctx = self.rgb_ctx(rgb)
        depth_ctx = self.depth_ctx(depth)
        seed = self.seed_fuse(torch.cat([fused_fea_img, rgb_ctx * coarse_mask, depth_ctx * coarse_mask, edge0], dim=1))

        token_img = self.token_pool(seed)
        tokens = ImageToPatch(token_img)

        block_aux = []
        for blk in self.blocks:
            tokens, aux = blk(tokens)
            block_aux.append(aux)

        token_img = PatchToImage(tokens)
        boundary_low = self.boundary_score(token_img)
        boundary_low_prob = torch.sigmoid(boundary_low)
        boundary_init = torch.clamp(
            edge0 + self.out_scale * F.interpolate(boundary_low_prob, size=edge0.shape[-2:], mode='bilinear', align_corners=True),
            0.0, 1.0
        )

        boundary_feat = F.interpolate(token_img, size=fused_fea_img.shape[-2:], mode='bilinear', align_corners=True)
        boundary_feat = boundary_feat + seed
        edge_feature = self.fuse_context_region(torch.cat([fused_fea_img, boundary_feat], dim=1))

        boundary_prob = boundary_init
        evo_aux = []
        evolution_elastica = torch.zeros((), device=fused_fea_img.device, dtype=fused_fea_img.dtype)
        evolution_update = torch.zeros((), device=fused_fea_img.device, dtype=fused_fea_img.dtype)
        for step in self.evolution:
            edge_feature, boundary_prob, step_aux = step(edge_feature, boundary_prob, prior=edge0)
            evo_aux.append({k: (v.detach() if torch.is_tensor(v) else v) for k, v in step_aux.items() if k != 'elastica_loss'})
            evolution_elastica = evolution_elastica + step_aux['elastica_loss']
            evolution_update = evolution_update + step_aux['update_mean'].to(fused_fea_img.dtype)
        if self.evolution_steps > 0:
            evolution_elastica = evolution_elastica / float(self.evolution_steps)
            evolution_update = evolution_update / float(self.evolution_steps)

        edge_feature = edge_feature + self.prior_to_feat(boundary_prob)
        contour_logits = self.contour(edge_feature)

        token_grad, token_curv = self._geometry(boundary_low_prob)
        final_grad, final_curv = self._geometry(boundary_prob)
        token_elastica = ((0.5 + 1.0 * token_curv.pow(2)) * token_grad).mean()
        final_elastica = ((0.5 + 1.0 * final_curv.pow(2)) * final_grad).mean()
        residual_energy = (boundary_prob - edge0).abs().mean()
        edge_elastica_loss = 0.2 * token_elastica + 0.6 * evolution_elastica + 1.0 * final_elastica + 0.2 * residual_energy

        aux = {
            'edge_prior_mean': boundary_prob.mean().detach(),
            'edge0_mean': edge0.mean().detach(),
            'token_grad_mean': token_grad.mean().detach(),
            'token_curvature_mean': token_curv.abs().mean().detach(),
            'token_elastica_mean': token_elastica.detach(),
            'evolution_elastica_mean': evolution_elastica.detach(),
            'evolution_update_mean': evolution_update.detach(),
            'final_grad_mean': final_grad.mean().detach(),
            'final_curvature_mean': final_curv.abs().mean().detach(),
            'final_elastica_mean': final_elastica.detach(),
            'boundary_residual_mean': residual_energy.detach(),
            'residual_scale': self.out_scale.detach(),
            'geom_scale_mean': torch.stack([a['geom_scale'] for a in block_aux]).mean().detach() if len(block_aux) > 0 else torch.tensor(0.0, device=fused_fea_img.device),
            'edge_elastica_loss': edge_elastica_loss,
            'evolution_aux': evo_aux,
        }
        return edge_feature, contour_logits, aux



def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PSRNet(nn.Module):
    def __init__(self, backbone_type='swin', num_slices=12, topk=3,
                 region_temperature=0.7, hyper_blocks=1, hyper_heads=None,
                 hyper_mlp_ratio=2.0, dropout=0.0,
                 boundary_blocks=2, boundary_heads=4, boundary_token_hw=14,
                 evolution_steps=3, evolution_hidden_ratio=1.0):
        super().__init__()
        assert backbone_type == 'swin'
        self.num_slices = int(num_slices)
        img_size = 224
        depths = [2, 2, 6, 2]
        patch_size = 4
        embed_dim = 96
        self.channels = 96

        self.backbone_rgb = SwinTransformerBackbone(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            embed_dim=embed_dim, depths=depths, num_heads=[3, 6, 12, 24], window_size=7
        )
        self.backbone_fs = SwinTransformerBackbone(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            embed_dim=embed_dim, depths=depths, num_heads=[3, 6, 12, 24], window_size=7
        )
        self.backbone_depth = SwinTransformerBackbone(
            img_size=img_size, patch_size=patch_size, in_chans=1,
            embed_dim=embed_dim, depths=depths, num_heads=[3, 6, 12, 24], window_size=7
        )

        self.image_size = img_size
        self.num_layers = len(depths)
        self.patch_reso = img_size // patch_size

        self.cm_module = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i_layer in range(self.num_layers):
            fea_reso = self.patch_reso // (2 ** i_layer)
            dim = (2 ** i_layer) * embed_dim
            fusion_layer = PSRTriModalFusionStage(
                dim=dim, fea_reso=fea_reso, num_slices=self.num_slices, topk=topk,
                region_temperature=region_temperature, hyper_blocks=hyper_blocks,
                hyper_heads=hyper_heads, hyper_mlp_ratio=hyper_mlp_ratio, dropout=dropout
            )
            self.cm_module.append(fusion_layer)
            self.upsample.append(PatchExpand(
                input_resolution=[fea_reso, fea_reso],
                in_dim=dim, out_dim=int(dim / 2), norm_layer=nn.LayerNorm
            ))

        self.cm_module = self.cm_module[::-1]
        self.upsample = self.upsample[::-1]
        self.upsample_x4 = FinalPatchExpand_X4(
            input_resolution=[self.patch_reso, self.patch_reso],
            dim=embed_dim, dim_scale=4, norm_layer=nn.LayerNorm
        )

        self.score_module = ScoreModule(self.channels)
        self.score_module_coarse = ScoreModule(self.channels)
        self.edge_head = EulerElasticaBoundaryTransformer(
            channels=self.channels,
            token_hw=boundary_token_hw,
            boundary_blocks=boundary_blocks,
            boundary_heads=boundary_heads,
            mlp_ratio=hyper_mlp_ratio,
            dropout=dropout,
            evolution_steps=evolution_steps,
            evolution_hidden_ratio=evolution_hidden_ratio,
        )
        self.fuse_edge_region = Conv3(2 * self.channels, self.channels)

    def load_pretrained(self, load_path):
        if not os.path.exists(load_path):
            print("[WARN] pretrained model path not exist:", load_path)
            return
        pretrained = torch.load(load_path, map_location='cpu')
        pretrained_dict = pretrained['model'] if isinstance(pretrained, dict) and 'model' in pretrained else pretrained

        def _load_backbone(backbone):
            model_dict = backbone.state_dict()
            renamed_dict = {}
            for k, v in pretrained_dict.items():
                k2 = k.replace('layers.0.downsample', 'downsamples.0')
                k2 = k2.replace('layers.1.downsample', 'downsamples.1')
                k2 = k2.replace('layers.2.downsample', 'downsamples.2')
                if k2 in model_dict:
                    renamed_dict[k2] = v
            model_dict.update(renamed_dict)
            backbone.load_state_dict(model_dict, strict=True)

        _load_backbone(self.backbone_rgb)
        _load_backbone(self.backbone_fs)

        model_dict = self.backbone_depth.state_dict()
        renamed_dict = {}
        for k, v in pretrained_dict.items():
            k2 = k.replace('layers.0.downsample', 'downsamples.0')
            k2 = k2.replace('layers.1.downsample', 'downsamples.1')
            k2 = k2.replace('layers.2.downsample', 'downsamples.2')
            if k2 in model_dict and tuple(model_dict[k2].shape) == tuple(v.shape):
                renamed_dict[k2] = v
        model_dict.update(renamed_dict)
        self.backbone_depth.load_state_dict(model_dict, strict=False)
        print("RGB/FS/Depth pretrained loaded.")

    def _normalize_fs_input(self, fs, B: int):
        assert fs.dim() in (4, 5), f"Unsupported fs shape: {fs.shape}"
        if fs.dim() == 5:
            b2, S, c, h, w = fs.shape
            assert b2 == B and c == 3
            return fs.view(B * S, 3, h, w).contiguous(), int(S)
        assert fs.size(1) == 3
        if fs.shape[0] == self.num_slices and B == 1:
            return fs.contiguous(), int(self.num_slices)
        BS = fs.shape[0]
        assert BS % B == 0
        S = BS // B
        return fs.contiguous(), int(S)

    def forward(self, fs, rgb, depth, gt=None, return_aux=False):
        assert rgb.dim() == 4 and rgb.size(1) == 3
        assert depth is not None and depth.dim() == 4 and depth.size(1) == 1
        B = rgb.shape[0]
        if gt is not None:
            assert gt.dim() == 4 and gt.size(1) == 1

        fs_in, S = self._normalize_fs_input(fs, B=B)
        assert S == self.num_slices

        side_rgb_x = self.backbone_rgb(rgb)[::-1]
        side_fs_x = self.backbone_fs(fs_in)[::-1]
        side_depth_x = self.backbone_depth(depth)[::-1]

        aux_all = [] if return_aux else None
        total_route_loss = torch.zeros((), device=rgb.device, dtype=rgb.dtype)

        if return_aux:
            fused_fea, aux0 = self.cm_module[0](side_rgb_x[0], side_fs_x[0], side_depth_x[0], gt=gt, return_aux=True)
            aux_all.append(aux0)
            total_route_loss = total_route_loss + aux0['route_loss']
        else:
            fused_fea = self.cm_module[0](side_rgb_x[0], side_fs_x[0], side_depth_x[0], gt=gt, return_aux=False)

        for i in range(1, self.num_layers):
            fused_fea = self.upsample[i - 1](fused_fea)
            rgb_in = side_rgb_x[i] + fused_fea
            if return_aux:
                fused_fea, auxi = self.cm_module[i](rgb_in, side_fs_x[i], side_depth_x[i], gt=gt, return_aux=True)
                aux_all.append(auxi)
                total_route_loss = total_route_loss + auxi['route_loss']
            else:
                fused_fea = self.cm_module[i](rgb_in, side_fs_x[i], side_depth_x[i], gt=gt, return_aux=False)

        fused_fea_img = self.upsample_x4(fused_fea)
        coarse = self.score_module_coarse(fused_fea_img)
        edge_feature, contour, edge_aux = self.edge_head(fused_fea_img, rgb, depth, coarse)
        final_feat = self.fuse_edge_region(torch.cat((edge_feature, fused_fea_img), dim=1))
        pred = self.score_module(final_feat)

        if return_aux:
            aux = {'stage_aux': aux_all, 'route_loss': total_route_loss, 'edge_aux': edge_aux}
            return pred, contour, coarse, aux
        return pred, contour, coarse


MPNet = PSRNet
MNet = PSRNet


if __name__ == "__main__":
    net = PSRNet(num_slices=12, topk=3, region_temperature=0.7, hyper_blocks=1, boundary_blocks=2, evolution_steps=3)
    net.apply(init_weights)
    print("Params (M):", sum(p.numel() for p in net.parameters()) / 1e6)
    rgb = torch.randn(1, 3, 224, 224)
    depth = torch.randn(1, 1, 224, 224)
    fs = torch.randn(12, 3, 224, 224)
    gt = torch.randint(0, 2, (1, 1, 224, 224)).float()
    with torch.no_grad():
        pred, contour, coarse, aux = net(fs, rgb, depth, gt=gt, return_aux=True)
        print(pred.shape, contour.shape, coarse.shape)
        print("route_loss:", float(aux['route_loss']))
        print("edge_elastica_loss:", float(aux['edge_aux']['edge_elastica_loss']))
