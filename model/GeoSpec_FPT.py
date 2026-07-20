import math
from typing import Any, Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor



class AffineInstanceNorm(nn.Module):
    """Instance Normalization with geometric confidence modulation.

    Applies InstanceNorm2d to aggressively strip scanner-specific style
    statistics, then applies confidence-modulated affine:
        out = IN(x) * (gamma + beta * C_conf) + delta

    At init: gamma=1, beta=0, delta=0  =>  IN(x) * 1 = IN(x).
    As training progresses, beta learns to spatially modulate based on
    the hyperbolic manifold's conformal confidence.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.in_norm = nn.InstanceNorm2d(
            num_channels, affine=False, track_running_stats=False
        )
        self.weight = nn.Parameter(torch.ones(num_channels))   # gamma
        self.beta = nn.Parameter(torch.zeros(num_channels))    # beta (confidence)
        self.bias = nn.Parameter(torch.zeros(num_channels))    # delta

    def forward(
        self, x: torch.Tensor, conf: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.in_norm(x)
        gamma = self.weight.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        delta = self.bias.view(1, -1, 1, 1)
        if conf is not None:
            return x * (gamma + beta * conf) + delta
        return x * gamma + delta

class ClassifierHead(nn.Module):
    """Simple PEFT classification head: LayerNorm -> GELU -> Dropout -> Linear."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, num_classes)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return self.proj(x)

class GradReverseFunction(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) for domain adversarial training."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


class DomainClassifier(nn.Module):
    """Small 2-layer MLP domain classifier with Gradient Reversal Layer."""

    def __init__(
        self,
        input_size: int,
        num_domains: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.grl_lambda = grl_lambda
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_domains)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = GradReverseFunction.apply(x, self.grl_lambda)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ===========================================================================
#  Multi-Scale CNN Backbone (ConvNeXtV2) with Selective Layer Freezing
# ===========================================================================
class MultiScaleConvNeXtV2Backbone(nn.Module):
    _STAGE_PREFIXES = ("blocks", "stages", "layers")
    _STEM_NAMES = ("stem",)

    def __init__(
        self,
        model_name: str,
        in_chans: int = 3,
        pretrained: bool = True,
        backbone_trainable_layers: Optional[List[int]] = None,
    ):
        super().__init__()

        if backbone_trainable_layers is None:
            backbone_trainable_layers = []

        self.trainable_set: set = set(int(i) for i in backbone_trainable_layers)

        self.cnn_backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        trainable_count = 0
        total_count = 0
        for name, param in self.cnn_backbone.named_parameters():
            total_count += param.numel()
            param.requires_grad = self._is_trainable_param(name)
            if param.requires_grad:
                trainable_count += param.numel()

        self.channels = self.cnn_backbone.feature_info.channels()

    def _find_stage_index(self, parts: List[str]) -> Optional[int]:
        for i in range(len(parts) - 1):
            if parts[i] in self._STAGE_PREFIXES:
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    def _is_stem_param(self, parts: List[str]) -> bool:
        for part in parts:
            if part in self._STAGE_PREFIXES:
                return False
            if part in self._STEM_NAMES:
                return True
        return False

    def _is_trainable_param(self, name: str) -> bool:
        parts = name.split(".")
        if self._is_stem_param(parts):
            return 0 in self.trainable_set

        stage_module_idx = self._find_stage_index(parts)
        if stage_module_idx is not None:
            feature_idx = stage_module_idx + 1
            return feature_idx in self.trainable_set

        return False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._set_frozen_to_eval()
        return self

    def _set_frozen_to_eval(self):
        self.cnn_backbone.eval()
        for name, module in self.cnn_backbone.named_modules():
            parts = name.split(".")
            if self._is_stem_param(parts) and 0 in self.trainable_set:
                module.train()
                continue

            stage_module_idx = self._find_stage_index(parts)
            if stage_module_idx is not None:
                feature_idx = stage_module_idx + 1
                if feature_idx in self.trainable_set:
                    module.train()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.cnn_backbone(x)

class UnifiedManifoldFusion(nn.Module):
    """Unified Manifold Fusion: couples Haar wavelet (frequency) and Poincaré
    (geometric) domains via a learned Chart-Transition tensor Psi.

    Updated:
    - Replaces Euclidean sum with Einstein Midpoint (Fréchet Mean) for true chart transition.
    - Softens coarse HH suppression with a learnable Bayesian parameter.
    """

    def __init__(
        self,
        low_level_channels: int,
        high_level_channels: int,
        output_channels: int,
        tucker_rank: int = 16,
        input_size: int = 384,
        low_stage_idx: int = 0,
        high_stage_idx: int = 2,
        c_min: float = 0.1,
        c_max: float = 10.0,
        curv_reg_weight: float = 1e-3,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.low_level_channels = low_level_channels
        self.high_level_channels = high_level_channels
        self.output_channels = output_channels
        self.r = tucker_rank
        self.c_min = c_min
        self.c_max = c_max
        self.curv_reg_weight = curv_reg_weight
        self.eps = eps

        C_low = low_level_channels
        C_high = high_level_channels
        C_out = output_channels
        r = tucker_rank

        # ---- DWT Level Validation (strict, no fallback) ----
        low_size = input_size // (2 ** (low_stage_idx + 2))
        high_size = input_size // (2 ** (high_stage_idx + 2))
        if low_size <= 0 or high_size <= 0:
            raise ValueError(f"Invalid spatial sizes: low_size={low_size}, high_size={high_size}.")
        if low_size < high_size:
            raise ValueError(f"low_size ({low_size}) must be >= high_size ({high_size}).")
        ratio = low_size / high_size
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"Spatial ratio {ratio} is not an integer.")
        ratio_int = int(round(ratio))
        L = 0
        temp = ratio_int
        while temp > 1:
            if temp % 2 != 0:
                raise ValueError("Spatial ratio is not a power of 2.")
            temp //= 2
            L += 1
        temp_h = low_size
        for _ in range(L):
            if temp_h % 2 != 0:
                raise ValueError("Intermediate spatial size is not even.")
            temp_h //= 2
        if temp_h != high_size:
            raise ValueError("DWT level mismatch.")
        self.L = L
        assert self.L >= 1, "At least one DWT level is required."

        # ================================================================
        # Frequency Path
        # ================================================================
        self.high_align = nn.Conv2d(C_high, C_low, kernel_size=1, bias=False)
        spatial_in_dim = 2 * C_low
        self.mlp_spatial = nn.Sequential(
            nn.Linear(spatial_in_dim, max(spatial_in_dim // 4, 32)),
            nn.GELU(),
            nn.Linear(max(spatial_in_dim // 4, 32), r * r * C_out),
        )
        self.spectral_conv = nn.Conv2d(C_low, r * r * C_out, kernel_size=1, bias=False)
        self.U_high = nn.Conv2d(C_low, r, kernel_size=1, bias=False)
        self.U_LL = nn.Conv2d(C_low, r, kernel_size=1, bias=False)

        self.band_proj_lh = nn.Conv2d(C_low, C_out, kernel_size=1, bias=False)
        self.band_proj_hl = nn.Conv2d(C_low, C_out, kernel_size=1, bias=False)
        self.band_proj_hh = nn.Conv2d(C_low, C_out, kernel_size=1, bias=False)

        band_in_dim = 3 * C_low
        self.mlp_bands = nn.Sequential(
            nn.Linear(band_in_dim, max(band_in_dim // 4, 32)),
            nn.GELU(),
            nn.Linear(max(band_in_dim // 4, 32), 3 * C_out),
        )
        self.theta_steer = nn.Parameter(torch.zeros(1))

        self.coarse_prior_logits = nn.Parameter(torch.zeros(self.L, 3))
        if self.L > 0:
            self.coarse_prior_logits.data[0, 2] = -1.0

        # ================================================================
        # Hyperbolic Path
        # ================================================================
        init_theta_c = math.log(math.expm1(1.0))
        self.theta_c = nn.Parameter(torch.tensor(init_theta_c, dtype=torch.float32))
        self.W_low = nn.Conv2d(C_low, C_out, kernel_size=1, bias=False)
        self.W_high = nn.Conv2d(C_high, C_out, kernel_size=1, bias=False)
        self.tau_low = nn.Parameter(torch.ones(1))
        self.tau_high = nn.Parameter(torch.ones(1))
        self.g_low = nn.Conv2d(C_out, 1, kernel_size=1, bias=True)
        self.g_high = nn.Conv2d(C_out, 1, kernel_size=1, bias=True)

        self.P_c = nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)
        self.P_s = nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)

        self.register_buffer("rho_c", torch.tensor(0.5))
        self.register_buffer("rho_s", torch.tensor(0.1))

        # ================================================================
        # Chart-Transition Coupling (Psi) & Einstein Midpoint
        # ================================================================
        self.psi = nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)
        self.einstein_weight = nn.Parameter(torch.tensor(0.0))

        self.W_euc = nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)
        self.W_o_hyp = nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)
        self.W_o_euc = nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)
        self.gn_hyp = nn.GroupNorm(1, C_out)
        self.gn_euc = nn.GroupNorm(1, C_out)
        self.final_norm = nn.GroupNorm(1, C_out)

        init_phi = math.log(0.3 / 0.7)
        self.phi = nn.Parameter(torch.tensor(init_phi, dtype=torch.float32))
        self.act = nn.GELU()
        self.skip_proj = nn.Conv2d(C_low, C_out, kernel_size=1, bias=False)
        self.alpha_skip = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for conv in [self.high_align, self.U_high, self.U_LL, self.band_proj_lh,
                     self.band_proj_hl, self.band_proj_hh, self.spectral_conv,
                     self.W_low, self.W_high, self.W_euc, self.W_o_hyp, self.W_o_euc,
                     self.skip_proj]:
            nn.init.xavier_uniform_(conv.weight)
        for m in [*self.mlp_spatial.modules(), *self.mlp_bands.modules()]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.g_low.weight); nn.init.zeros_(self.g_low.bias)
        nn.init.zeros_(self.g_high.weight); nn.init.zeros_(self.g_high.bias)
        with torch.no_grad():
            eye = torch.eye(self.output_channels, device=self.P_c.weight.device, dtype=self.P_c.weight.dtype)
            self.P_c.weight.data.copy_(eye.view(self.output_channels, self.output_channels, 1, 1))
            self.P_s.weight.data.zero_()
            self.psi.weight.data.copy_(eye.view(self.output_channels, self.output_channels, 1, 1))
        for gn in [self.gn_hyp, self.gn_euc, self.final_norm]:
            nn.init.ones_(gn.weight)
            nn.init.zeros_(gn.bias)

    @staticmethod
    def _haar_dwt_2d(x: torch.Tensor):
        a = x[..., 0::2, 0::2]; b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]; d = x[..., 1::2, 1::2]
        LL = (a + b + c + d) * 0.5
        LH = (a - b + c - d) * 0.5
        HL = (a + b - c - d) * 0.5
        HH = (a - b - c + d) * 0.5
        return LL, LH, HL, HH

    @staticmethod
    def _haar_idwt_2d(ll, lh, hl, hh):
        a = (ll + lh + hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll - lh - hl + hh) * 0.5
        out = torch.stack([a, b, c, d], dim=-1)
        out = out.reshape(ll.shape[0], ll.shape[1], ll.shape[2], ll.shape[3], 2, 2)
        return out.permute(0, 1, 2, 4, 3, 5).reshape(ll.shape[0], ll.shape[1], ll.shape[2]*2, ll.shape[3]*2)

    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.theta_c) + self.eps

    def curvature_reg_loss(self) -> torch.Tensor:
        c = self.curvature
        return self.curv_reg_weight * (torch.relu(c - self.c_max).pow(2) + torch.relu(self.c_min - c).pow(2))

    def _expmap0(self, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v_norm = v.norm(dim=1, keepdim=True)
        sqrt_c = torch.sqrt(c)
        tanh_val = torch.tanh(torch.clamp(sqrt_c * v_norm, max=15.0))
        return tanh_val * v / (sqrt_c * (v_norm + self.eps))

    def _logmap0(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_norm = x.norm(dim=1, keepdim=True)
        sqrt_c = torch.sqrt(c)
        artanh_val = torch.atanh(torch.clamp(sqrt_c * x_norm, min=0.0, max=1.0 - self.eps))
        return artanh_val * x / (sqrt_c * (x_norm + self.eps))


    def _disentangle_norm_loss(self, x_lc, x_ls, x_hc, x_hs):
        content_norm = (x_lc.norm(dim=1, keepdim=True).mean() + x_hc.norm(dim=1, keepdim=True).mean()) / 2.0
        style_norm = (x_ls.norm(dim=1, keepdim=True).mean() + x_hs.norm(dim=1, keepdim=True).mean()) / 2.0
        return (content_norm - self.rho_c).pow(2) + (style_norm - self.rho_s).pow(2)

    def _einstein_midpoint(self, x1, x2, w1, w2, c):
        """Exact Einstein midpoint via Klein model projection."""
        xK1 = 2.0 * x1 / (1.0 + c * (x1 * x1).sum(dim=1, keepdim=True)).clamp_min(self.eps)
        xK2 = 2.0 * x2 / (1.0 + c * (x2 * x2).sum(dim=1, keepdim=True)).clamp_min(self.eps)
        gamma1 = 1.0 / torch.sqrt((1.0 - c * (xK1 * xK1).sum(dim=1, keepdim=True)).clamp_min(self.eps))
        gamma2 = 1.0 / torch.sqrt((1.0 - c * (xK2 * xK2).sum(dim=1, keepdim=True)).clamp_min(self.eps))
        w1_act = w1 * gamma1
        w2_act = w2 * gamma2
        sum_gamma_xK = w1_act * xK1 + w2_act * xK2
        sum_gamma = w1_act + w2_act
        m_Klein = sum_gamma_xK / sum_gamma.clamp_min(self.eps)
        m_P = m_Klein / (1.0 + torch.sqrt((1.0 - c * (m_Klein * m_Klein).sum(dim=1, keepdim=True)).clamp_min(self.eps)))
        return m_P

    def forward(self, low_level_feat, high_level_feat, wavelet_mask=None):
        B, C_low, H_low, W_low = low_level_feat.shape

        # 1. Frequency Path
        X_high_tilde = self.high_align(high_level_feat)
        LL = low_level_feat
        high_freq_bands = []
        for level_idx in range(self.L):
            LL, LH, HL, HH = self._haar_dwt_2d(LL)

            if wavelet_mask is not None:
                mask_l = wavelet_mask[:, :, level_idx]
                LH = LH * mask_l[:, 0:1].view(B, 1, 1, 1)
                HL = HL * mask_l[:, 1:2].view(B, 1, 1, 1)
                HH = HH * mask_l[:, 2:3].view(B, 1, 1, 1)

            high_freq_bands.append((LH, HL, HH))
        LL_L = LL

        g_spatial = torch.cat([F.adaptive_avg_pool2d(X_high_tilde, 1).flatten(1), F.adaptive_avg_pool2d(LL_L, 1).flatten(1)], dim=1)
        G = self.mlp_spatial(g_spatial).view(B, self.r, self.r, self.output_channels)
        G_spectral = self.spectral_conv(F.adaptive_avg_pool2d(LL_L, 1)).flatten(1).view(B, self.r, self.r, self.output_channels)
        G = G * G_spectral
        P_high = self.U_high(X_high_tilde)
        P_LL = self.U_LL(LL_L)
        LL_fused = torch.einsum("bijk,bihw,bjhw->bkhw", G, P_high, P_LL)

        cos_t = torch.cos(self.theta_steer)
        sin_t = torch.sin(self.theta_steer)
        zero = torch.zeros_like(cos_t)
        one = torch.ones_like(cos_t)
        row_0 = torch.stack([cos_t, -sin_t, zero])
        row_1 = torch.stack([sin_t,  cos_t, zero])
        row_2 = torch.stack([zero,   zero,  one])
        M_steer = torch.stack([row_0, row_1, row_2]).squeeze(-1)

        G_bands_list = []
        modulated_bands = []
        for l in range(self.L):
            LH_l, HL_l, HH_l = high_freq_bands[l]
            stacked = torch.cat([LH_l, HL_l, HH_l], dim=1)
            G_bands_raw = self.mlp_bands(F.adaptive_avg_pool2d(stacked, 1).flatten(1)).view(B, 3, self.output_channels)
            G_hat = torch.einsum("ij,bjk->bik", M_steer, G_bands_raw)
            G_hat = G_hat + self.coarse_prior_logits[l].view(1, 3, 1)
            G_hat = F.softmax(G_hat, dim=1)
            G_bands_list.append(G_hat)

            tilde_LH = self.band_proj_lh(LH_l) * G_hat[:, 0, :].unsqueeze(-1).unsqueeze(-1)
            tilde_HL = self.band_proj_hl(HL_l) * G_hat[:, 1, :].unsqueeze(-1).unsqueeze(-1)
            tilde_HH = self.band_proj_hh(HH_l) * G_hat[:, 2, :].unsqueeze(-1).unsqueeze(-1)
            modulated_bands.append((tilde_LH, tilde_HL, tilde_HH))

        current_LL = LL_fused
        for l in range(self.L - 1, -1, -1):
            current_LL = self._haar_idwt_2d(current_LL, *modulated_bands[l])

        u_freq = self.psi(current_LL)

        # 2. Hyperbolic Path
        high_level_up = F.interpolate(high_level_feat, size=(H_low, W_low), mode="bilinear", align_corners=False)
        c = self.curvature
        v_low = self.tau_low * self.W_low(low_level_feat)
        v_high = self.tau_high * self.W_high(high_level_up)
        x_low_poinc = self._expmap0(v_low, c)
        x_high_poinc = self._expmap0(v_high, c)

        g_low_val = torch.sigmoid(self.g_low(x_low_poinc))
        g_high_val = torch.sigmoid(self.g_high(x_high_poinc))

        v_low_tan = self._logmap0(x_low_poinc, c)
        v_high_tan = self._logmap0(x_high_poinc, c)

        v_low_content = self.P_c(v_low_tan)
        v_low_style = self.P_s(v_low_tan)
        v_high_content = self.P_c(v_high_tan)
        v_high_style = self.P_s(v_high_tan)

        x_low_content = self._expmap0(v_low_content, c)
        x_low_style = self._expmap0(v_low_style, c)
        x_high_content = self._expmap0(v_high_content, c)
        x_high_style = self._expmap0(v_high_style, c)

        xK_low = 2.0 * x_low_content / (1.0 + c * (x_low_content * x_low_content).sum(dim=1, keepdim=True)).clamp_min(self.eps)
        xK_high = 2.0 * x_high_content / (1.0 + c * (x_high_content * x_high_content).sum(dim=1, keepdim=True)).clamp_min(self.eps)
        gamma_K_low = 1.0 / torch.sqrt((1.0 - c * (xK_low * xK_low).sum(dim=1, keepdim=True)).clamp_min(self.eps))
        gamma_K_high = 1.0 / torch.sqrt((1.0 - c * (xK_high * xK_high).sum(dim=1, keepdim=True)).clamp_min(self.eps))

        w_low_K, w_high_K = g_low_val * gamma_K_low, g_high_val * gamma_K_high
        sum_gamma_xK = w_low_K * xK_low + w_high_K * xK_high
        sum_gamma = w_low_K + w_high_K

        m_Klein_content = sum_gamma_xK / sum_gamma.clamp_min(self.eps)
        m_H_poinc = m_Klein_content / (1.0 + torch.sqrt((1.0 - c * (m_Klein_content * m_Klein_content).sum(dim=1, keepdim=True)).clamp_min(self.eps)))

        w_half = torch.tensor(0.5, device=x_low_style.device, dtype=x_low_style.dtype)
        x_style = self._einstein_midpoint(x_low_style, x_high_style, w_half, w_half, c)

        # 3. Genuine Chart-Transition Coupling via Einstein Midpoint
        x_freq_poinc = self._expmap0(u_freq, c)
        w1 = torch.sigmoid(self.einstein_weight)
        w2 = 1.0 - w1

        m_unified_poinc = self._einstein_midpoint(m_H_poinc, x_freq_poinc, w1, w2, c)

        u_unified = self._logmap0(m_unified_poinc, c)

        # 4. Dual-Path Euclidean Residual & Output
        u_euc = self.W_euc(v_low + v_high)
        F_hyp = self.gn_hyp(self.W_o_hyp(u_unified))
        F_euc = self.gn_euc(self.W_o_euc(u_euc))
        alpha = torch.sigmoid(torch.clamp(self.phi, min=-3.0, max=3.0))
        F_out = self.act(self.final_norm(alpha * F_hyp + (1.0 - alpha) * F_euc))
        F_out = F_out + self.alpha_skip * self.skip_proj(low_level_feat)

        C_conf = (1.0 - c * (x_low_poinc * x_low_poinc).sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        G_bands_out = torch.stack(G_bands_list, dim=2) if G_bands_list else None
        aux_losses = {
            "curvature_reg": self.curvature_reg_loss(),
            "disentangle_norm": self._disentangle_norm_loss(x_low_content, x_low_style, x_high_content, x_high_style),
            "bayesian_prior": (self.coarse_prior_logits ** 2).sum() * 0.1
        }

        return F_out, C_conf, G_bands_out, x_style, aux_losses, F_hyp


class LinearRFFCrossAttention(nn.Module):
    def __init__(self, query_channels, context_channels, output_channels, num_features=64, dropout=0.0, fold_id=0, pad_factor=16):
        super().__init__()
        assert query_channels == output_channels
        inter_channels = max(context_channels // 2, query_channels * 2, 32)
        self.query_conv = nn.Conv2d(query_channels, inter_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(context_channels, inter_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(context_channels, query_channels, kernel_size=1, bias=False)

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(fold_id)
        self.register_buffer("W", torch.randn(num_features, inter_channels))
        torch.random.set_rng_state(rng_state)

        self.num_features = num_features
        self.pad_factor = pad_factor
        self.scale = inter_channels**-0.25
        self.attn_drop = nn.Dropout(dropout)
        self.local_bias = nn.Conv2d(query_channels, query_channels, kernel_size=3, padding=1, groups=query_channels, bias=False)
        self.proj_conv = nn.Conv2d(query_channels, output_channels, kernel_size=1, bias=False)
        self.proj_norm = nn.GroupNorm(1, output_channels)
        self.proj_drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(1e-2))
        self._init_weights()

    def _init_weights(self):
        for conv in [self.query_conv, self.key_conv, self.value_conv, self.proj_conv, self.local_bias]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="linear")

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        log_phi = x @ self.W.float().t() - (x**2).sum(dim=-1, keepdim=True) / 2.0
        return torch.exp(torch.clamp(log_phi, max=0.0)).to(orig_dtype)

    def forward(self, query_feat, context_feat) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C_q, H, W = query_feat.shape
        H_orig, W_orig = H, W

        pad_h = (self.pad_factor - H % self.pad_factor) % self.pad_factor
        pad_w = (self.pad_factor - W % self.pad_factor) % self.pad_factor
        query_feat_padded = F.pad(query_feat, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else query_feat
        H_p, W_p = query_feat_padded.shape[-2:]

        q = self.query_conv(query_feat_padded).flatten(2).transpose(1, 2)
        k = self.key_conv(context_feat).flatten(2).transpose(1, 2)
        v = self.value_conv(context_feat).flatten(2).transpose(1, 2)

        q, k = q * self.scale, k * self.scale
        phi_q, phi_k = self._phi(q), self._phi(k)

        mean_phi_q = phi_q.mean(dim=1, keepdim=True)  # (B, 1, D)
        grounding = torch.bmm(mean_phi_q, phi_k.transpose(1, 2))  # (B, 1, N_k)
        grounding = grounding / (grounding.sum(dim=-1, keepdim=True) + 1e-6)
        H_ctx, W_ctx = context_feat.shape[-2], context_feat.shape[-1]
        grounding_attn = grounding.view(B, 1, H_ctx, W_ctx)

        # Linear Attention Forward
        k_context = torch.bmm(phi_k.transpose(1, 2), v)
        k_norm = phi_k.sum(dim=1, keepdim=True).transpose(1, 2)
        out = torch.bmm(phi_q, k_context) / torch.bmm(phi_q, k_norm).clamp_min(1e-6)
        out = self.attn_drop(out).transpose(1, 2).reshape(B, C_q, H_p, W_p)
        out = self.local_bias(out)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H_orig, :W_orig]

        enhancement = self.proj_drop(self.proj_norm(self.proj_conv(out)))
        return query_feat + self.gamma * enhancement, grounding_attn


class SpatialCrossAttention(nn.Module):
    def __init__(
        self,
        query_channels: int,
        context_channels: int,
        output_channels: int,
        patch_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert query_channels == output_channels, (
            f"query_channels ({query_channels}) must equal output_channels ({output_channels}) "
            f"for residual addition in SpatialCrossAttention."
        )

        self.patch_size = patch_size
        inter_channels = max(context_channels // 2, query_channels * 2, 32)

        # 1. Patch-Token Query Encoder
        self.query_patch_embed = nn.Conv2d(
            query_channels,
            inter_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # 2. Context K/V Projections (at native resolution)
        self.key_conv = nn.Conv2d(
            context_channels, inter_channels, kernel_size=1, bias=False
        )
        self.value_conv = nn.Conv2d(
            context_channels, query_channels, kernel_size=1, bias=False
        )

        self.scale = inter_channels**-0.5
        self.attn_drop = nn.Dropout(dropout)

        # 3. Pixel-Shuffle Reconstruction
        self.reconstruct_proj = nn.Conv2d(
            query_channels,
            query_channels * (patch_size**2),
            kernel_size=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=patch_size)

        # 4. Refinement
        self.proj_conv = nn.Conv2d(
            query_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.proj_norm = nn.GroupNorm(1, output_channels)
        self.proj_drop = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.tensor(1e-2))
        self._init_weights()

    def _init_weights(self):
        for conv in [
            self.query_patch_embed,
            self.key_conv,
            self.value_conv,
            self.reconstruct_proj,
            self.proj_conv,
        ]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="linear")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(
        self, query_feat: torch.Tensor, context_feat: torch.Tensor
    ) -> torch.Tensor:
        B, C_q, H, W = query_feat.shape

        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Query spatial dimensions ({H}x{W}) must be divisible by "
            f"patch_size ({self.patch_size})."
        )
        q_h, q_w = H // self.patch_size, W // self.patch_size

        # (1) Patch-Token Query Encoding
        q = self.query_patch_embed(query_feat)
        q = q.flatten(2).transpose(1, 2)  # (B, N_q, inter_channels)

        # (2) Unpooled Context K/V
        k = self.key_conv(context_feat).flatten(2)  # (B, inter_channels, N_k)
        v = self.value_conv(context_feat).flatten(2)  # (B, C_q, N_k)

        # (3) Exact Attention
        attn = torch.bmm(q, k) * self.scale  # (B, N_q, N_k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (4) Aggregate values and reshape back to spatial token grid
        out = torch.bmm(v, attn.transpose(1, 2))  # (B, C_q, N_q)
        out = out.view(B, C_q, q_h, q_w)  # (B, C_q, q_h, q_w)

        # (5) Pixel-Shuffle Reconstruction
        recon = self.reconstruct_proj(out)  # (B, C_q * patch^2, q_h, q_w)
        recon = self.pixel_shuffle(recon)  # (B, C_q, H, W)

        # (6) Zero-init residual refinement
        enhancement = self.proj_conv(recon)
        enhancement = self.proj_norm(enhancement)
        enhancement = self.proj_drop(enhancement)

        fused = query_feat + self.gamma * enhancement
        return fused


class GeoSpecClassifier(nn.Module):
    """Single-branch unified pipeline with explicit Bilateral Equivariance and Möbius Regularization."""
    SIDE_VIT_INPUT_CHANNELS: int = 3

    def __init__(self, side_vit: nn.Module, cfg: Any):
        super().__init__()
        raw_trainable = getattr(cfg.network, "backbone_trainable_layers", []) or []
        for i in raw_trainable: assert 0 <= int(i) <= 4
        backbone_trainable_layers = [int(i) for i in raw_trainable]

        raw_branch_cfg = getattr(cfg.network, "vit_feature_branch", getattr(cfg.network, "vit1_feature_branch", [0, 2]))
        self.feature_branch = sorted([int(i) for i in raw_branch_cfg])
        for idx in self.feature_branch: assert 0 <= idx <= 3
        assert len(self.feature_branch) == 2 and self.feature_branch[0] != self.feature_branch[1]

        self.cfg = cfg
        self.num_classes = cfg.dataset.num_classes
        image_channels = cfg.dataset.image_channel_num
        side_input_size = cfg.network.side_input_size
        assert side_input_size % 16 == 0

        self._image_channels = image_channels
        self._side_vit_ch = self.SIDE_VIT_INPUT_CHANNELS
        self._side_input_size = side_input_size
        self.fold_id = getattr(cfg.network, "fold_id", 0)
        input_size = getattr(cfg.dataset, "image_size", getattr(cfg.network, "input_size", 384))

        self.cnn_backbone = MultiScaleConvNeXtV2Backbone(
            model_name="timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384",
            pretrained=True, in_chans=image_channels, backbone_trainable_layers=backbone_trainable_layers
        )
        feat_dims = self.cnn_backbone.channels
        proj_channels, side_vit_ch = 64, self.SIDE_VIT_INPUT_CHANNELS
        f_low, f_high = self.feature_branch

        self.fusion_module = UnifiedManifoldFusion(
            low_level_channels=feat_dims[f_low], high_level_channels=feat_dims[f_high],
            output_channels=proj_channels, tucker_rank=16, input_size=input_size,
            low_stage_idx=f_low, high_stage_idx=f_high
        )
        self.cross_attn = LinearRFFCrossAttention(
            query_channels=side_vit_ch, context_channels=proj_channels, output_channels=side_vit_ch,
            num_features=64, fold_id=self.fold_id, pad_factor=16
        )
        self.refine = self._make_refinement(side_vit_ch)
        self.drop_path = DropPath(getattr(cfg.network, "drop_path_rate", 0.0))
        self.sv_input_norm = AffineInstanceNorm(side_vit_ch)

        if self._image_channels != self._side_vit_ch and not (self._image_channels == 1 and self._side_vit_ch == 3):
            self._context_proj = nn.Conv2d(self._image_channels, self._side_vit_ch, kernel_size=1, bias=False)

        self.side_vit = side_vit
        side_vit_output_hidden_size = self.side_vit.side_encoder.hidden_size
        self.classifier_head = ClassifierHead(side_vit_output_hidden_size, self.num_classes)

        self.domain_classifier = DomainClassifier(proj_channels, getattr(cfg.dataset, "num_domains", 4))
        self.register_buffer("class_prototypes", torch.zeros(self.num_classes, side_vit_output_hidden_size))
        self.prototype_momentum = getattr(cfg.network, "prototype_momentum", 0.99)
        self.tta_lr = getattr(cfg.network, "tta_lr", 1e-3)
        self.tta_lambda_conf = getattr(cfg.network, "tta_lambda_conf", 1.0)
        self._last_aux_losses = {}

        self._tta_init_params = None

    @staticmethod
    def _make_refinement(channels: int) -> nn.Module:
        refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(1, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(1, channels)
        )
        nn.init.kaiming_normal_(refine[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(refine[3].weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(refine[-1].weight); nn.init.zeros_(refine[-1].bias)
        return refine

    def _expand_context(self, x):
        if self._image_channels == self._side_vit_ch: return x
        if self._image_channels == 1 and self._side_vit_ch == 3: return x.repeat(1, 3, 1, 1)
        return self._context_proj(x)

    def _prepare_context(self, x):
        context = self._expand_context(x)
        return F.interpolate(context, size=(self._side_input_size, self._side_input_size), mode="bilinear", align_corners=False)

    def _compute_hyperbolic_distance(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """Möbius addition distance for geometric regularization."""
        c = self.fusion_module.curvature
        eps = self.fusion_module.eps
        z_poinc = self.fusion_module._expmap0(cls_embedding.unsqueeze(-1).unsqueeze(-1), c).squeeze(-1).squeeze(-1)
        proto_poinc = self.fusion_module._expmap0(self.class_prototypes.unsqueeze(-1).unsqueeze(-1), c).squeeze(-1).squeeze(-1)

        z_exp = z_poinc.unsqueeze(1)
        P_exp = proto_poinc.unsqueeze(0)
        neg_z = -z_exp

        dot_np = (neg_z * P_exp).sum(dim=-1, keepdim=True)
        norm_n_sq = (neg_z ** 2).sum(dim=-1, keepdim=True)
        norm_p_sq = (P_exp ** 2).sum(dim=-1, keepdim=True)

        numerator = ((1 + 2 * c * dot_np + c * norm_p_sq) * neg_z + (1 - c * norm_n_sq) * P_exp)
        denominator = (1 + 2 * c * dot_np + c**2 * norm_n_sq * norm_p_sq).clamp_min(eps)
        mobius_norm = (numerator / denominator).norm(dim=-1)

        return (2.0 / torch.sqrt(c)) * torch.atanh(torch.clamp(torch.sqrt(c) * mobius_norm, max=1.0 - eps))

    def _pipeline(self, features, context, key_states, value_states, fusion_out=None, wavelet_mask=None):
        if fusion_out is None:
            F_low, F_high = features[self.feature_branch[0]], features[self.feature_branch[1]]
            proc_feat, C_conf, G_bands, x_style, aux_losses, F_hyp = self.fusion_module(F_low, F_high, wavelet_mask)
        else:
            proc_feat, C_conf, G_bands, x_style, aux_losses, F_hyp = fusion_out

        self._last_aux_losses = aux_losses

        vit_input, grounding_attn = self.cross_attn(context, proc_feat)
        vit_input = vit_input + self.drop_path(self.refine(vit_input))
        C_conf_resized = F.interpolate(C_conf, size=vit_input.shape[-2:], mode="bilinear", align_corners=False)
        vit_input = self.sv_input_norm(vit_input, C_conf_resized)

        cls_embedding = self.side_vit(vit_input, key_states, value_states)
        logits = self.classifier_head(cls_embedding)

        c = self.fusion_module.curvature
        x_style_tan = self.fusion_module._logmap0(x_style, c)
        x_style_gap = F.adaptive_avg_pool2d(x_style_tan, 1).flatten(1)
        domain_logits = self.domain_classifier(x_style_gap)

        return {
            "logits": logits, "domain_logits": domain_logits, "aux_losses": aux_losses,
            "G_bands": G_bands, "C_conf": C_conf, "cls_embedding": cls_embedding,
            "x_style": x_style, "grounding_attn": grounding_attn,
            "hyp_dist_matrix": self._compute_hyperbolic_distance(cls_embedding),
            "F_hyp": F_hyp
        }

    def get_auxiliary_losses(self):
        return self._last_aux_losses if self._last_aux_losses else {}

    def forward(self, x, key_states, value_states, use_causal_mask=False, compute_equivariance=True):
        features = self.cnn_backbone(x)
        context = self._prepare_context(x)

        F_low, F_high = features[self.feature_branch[0]], features[self.feature_branch[1]]
        fusion_out_nomask = self.fusion_module(F_low, F_high, None)
        proc_feat_nomask = fusion_out_nomask[0]
        F_hyp_nomask = fusion_out_nomask[5]

        # Explicit Bilateral Equivariance Loss
        if compute_equivariance:
            x_flip = torch.flip(x, dims=[-1])
            features_flip = self.cnn_backbone(x_flip)
            F_low_flip, F_high_flip = features_flip[self.feature_branch[0]], features_flip[self.feature_branch[1]]
            _, _, _, _, _, F_hyp_flip = self.fusion_module(F_low_flip, F_high_flip, None)
            sym_loss = F.l1_loss(F_hyp_nomask, torch.flip(F_hyp_flip, dims=[-1]))
        else:
            sym_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if use_causal_mask and self.training:
            B, L = x.shape[0], self.fusion_module.L
            wavelet_mask = torch.randint(0, 2, (B, 3, L), device=x.device).float()

            out_nomask = self._pipeline(features, context, key_states, value_states, fusion_out=fusion_out_nomask, wavelet_mask=None)
            nomask_aux_losses = self._last_aux_losses.copy()

            out_masked = self._pipeline(features, context, key_states, value_states, fusion_out=None, wavelet_mask=wavelet_mask)
            self._last_aux_losses = nomask_aux_losses

            result = dict(out_nomask)
            result["masked_logits"] = out_masked["logits"]
            result["sym_loss"] = sym_loss
            return result
        else:
            result = self._pipeline(features, context, key_states, value_states, fusion_out=fusion_out_nomask, wavelet_mask=None)
            result["sym_loss"] = sym_loss
            return result

    def reset_tta_state(self):
        """Initializes or resets the persistent TTA parameters for anti-forgetting regularization."""
        self._tta_init_params = {n: p.data.clone() for n, p in self.sv_input_norm.named_parameters()}
        self._tta_optimizer = None

    def tta_step(self, x, key_states, value_states):
        """Tent/EATA-style TTA with reliable sample selection and anti-forgetting."""
        if self._tta_init_params is None:
            self.reset_tta_state()

        req_grad_state = {n: p.requires_grad for n, p in self.named_parameters()}
        for n, p in self.named_parameters(): p.requires_grad = False

        tta_params = []
        for n, p in self.sv_input_norm.named_parameters():
            p.requires_grad = True
            tta_params.append(p)

        if self._tta_optimizer is None:
            self._tta_optimizer = torch.optim.SGD(tta_params, lr=self.tta_lr, momentum=0.9)
        else:
            self._tta_optimizer.param_groups[0]['params'] = tta_params

        was_training = self.training
        self.eval()

        outputs = self.forward(x, key_states, value_states, use_causal_mask=False, compute_equivariance=False)
        logits, C_conf = outputs["logits"], outputs["C_conf"]
        p = F.softmax(logits, dim=-1)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)

        e0 = 0.4 * math.log(self.num_classes)
        entropy_reliable = entropy < e0

        conf_per_sample = C_conf.mean(dim=[1, 2, 3])
        conf_reliable = conf_per_sample > 0.5
        reliable_mask = entropy_reliable & conf_reliable

        if reliable_mask.any():
            conf_weight = self.tta_lambda_conf * conf_per_sample[reliable_mask] + (1 - self.tta_lambda_conf)
            ent_loss = (entropy[reliable_mask] * conf_weight).mean()

            reg_loss = sum(((param - self._tta_init_params[n]) ** 2).sum() for n, param in self.sv_input_norm.named_parameters())

            loss = ent_loss + 0.1 * reg_loss
            self._tta_optimizer.zero_grad()
            loss.backward()
            self._tta_optimizer.step()

        for n, p in self.named_parameters(): p.requires_grad = req_grad_state[n]
        self.zero_grad()
        if was_training: self.train()

    def update_prototypes(self, cls_embeddings, labels, predictions, momentum=None):
        if momentum is None: momentum = self.prototype_momentum
        correct = predictions.eq(labels)
        for cls in range(self.num_classes):
            mask = correct & (labels == cls)
            if mask.any():
                batch_proto = cls_embeddings[mask].mean(dim=0)
                self.class_prototypes.data[cls] = momentum * self.class_prototypes.data[cls] + (1.0 - momentum) * batch_proto

    def get_wbs(self, G_bands):
        if G_bands is None: return torch.zeros(1, 3)
        return G_bands.abs().sum(dim=(2, 3))

    def compute_hsm(self, x, target_class, key_states, value_states):
        x = x.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            outputs = self.forward(x, key_states, value_states, use_causal_mask=False, compute_equivariance=False)
            cls_embedding = outputs["cls_embedding"]
            c = self.fusion_module.curvature
            z_poinc = self.fusion_module._expmap0(cls_embedding.unsqueeze(-1).unsqueeze(-1), c).squeeze(-1).squeeze(-1)
            proto = self.class_prototypes[target_class].unsqueeze(0).expand_as(z_poinc)
            proto_poinc = self.fusion_module._expmap0(proto.unsqueeze(-1).unsqueeze(-1), c).squeeze(-1).squeeze(-1)

            neg_z = -z_poinc
            dot_np = (neg_z * proto_poinc).sum(dim=-1, keepdim=True)
            norm_n_sq = (neg_z ** 2).sum(dim=-1, keepdim=True)
            norm_p_sq = (proto_poinc ** 2).sum(dim=-1, keepdim=True)

            numerator = ((1 + 2 * c * dot_np + c * norm_p_sq) * neg_z + (1 - c * norm_n_sq) * proto_poinc)
            denominator = (1 + 2 * c * dot_np + c**2 * norm_n_sq * norm_p_sq).clamp_min(self.fusion_module.eps)
            mobius_norm = (numerator / denominator).norm(dim=-1, keepdim=True)

            dist = (2.0 / torch.sqrt(c)) * torch.atanh(
                torch.clamp(torch.sqrt(c) * mobius_norm, max=1.0 - self.fusion_module.eps)
            ).sum()

            grad = torch.autograd.grad(dist, x, create_graph=False)[0]
        return grad.norm(dim=1)
