import math
from typing import Any, Dict, List, Optional, Tuple
from geospec_train.config import CoreConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from geospec_model.medmamba import medmamba_t


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(
        self, x: torch.Tensor, conf: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del conf

        drop_prob = float(self.drop_prob)

        if not self.training or not math.isfinite(drop_prob) or drop_prob <= 0.0:
            return x

        if drop_prob >= 1.0:
            return torch.zeros_like(x)

        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        mask = (torch.rand(shape, device=x.device, dtype=x.dtype) < keep_prob).to(
            x.dtype
        )

        return x * (mask / keep_prob)


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
        self.weight = nn.Parameter(torch.ones(num_channels))  # gamma
        self.beta = nn.Parameter(torch.zeros(num_channels))  # beta (confidence)
        self.bias = nn.Parameter(torch.zeros(num_channels))  # delta

    def forward(
        self, x: torch.Tensor, conf: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.in_norm(x)
        dtype = x.dtype

        gamma = self.weight.to(dtype=dtype).view(1, -1, 1, 1)
        beta = self.beta.to(dtype=dtype).view(1, -1, 1, 1)
        delta = self.bias.to(dtype=dtype).view(1, -1, 1, 1)

        if conf is not None:
            if conf.dim() == 3:
                conf = conf.unsqueeze(1)

            if conf.shape[-2:] != x.shape[-2:]:
                conf = F.interpolate(
                    conf.float(),
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if conf.shape[1] != 1 and conf.shape[1] != x.shape[1]:
                conf = conf.mean(dim=1, keepdim=True)

            conf = conf.to(dtype=dtype)
            return x * (gamma + beta * conf) + delta

        return x * gamma + delta


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
        x = x.float()
        x = GradReverseFunction.apply(x, float(self.grl_lambda))

        fc1_weight = self.fc1.weight.float()
        fc1_bias = self.fc1.bias.float() if self.fc1.bias is not None else None

        x = F.linear(x, fc1_weight, fc1_bias)
        x = self.relu(x)
        x = self.drop(x)

        fc2_weight = self.fc2.weight.float()
        fc2_bias = self.fc2.bias.float() if self.fc2.bias is not None else None

        x = F.linear(x, fc2_weight, fc2_bias)

        return x


class MultiScaleMedMambaTBackbone(nn.Module):
    """Multi-Scale MedMamba-T Backbone with absolute freezing configuration."""

    def __init__(self, in_chans: int = 3, pretrained: bool = True):
        super().__init__()

        # Instantiate MedMamba-T with multi-scale feature output extraction enabled
        self.backbone = medmamba_t(
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Globally freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.channels = self.backbone.feature_info.channels()

    def train(self, mode: bool = True):
        # Always maintain the frozen backbone in evaluation mode
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


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
            raise ValueError(
                f"Invalid spatial sizes: low_size={low_size}, high_size={high_size}."
            )

        if low_size < high_size:
            raise ValueError(
                f"low_size ({low_size}) must be >= high_size ({high_size})."
            )

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
            self.coarse_prior_logits.data[self.L - 1, 2] = -1.0

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

        self.register_buffer("w_half", torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        for conv in [
            self.high_align,
            self.U_high,
            self.U_LL,
            self.band_proj_lh,
            self.band_proj_hl,
            self.band_proj_hh,
            self.spectral_conv,
            self.W_low,
            self.W_high,
            self.W_euc,
            self.W_o_hyp,
            self.W_o_euc,
            self.skip_proj,
        ]:
            nn.init.xavier_uniform_(conv.weight)

        nn.init.normal_(self.P_s.weight, std=1e-3)

        for m in [*self.mlp_spatial.modules(), *self.mlp_bands.modules()]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.g_low.weight)
        nn.init.zeros_(self.g_low.bias)
        nn.init.zeros_(self.g_high.weight)
        nn.init.zeros_(self.g_high.bias)
        with torch.no_grad():
            eye = torch.eye(
                self.output_channels,
                device=self.P_c.weight.device,
                dtype=self.P_c.weight.dtype,
            )
            self.P_c.weight.data.copy_(
                eye.view(self.output_channels, self.output_channels, 1, 1)
            )
            self.psi.weight.data.copy_(
                eye.view(self.output_channels, self.output_channels, 1, 1)
            )
        for gn in [self.gn_hyp, self.gn_euc, self.final_norm]:
            nn.init.ones_(gn.weight)
            nn.init.zeros_(gn.bias)

    @staticmethod
    def _haar_dwt_2d(x: torch.Tensor):
        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]
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
        return out.permute(0, 1, 2, 4, 3, 5).reshape(
            ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2
        )

    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.theta_c.float()) + float(self.eps)

    def _safe_feature(
        self, x: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(-1.0e4, 1.0e4)

        if dtype is not None:
            x = x.to(dtype=dtype)

        return x

    def _safe_poincare(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        eps = float(self.eps)

        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        x = x.clamp(-1.0e4, 1.0e4)

        c = torch.as_tensor(c, dtype=torch.float32, device=x.device)
        c = torch.where(torch.isfinite(c), c, torch.ones_like(c))
        c = c.clamp_min(eps)

        sqrt_c = torch.sqrt(c)
        norm = torch.sqrt((x * x).sum(dim=1, keepdim=True) + eps)

        max_norm = (1.0 - eps) / sqrt_c.clamp_min(eps)
        scale = torch.where(
            norm > max_norm,
            max_norm / (norm + eps),
            torch.ones_like(norm),
        )

        x = x * scale
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(-1.0e4, 1.0e4)

        return x

    def _validate_hyperbolic_cache(
        self,
        hyperbolic_cache,
        c: torch.Tensor,
        B: int,
        H_low: int,
        W_low: int,
    ):
        if (
            not isinstance(hyperbolic_cache, (tuple, list))
            or len(hyperbolic_cache) != 7
        ):
            return None

        (
            v_low,
            v_high,
            m_H_poinc,
            x_style,
            C_conf,
            disentangle_loss,
            cached_c,
        ) = hyperbolic_cache

        try:
            cached_c = torch.as_tensor(cached_c, device=c.device, dtype=torch.float32)
        except Exception:
            return None

        if cached_c.numel() != 1:
            return None

        if not torch.isfinite(cached_c).all().item():
            return None

        if not torch.allclose(cached_c, c.float(), rtol=1e-3, atol=1e-5):
            return None

        expected_shapes = (
            (B, self.output_channels, H_low, W_low),
            (B, self.output_channels, H_low, W_low),
            (B, self.output_channels, H_low, W_low),
            (B, self.output_channels, H_low, W_low),
            (B, 1, H_low, W_low),
        )

        cached_tensors = (v_low, v_high, m_H_poinc, x_style, C_conf)

        for tensor, expected_shape in zip(cached_tensors, expected_shapes):
            if not torch.is_tensor(tensor):
                return None

            if tuple(tensor.shape) != expected_shape:
                return None

            if not torch.isfinite(tensor).all().item():
                return None

        v_low = self._safe_tangent(v_low.detach().to(device=c.device), c)
        v_high = self._safe_tangent(v_high.detach().to(device=c.device), c)

        m_H_poinc = self._safe_poincare(m_H_poinc.detach().to(device=c.device), c)
        x_style = self._safe_poincare(x_style.detach().to(device=c.device), c)

        C_conf = C_conf.detach().to(device=c.device).float()
        C_conf = torch.nan_to_num(C_conf, nan=0.0, posinf=1.0, neginf=0.0)
        C_conf = C_conf.clamp(0.0, 1.0)

        if torch.is_tensor(disentangle_loss):
            disentangle_loss = disentangle_loss.detach().to(device=c.device).float()
            if not torch.isfinite(disentangle_loss).all().item():
                disentangle_loss = torch.zeros((), device=c.device, dtype=torch.float32)
        else:
            disentangle_loss = torch.zeros((), device=c.device, dtype=torch.float32)

        return (
            v_low,
            v_high,
            m_H_poinc,
            x_style,
            C_conf,
            disentangle_loss,
            cached_c.detach(),
        )

    def _safe_tangent(
        self, v: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        eps = float(self.eps)

        v = v.float()
        v = torch.nan_to_num(v, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        v = v.clamp(-1.0e4, 1.0e4)

        if c is not None:
            c = torch.as_tensor(c, dtype=torch.float32, device=v.device)
            c = torch.where(torch.isfinite(c), c, torch.ones_like(c))
            c = c.clamp_min(eps)

            sqrt_c = torch.sqrt(c)
            v_norm = torch.sqrt((v * v).sum(dim=1, keepdim=True) + eps)

            max_norm = 15.0 / sqrt_c.clamp_min(eps)
            scale = torch.where(
                v_norm > max_norm,
                max_norm / (v_norm + eps),
                torch.ones_like(v_norm),
            )

            v = v * scale
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            v = v.clamp(-1.0e4, 1.0e4)

        return v

    def curvature_reg_loss(self) -> torch.Tensor:
        raw = F.softplus(self.theta_c.float()) + float(self.eps)
        fallback = torch.full_like(raw, float(self.c_max) + 1.0)
        raw = torch.where(torch.isfinite(raw), raw, fallback)
        raw = raw.clamp(0.0, 1.0e4)

        return self.curv_reg_weight * (
            torch.relu(raw - float(self.c_max)).pow(2)
            + torch.relu(float(self.c_min) - raw).pow(2)
        )

    def _expmap0(self, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        eps = float(self.eps)

        c = torch.as_tensor(c, dtype=torch.float32, device=v.device)
        c = torch.where(torch.isfinite(c), c, torch.ones_like(c))
        c = c.clamp_min(eps)

        v = self._safe_tangent(v, c)

        sqrt_c = torch.sqrt(c)
        v_norm = torch.sqrt((v * v).sum(dim=1, keepdim=True) + eps)

        radial = torch.clamp(sqrt_c * v_norm, max=15.0)
        tanh_val = torch.tanh(radial)

        denom = (sqrt_c * v_norm).clamp_min(eps)
        out = tanh_val * v / denom

        return self._safe_poincare(out, c)

    def _logmap0(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        eps = float(self.eps)

        c = torch.as_tensor(c, dtype=torch.float32, device=x.device)
        c = torch.where(torch.isfinite(c), c, torch.ones_like(c))
        c = c.clamp_min(eps)

        x = self._safe_poincare(x, c)

        sqrt_c = torch.sqrt(c)
        x_norm = torch.sqrt((x * x).sum(dim=1, keepdim=True) + eps)

        scaled = torch.clamp(sqrt_c * x_norm, min=0.0, max=1.0 - eps)
        artanh_val = torch.atanh(scaled)

        denom = (sqrt_c * x_norm).clamp_min(eps)
        out = artanh_val * x / denom

        return self._safe_tangent(out, c)

    def _disentangle_norm_loss(self, x_lc, x_ls, x_hc, x_hs, c):
        c = torch.as_tensor(c, dtype=torch.float32, device=x_lc.device)
        sqrt_c = torch.sqrt(c).clamp_min(float(self.eps))
        eps = float(self.eps)

        def radius(x: torch.Tensor) -> torch.Tensor:
            x = x.float()
            x_norm = torch.sqrt((x * x).sum(dim=1, keepdim=True) + eps)
            scaled = torch.clamp(sqrt_c * x_norm, min=0.0, max=1.0 - eps)
            return ((2.0 / sqrt_c) * torch.atanh(scaled)).mean()

        content_norm = (radius(x_lc) + radius(x_hc)) * 0.5
        style_norm = (radius(x_ls) + radius(x_hs)) * 0.5

        loss = (content_norm - self.rho_c.float()).pow(2) + (
            style_norm - self.rho_s.float()
        ).pow(2)

        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    def _einstein_midpoint(self, x1, x2, w1, w2, c):
        eps = float(self.eps)

        c = torch.as_tensor(c, dtype=torch.float32, device=x1.device)
        c = torch.where(torch.isfinite(c), c, torch.ones_like(c))
        c = c.clamp_min(eps)

        x1 = self._safe_poincare(x1, c)
        x2 = self._safe_poincare(x2, c)

        w1 = torch.as_tensor(w1, dtype=torch.float32, device=x1.device)
        w2 = torch.as_tensor(w2, dtype=torch.float32, device=x1.device)

        w1 = torch.nan_to_num(w1, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        w2 = torch.nan_to_num(w2, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        norm1 = (x1 * x1).sum(dim=1, keepdim=True)
        norm2 = (x2 * x2).sum(dim=1, keepdim=True)

        xK1 = 2.0 * x1 / (1.0 + c * norm1).clamp_min(eps)
        xK2 = 2.0 * x2 / (1.0 + c * norm2).clamp_min(eps)

        xK1 = torch.nan_to_num(xK1, nan=0.0, posinf=0.0, neginf=0.0)
        xK2 = torch.nan_to_num(xK2, nan=0.0, posinf=0.0, neginf=0.0)

        gamma1 = 1.0 / torch.sqrt(
            (1.0 - c * (xK1 * xK1).sum(dim=1, keepdim=True)).clamp_min(eps)
        )
        gamma2 = 1.0 / torch.sqrt(
            (1.0 - c * (xK2 * xK2).sum(dim=1, keepdim=True)).clamp_min(eps)
        )

        gamma1 = torch.nan_to_num(gamma1, nan=1.0, posinf=1.0, neginf=1.0)
        gamma2 = torch.nan_to_num(gamma2, nan=1.0, posinf=1.0, neginf=1.0)

        w1_act = w1 * gamma1
        w2_act = w2 * gamma2

        sum_gamma = (w1_act + w2_act).clamp_min(eps)
        sum_gamma_xK = w1_act * xK1 + w2_act * xK2

        m_Klein = sum_gamma_xK / sum_gamma
        m_Klein = torch.nan_to_num(m_Klein, nan=0.0, posinf=0.0, neginf=0.0)

        m_norm = (m_Klein * m_Klein).sum(dim=1, keepdim=True)
        m_P = m_Klein / (1.0 + torch.sqrt((1.0 - c * m_norm).clamp_min(eps)))

        return self._safe_poincare(m_P, c)

    def _projection_incoherence_loss(self) -> torch.Tensor:
        pc = self.P_c.weight.flatten(1).float()
        ps = self.P_s.weight.flatten(1).float()

        pc = F.normalize(pc, p=2, dim=1, eps=float(self.eps))
        ps = F.normalize(ps, p=2, dim=1, eps=float(self.eps))

        cross = pc @ ps.t()
        loss = cross.pow(2).mean()

        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self, low_level_feat, high_level_feat, wavelet_mask=None, hyperbolic_cache=None
    ):
        module_dtype = self.W_low.weight.dtype

        low_level_feat_in = low_level_feat.to(module_dtype)
        high_level_feat_in = high_level_feat.to(module_dtype)

        B, _, H_low, W_low = low_level_feat_in.shape

        stride = 2**self.L

        pad_h = (stride - H_low % stride) % stride
        pad_w = (stride - W_low % stride) % stride

        if pad_h > 0 or pad_w > 0:
            low_freq = F.pad(low_level_feat_in, (0, pad_w, 0, pad_h))
        else:
            low_freq = low_level_feat_in

        H_freq, W_freq = low_freq.shape[-2:]
        target_high_size = (H_freq // stride, W_freq // stride)

        actual_h, actual_w = high_level_feat_in.shape[-2:]

        if actual_h <= 0 or actual_w <= 0:
            raise ValueError("High-level feature map has invalid spatial size.")

        if (
            H_freq % actual_h != 0
            or W_freq % actual_w != 0
            or H_freq // actual_h != stride
            or W_freq // actual_w != stride
        ):
            raise ValueError(
                "Low/high feature spatial ratio is inconsistent with the configured DWT levels."
            )

        if wavelet_mask is not None:
            if (
                wavelet_mask.dim() != 3
                or wavelet_mask.shape[0] != B
                or wavelet_mask.shape[1] != 3
                or wavelet_mask.shape[2] != self.L
            ):
                raise ValueError("wavelet_mask must have shape (B, 3, L).")

        def _group_norm_dtype_safe(gn: nn.Module, x: torch.Tensor) -> torch.Tensor:
            weight = (
                gn.weight.float() if getattr(gn, "weight", None) is not None else None
            )
            bias = gn.bias.float() if getattr(gn, "bias", None) is not None else None
            y = F.group_norm(
                x.float(),
                gn.num_groups,
                weight=weight,
                bias=bias,
                eps=gn.eps,
            )
            return y.to(dtype=module_dtype)

        X_high_tilde = self.high_align(high_level_feat_in)

        if X_high_tilde.shape[-2:] != target_high_size:
            X_high_tilde = F.interpolate(
                X_high_tilde,
                size=target_high_size,
                mode="bilinear",
                align_corners=False,
            )

        X_high_tilde = self._safe_feature(X_high_tilde, module_dtype)

        LL = low_freq
        high_freq_bands = []

        for level_idx in range(self.L):
            LL, LH, HL, HH = self._haar_dwt_2d(LL)

            if wavelet_mask is not None:
                mask_l = wavelet_mask[:, :, level_idx].to(dtype=LH.dtype)

                LH = LH * mask_l[:, 0:1].view(B, 1, 1, 1)
                HL = HL * mask_l[:, 1:2].view(B, 1, 1, 1)
                HH = HH * mask_l[:, 2:3].view(B, 1, 1, 1)

            LH = self._safe_feature(LH, module_dtype)
            HL = self._safe_feature(HL, module_dtype)
            HH = self._safe_feature(HH, module_dtype)

            high_freq_bands.append((LH, HL, HH))

        LL_L = self._safe_feature(LL, module_dtype)

        g_spatial = torch.cat(
            [
                F.adaptive_avg_pool2d(X_high_tilde, 1).flatten(1),
                F.adaptive_avg_pool2d(LL_L, 1).flatten(1),
            ],
            dim=1,
        )

        G = self.mlp_spatial(g_spatial).view(B, self.r, self.r, self.output_channels)
        G = self._safe_feature(G, module_dtype)

        G_spectral = (
            self.spectral_conv(F.adaptive_avg_pool2d(LL_L, 1))
            .flatten(1)
            .view(B, self.r, self.r, self.output_channels)
        )
        G_spectral = self._safe_feature(G_spectral, module_dtype)

        G = G * G_spectral

        P_high = self.U_high(X_high_tilde)
        P_LL = self.U_LL(LL_L)

        P_high = self._safe_feature(P_high, module_dtype)
        P_LL = self._safe_feature(P_LL, module_dtype)

        LL_fused = torch.einsum("bijk,bihw,bjhw->bkhw", G, P_high, P_LL)
        LL_fused = self._safe_feature(LL_fused, module_dtype)

        theta = self.theta_steer.to(dtype=module_dtype)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        zero = torch.zeros_like(cos_t)
        one = torch.ones_like(cos_t)

        row_0 = torch.stack([cos_t, -sin_t, zero])
        row_1 = torch.stack([sin_t, cos_t, zero])
        row_2 = torch.stack([zero, zero, one])

        M_steer = torch.stack([row_0, row_1, row_2]).squeeze(-1)

        G_bands_list = []
        modulated_bands = []

        for l in range(self.L):
            LH_l, HL_l, HH_l = high_freq_bands[l]

            stacked = torch.cat([LH_l, HL_l, HH_l], dim=1)
            stacked = self._safe_feature(stacked, module_dtype)

            G_bands_raw = self.mlp_bands(
                F.adaptive_avg_pool2d(stacked.abs(), 1).flatten(1)
            ).view(B, 3, self.output_channels)

            G_hat = torch.einsum(
                "ij,bjk->bik", M_steer.to(dtype=G_bands_raw.dtype), G_bands_raw
            )

            G_hat = G_hat + self.coarse_prior_logits[l].to(dtype=G_hat.dtype).view(
                1, 3, 1
            )

            G_hat = torch.nan_to_num(G_hat.float(), nan=0.0, posinf=50.0, neginf=-50.0)
            G_hat = F.softmax(G_hat, dim=1).to(dtype=module_dtype)

            G_bands_list.append(G_hat)

            tilde_LH = self.band_proj_lh(LH_l) * G_hat[:, 0, :].unsqueeze(-1).unsqueeze(
                -1
            )
            tilde_HL = self.band_proj_hl(HL_l) * G_hat[:, 1, :].unsqueeze(-1).unsqueeze(
                -1
            )
            tilde_HH = self.band_proj_hh(HH_l) * G_hat[:, 2, :].unsqueeze(-1).unsqueeze(
                -1
            )

            tilde_LH = self._safe_feature(tilde_LH, module_dtype)
            tilde_HL = self._safe_feature(tilde_HL, module_dtype)
            tilde_HH = self._safe_feature(tilde_HH, module_dtype)

            modulated_bands.append((tilde_LH, tilde_HL, tilde_HH))

        current_LL = LL_fused

        for l in range(self.L - 1, -1, -1):
            current_LL = self._haar_idwt_2d(current_LL, *modulated_bands[l])

        if pad_h > 0 or pad_w > 0:
            current_LL = current_LL[:, :, :H_low, :W_low]

        current_LL = self._safe_feature(current_LL, module_dtype)

        u_freq = self.psi(current_LL)

        c = self.curvature

        validated_cache = self._validate_hyperbolic_cache(
            hyperbolic_cache, c, B, H_low, W_low
        )

        if validated_cache is None:
            eps = float(self.eps)

            high_level_up = F.interpolate(
                high_level_feat_in,
                size=(H_low, W_low),
                mode="bilinear",
                align_corners=False,
            )
            high_level_up = self._safe_feature(high_level_up, module_dtype)

            v_low = (self.tau_low * self.W_low(low_level_feat_in)).float()
            v_high = (self.tau_high * self.W_high(high_level_up)).float()

            v_low = self._safe_tangent(v_low, c)
            v_high = self._safe_tangent(v_high, c)

            x_low_poinc = self._expmap0(v_low, c)
            x_high_poinc = self._expmap0(v_high, c)

            g_low_val = torch.sigmoid(self.g_low(x_low_poinc.to(module_dtype))).float()
            g_high_val = torch.sigmoid(
                self.g_high(x_high_poinc.to(module_dtype))
            ).float()

            g_low_val = torch.nan_to_num(g_low_val, nan=0.5, posinf=1.0, neginf=0.0)
            g_high_val = torch.nan_to_num(g_high_val, nan=0.5, posinf=1.0, neginf=0.0)

            v_low_tan = self._logmap0(x_low_poinc, c)
            v_high_tan = self._logmap0(x_high_poinc, c)

            v_low_content = self.P_c(v_low_tan.to(module_dtype))
            v_low_style = self.P_s(v_low_tan.to(module_dtype))

            v_high_content = self.P_c(v_high_tan.to(module_dtype))
            v_high_style = self.P_s(v_high_tan.to(module_dtype))

            x_low_content = self._expmap0(v_low_content, c)
            x_low_style = self._expmap0(v_low_style, c)

            x_high_content = self._expmap0(v_high_content, c)
            x_high_style = self._expmap0(v_high_style, c)

            xK_low = (
                2.0
                * x_low_content
                / (
                    1.0 + c * (x_low_content * x_low_content).sum(dim=1, keepdim=True)
                ).clamp_min(eps)
            )
            xK_high = (
                2.0
                * x_high_content
                / (
                    1.0 + c * (x_high_content * x_high_content).sum(dim=1, keepdim=True)
                ).clamp_min(eps)
            )

            xK_low = torch.nan_to_num(xK_low, nan=0.0, posinf=0.0, neginf=0.0)
            xK_high = torch.nan_to_num(xK_high, nan=0.0, posinf=0.0, neginf=0.0)

            gamma_K_low = 1.0 / torch.sqrt(
                (1.0 - c * (xK_low * xK_low).sum(dim=1, keepdim=True)).clamp_min(eps)
            )
            gamma_K_high = 1.0 / torch.sqrt(
                (1.0 - c * (xK_high * xK_high).sum(dim=1, keepdim=True)).clamp_min(eps)
            )

            gamma_K_low = torch.nan_to_num(gamma_K_low, nan=1.0, posinf=1.0, neginf=1.0)
            gamma_K_high = torch.nan_to_num(
                gamma_K_high, nan=1.0, posinf=1.0, neginf=1.0
            )

            w_low_K = g_low_val.float() * gamma_K_low
            w_high_K = g_high_val.float() * gamma_K_high

            sum_gamma_xK = w_low_K * xK_low + w_high_K * xK_high
            sum_gamma = (w_low_K + w_high_K).clamp_min(eps)

            m_Klein_content = sum_gamma_xK / sum_gamma
            m_Klein_content = torch.nan_to_num(
                m_Klein_content, nan=0.0, posinf=0.0, neginf=0.0
            )

            m_H_poinc = m_Klein_content / (
                1.0
                + torch.sqrt(
                    (
                        1.0
                        - c
                        * (m_Klein_content * m_Klein_content).sum(dim=1, keepdim=True)
                    ).clamp_min(eps)
                )
            )

            m_H_poinc = self._safe_poincare(m_H_poinc, c)

            x_style = self._einstein_midpoint(
                x_low_style, x_high_style, self.w_half, self.w_half, c
            )

            C_conf = (
                1.0 - c * (x_low_poinc * x_low_poinc).sum(dim=1, keepdim=True)
            ).clamp(0.0, 1.0)

            C_conf = torch.nan_to_num(C_conf.float(), nan=0.0, posinf=1.0, neginf=0.0)
            C_conf = C_conf.clamp(0.0, 1.0)

            disentangle_loss = (
                self._disentangle_norm_loss(
                    x_low_content, x_low_style, x_high_content, x_high_style, c
                )
                + 0.1 * self._projection_incoherence_loss()
            )

            disentangle_loss = torch.nan_to_num(
                disentangle_loss.float(), nan=0.0, posinf=0.0, neginf=0.0
            )

            hyperbolic_cache = (
                v_low,
                v_high,
                m_H_poinc,
                x_style,
                C_conf,
                disentangle_loss,
                c.detach(),
            )

            prior_target = torch.zeros(
                self.coarse_prior_logits.shape,
                dtype=torch.float32,
                device=self.coarse_prior_logits.device,
            )

            if self.L > 0:
                prior_target[self.L - 1, 2] = -1.0

            bayesian_prior_loss = (
                (self.coarse_prior_logits.float() - prior_target) ** 2
            ).sum() * 0.1

            aux_losses = {
                "curvature_reg": torch.nan_to_num(
                    self.curvature_reg_loss(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                "disentangle_norm": disentangle_loss,
                "bayesian_prior": torch.nan_to_num(
                    bayesian_prior_loss,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
            }

        else:
            (
                v_low,
                v_high,
                m_H_poinc,
                x_style,
                C_conf,
                disentangle_loss,
                cached_c,
            ) = validated_cache

            zero_loss = c.float() * 0.0

            aux_losses = {
                "curvature_reg": zero_loss,
                "disentangle_norm": zero_loss,
                "bayesian_prior": zero_loss,
            }

            hyperbolic_cache = (
                v_low,
                v_high,
                m_H_poinc,
                x_style,
                C_conf,
                disentangle_loss,
                c.detach(),
            )

        x_freq_poinc = self._expmap0(u_freq, c)

        w1 = torch.sigmoid(self.einstein_weight.float())
        w2 = 1.0 - w1

        w1 = torch.nan_to_num(w1, nan=0.5, posinf=1.0, neginf=0.0)
        w2 = torch.nan_to_num(w2, nan=0.5, posinf=1.0, neginf=0.0)

        m_unified_poinc = self._einstein_midpoint(
            m_H_poinc.float(), x_freq_poinc, w1, w2, c
        )

        u_unified = self._logmap0(m_unified_poinc, c)

        u_euc = self.W_euc((v_low + v_high).to(module_dtype))

        F_hyp = _group_norm_dtype_safe(
            self.gn_hyp,
            self.W_o_hyp(u_unified.to(module_dtype)),
        )
        F_euc = _group_norm_dtype_safe(
            self.gn_euc,
            self.W_o_euc(u_euc),
        )

        F_hyp = self._safe_feature(F_hyp, None)
        F_euc = self._safe_feature(F_euc, None)

        alpha = torch.sigmoid(torch.clamp(self.phi.float(), min=-3.0, max=3.0))
        alpha = torch.nan_to_num(alpha, nan=0.5, posinf=1.0, neginf=0.0)

        mixed = alpha * F_hyp.float() + (1.0 - alpha) * F_euc.float()
        mixed = self._safe_feature(mixed, module_dtype)

        F_out = self.act(_group_norm_dtype_safe(self.final_norm, mixed))

        skip = self._safe_feature(self.skip_proj(low_level_feat_in), None)

        alpha_skip = torch.clamp(self.alpha_skip.float(), -10.0, 10.0)

        F_out = self._safe_feature(F_out, None) + alpha_skip * skip
        F_out = self._safe_feature(F_out, None)

        G_bands_out = torch.stack(G_bands_list, dim=2).float() if G_bands_list else None

        C_conf = torch.nan_to_num(C_conf.float(), nan=0.0, posinf=1.0, neginf=0.0)
        C_conf = C_conf.clamp(0.0, 1.0)

        x_style = self._safe_poincare(x_style, c)
        F_hyp = self._safe_feature(F_hyp, None)

        if G_bands_out is not None:
            G_bands_out = torch.nan_to_num(G_bands_out, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            F_out,
            C_conf,
            G_bands_out,
            x_style,
            aux_losses,
            F_hyp,
            hyperbolic_cache,
        )


class LinearRFFCrossAttention(nn.Module):
    def __init__(
        self,
        query_channels: int,
        context_channels: int,
        output_channels: int,
        num_features: int = 64,
        dropout: float = 0.0,
        fold_id: int = 0,
        pad_factor: int = 16,
    ):
        super().__init__()

        assert query_channels == output_channels
        assert num_features > 0
        assert pad_factor > 0

        inter_channels = max(context_channels // 2, query_channels * 2, 32)

        self.query_conv = nn.Conv2d(
            query_channels, inter_channels, kernel_size=1, bias=False
        )
        self.key_conv = nn.Conv2d(
            context_channels, inter_channels, kernel_size=1, bias=False
        )
        self.value_conv = nn.Conv2d(
            context_channels, query_channels, kernel_size=1, bias=False
        )

        seed = int(fold_id) % (232)
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.register_buffer(
            "W", torch.randn(num_features, inter_channels, generator=generator)
        )

        self.pad_factor = pad_factor

        self.scale = inter_channels ** (-0.25)

        self.attn_drop = nn.Dropout(dropout)

        self.local_bias = nn.Conv2d(
            query_channels,
            query_channels,
            kernel_size=3,
            padding=1,
            groups=query_channels,
            bias=False,
        )

        self.proj_conv = nn.Conv2d(
            query_channels, output_channels, kernel_size=1, bias=False
        )
        self.proj_norm = nn.GroupNorm(1, output_channels)
        self.proj_drop = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.tensor(1e-2))

        self._init_weights()

    def _init_weights(self):
        for conv in [
            self.query_conv,
            self.key_conv,
            self.value_conv,
            self.proj_conv,
            self.local_bias,
        ]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="linear")

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        W = self.W.float()

        log_phi = x @ W.t() - 0.5 * (x * x).sum(dim=-1, keepdim=True)
        log_phi = torch.nan_to_num(log_phi, nan=0.0, posinf=50.0, neginf=-50.0)
        log_phi = log_phi.clamp(-1.0e4, 1.0e4)

        feature_dim = log_phi.shape[-1] if log_phi.dim() > 0 else 1

        if log_phi.numel() == 0:
            return log_phi.new_zeros(log_phi.shape) / (
                math.sqrt(float(feature_dim)) + 1e-6
            )

        # Token-independent stabilization scalar.
        # Per-sample max over all tokens and random-feature dimensions.
        reduce_dims = tuple(range(1, log_phi.dim()))
        if reduce_dims:
            shift = log_phi.detach().amax(dim=reduce_dims, keepdim=True)
        else:
            shift = log_phi.detach()

        shift = torch.nan_to_num(shift, nan=0.0, posinf=50.0, neginf=-50.0)

        log_phi = log_phi - shift
        phi = torch.exp(log_phi)
        phi = torch.nan_to_num(phi, nan=0.0, posinf=1.0, neginf=0.0)

        return phi / (math.sqrt(float(feature_dim)) + 1e-6)

    def forward(self, query_feat, context_feat) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C_q, H, W = query_feat.shape
        H_orig, W_orig = H, W

        orig_dtype = query_feat.dtype
        conv_dtype = self.query_conv.weight.dtype

        pad_h = (self.pad_factor - H % self.pad_factor) % self.pad_factor
        pad_w = (self.pad_factor - W % self.pad_factor) % self.pad_factor

        if pad_h > 0 or pad_w > 0:
            query_feat_padded = F.pad(query_feat, (0, pad_w, 0, pad_h))
        else:
            query_feat_padded = query_feat

        query_feat_padded = query_feat_padded.to(conv_dtype)
        context_feat = context_feat.to(conv_dtype)

        H_p, W_p = query_feat_padded.shape[-2:]

        q = self.query_conv(query_feat_padded).flatten(2).transpose(1, 2)
        k = self.key_conv(context_feat).flatten(2).transpose(1, 2)
        v = self.value_conv(context_feat).flatten(2).transpose(1, 2)

        q = q.float() * self.scale
        k = k.float() * self.scale
        v = v.float()

        phi_q = self._phi(q)
        phi_k = self._phi(k)

        mean_q = q.mean(dim=1, keepdim=True)
        mean_phi_q = self._phi(mean_q)

        grounding = torch.bmm(mean_phi_q, phi_k.transpose(1, 2))
        grounding = grounding / (grounding.sum(dim=-1, keepdim=True) + 1e-6)

        H_ctx, W_ctx = context_feat.shape[-2], context_feat.shape[-1]
        grounding_attn = grounding.view(B, 1, H_ctx, W_ctx).to(orig_dtype)

        k_context = torch.bmm(phi_k.transpose(1, 2), v)
        k_norm = phi_k.sum(dim=1, keepdim=True).transpose(1, 2)

        denom = torch.bmm(phi_q, k_norm).clamp_min(1e-6)
        out = torch.bmm(phi_q, k_context) / denom

        out = self.attn_drop(out).transpose(1, 2).reshape(B, C_q, H_p, W_p)
        out = out.to(conv_dtype)

        out = self.local_bias(out)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H_orig, :W_orig]

        enhancement = self.proj_drop(self.proj_norm(self.proj_conv(out)))
        enhancement = enhancement.to(orig_dtype)

        gamma = self.gamma.to(dtype=orig_dtype)

        return query_feat + gamma * enhancement, grounding_attn


class DECATHCHead(nn.Module):
    """Dynamic Evidential Clifford Algebra & Topo-Homological Classifier (DECA-THC).

    Replaces heuristic multi-branch feature concatenation with a rigorous
    non-Euclidean geometric algebra and algebraic topology formulation.

    Pipeline:
        A. CMLL  - Clifford Multivector Lifting: z1, z2 -> 1-vectors in Cl(d,0)
        B. GPFM  - Geometric Product Fusion:     v1 * v2 = S (grade-0) + B (grade-2)
        C. TPF   - Topological Persistence Filter: SVD-based spectral persistence
                   approximation of B -> topological signature vector
        D. SLEH  - Subjective Logic Evidential Head: concat(S, vech(B), t_topo) ->
                   Dirichlet concentration parameters via Softplus + 1

    The grade-0 scalar captures invariant anatomical consensus; the grade-2
    bivector encodes the orthogonal plane of feature divergence between
    unmasked and frequency-masked embeddings; the spectral topological
    signature extracts scale-invariant homological invariants; and the
    evidential head parameterizes a Dirichlet density yielding both
    diagnostic class expectations and exact epistemic vacuity.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        topo_k: int = 16,
        bivec_bottleneck_dim: int = 128,
        bivec_threshold: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert hidden_dim >= 2, (
            f"hidden_dim must be >= 2 for bivector construction, got {hidden_dim}"
        )
        assert num_classes >= 2, f"num_classes must be >= 2, got {num_classes}"

        self.d = hidden_dim
        self.C = num_classes
        self.eps = eps
        # top-k singular values: cannot exceed d
        self.topo_k = min(topo_k, hidden_dim)
        self.bivec_threshold = bivec_threshold

        bivec_dim = hidden_dim * (hidden_dim - 1) // 2

        # Dimensionality Bottleneck Contract: only apply if d > threshold
        if hidden_dim > bivec_threshold and bivec_dim > bivec_bottleneck_dim:
            self.bivec_bottleneck = nn.Sequential(
                nn.Linear(bivec_dim, bivec_bottleneck_dim),
                nn.GELU(),
                nn.Linear(bivec_bottleneck_dim, bivec_bottleneck_dim),
            )
            self.effective_bivec_dim = bivec_bottleneck_dim
        else:
            self.bivec_bottleneck = None
            self.effective_bivec_dim = bivec_dim

        # K_topo = topo_k + 3 power-sum moments (q in {1,2,3})
        topo_dim = self.topo_k + 3

        # Total input dimension to SLEH: 1 (scalar) + bivec_dim + topo_dim
        D_in = 1 + self.effective_bivec_dim + topo_dim
        self.D_in = D_in

        # Subjective Logic Evidential Head: linear projection -> Softplus
        self.evidential_linear = nn.Linear(D_in, num_classes)
        nn.init.trunc_normal_(self.evidential_linear.weight, std=0.02)
        nn.init.zeros_(self.evidential_linear.bias)

    @staticmethod
    def _geometric_product(z1: torch.Tensor, z2: torch.Tensor):
        """Clifford Geometric Product: M = v1 * v2 = <M>_0 + <M>_2.

        z1, z2: (B, d_proj) - coefficients of orthonormal basis blades
        Returns:
            S: (B, 1) - grade-0 scalar (symmetric inner product)
            B_mat: (B, d_proj, d_proj) - grade-2 skew-symmetric bivector
        """
        S = (z1 * z2).sum(dim=-1, keepdim=True)  # (B, 1)
        B_mat = z1.unsqueeze(2) * z2.unsqueeze(1) - z2.unsqueeze(2) * z1.unsqueeze(1)
        return S, B_mat

    def _topological_signature(self, B_mat: torch.Tensor) -> torch.Tensor:
        """
        True truncated spectral signature of the bivector matrix.

        Returns:
            tensor of shape (B, topo_k + 3)
            - first topo_k entries: largest singular values
            - last 3 entries: power-sum moments q=1,2,3
        """
        B = B_mat.float()
        B = torch.nan_to_num(B, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        batch_size = B.shape[0]

        if self.topo_k <= 0:
            top_k = B.new_zeros(batch_size, 0)
        else:
            singulars = None

            # Prefer a real spectral computation.
            # Detach to avoid unstable SVD backward in degenerate/repeated-spectrum cases.
            # The classifier still receives gradient through S and vech(B_mat).
            try:
                singulars = torch.linalg.svdvals(B).detach()
            except Exception:
                singulars = None

            if singulars is None or (not torch.isfinite(singulars).all()):
                # Robust fallback: one aggregate spectral surrogate.
                fro2 = (B * B).sum(dim=(1, 2)).clamp_min(0.0)
                sigma = torch.sqrt(0.5 * fro2 + self.eps).clamp_max(1e4).detach()

                top_k = B.new_zeros(batch_size, self.topo_k)
                top_k[:, 0] = sigma
            else:
                singulars = singulars.float().clamp_min(0.0).clamp_max(1e4)

                k = min(self.topo_k, singulars.shape[-1])
                top_k = singulars[:, :k]

                if k < self.topo_k:
                    pad = singulars.new_zeros(batch_size, self.topo_k - k)
                    top_k = torch.cat([top_k, pad], dim=-1)

        # Power-sum moments from retained spectrum.
        s = top_k.clamp_min(0.0)
        m1 = s.sum(dim=-1)
        m2 = (s * s).sum(dim=-1)
        m3 = (s * s * s).sum(dim=-1)

        moments = torch.stack([m1, m2, m3], dim=-1)
        out = torch.cat([top_k, moments], dim=-1)

        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

    @staticmethod
    def _vech_upper(B_mat: torch.Tensor) -> torch.Tensor:
        """Extract strict upper-triangular elements (vech operator).

        B_mat: (B, d, d)
        Returns: (B, d*(d-1)/2)
        """
        _, d, _ = B_mat.shape
        idx = torch.triu_indices(d, d, offset=1, device=B_mat.device)
        return B_mat[:, idx[0], idx[1]]

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, torch.Tensor]:
        z1 = z1.float()
        z2 = z2.float()

        if z1.dim() == 3:
            if z1.shape[1] == 1:
                z1 = z1.squeeze(1)
            elif z1.shape[-1] == 1:
                z1 = z1.squeeze(-1)
            else:
                z1 = z1[:, 0, :]

        if z2.dim() == 3:
            if z2.shape[1] == 1:
                z2 = z2.squeeze(1)
            elif z2.shape[-1] == 1:
                z2 = z2.squeeze(-1)
            else:
                z2 = z2[:, 0, :]

        if z1.dim() != 2:
            z1 = z1.reshape(z1.shape[0], -1)

        if z2.dim() != 2:
            z2 = z2.reshape(z2.shape[0], -1)

        if z1.shape[-1] != self.d or z2.shape[-1] != self.d:
            raise ValueError(
                f"DECA-THC expected embeddings of dimension {self.d}, "
                f"got z1={z1.shape[-1]}, z2={z2.shape[-1]}."
            )

        S, B_mat = self._geometric_product(z1, z2)

        S = torch.nan_to_num(S, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        B_mat = torch.nan_to_num(B_mat, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
            -1e4, 1e4
        )

        t_topo = self._topological_signature(B_mat)
        bvec = self._vech_upper(B_mat)

        if self.bivec_bottleneck is not None:
            lin1 = self.bivec_bottleneck[0]
            lin2 = self.bivec_bottleneck[2]

            h = F.linear(
                bvec,
                lin1.weight.float(),
                lin1.bias.float() if lin1.bias is not None else None,
            )
            h = F.gelu(h)

            bvec = F.linear(
                h,
                lin2.weight.float(),
                lin2.bias.float() if lin2.bias is not None else None,
            )

        bvec = torch.nan_to_num(bvec, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        t_topo = torch.nan_to_num(t_topo, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
            -1e4, 1e4
        )

        Omega = torch.cat([S, bvec, t_topo], dim=-1)
        Omega = torch.nan_to_num(Omega, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
            -1e4, 1e4
        )

        logits_raw = F.linear(
            Omega,
            self.evidential_linear.weight.float(),
            self.evidential_linear.bias.float()
            if self.evidential_linear.bias is not None
            else None,
        )

        logits_raw = torch.nan_to_num(logits_raw, nan=0.0, posinf=50.0, neginf=-50.0)

        evidence = F.softplus(logits_raw)
        evidence = torch.nan_to_num(evidence, nan=0.0, posinf=1e4, neginf=0.0).clamp(
            1e-6, 1e4
        )

        alpha = evidence + 1.0
        alpha = torch.nan_to_num(alpha, nan=1.0, posinf=1e4, neginf=1.0).clamp_min(
            1.0 + self.eps
        )

        S_total = alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        probs = alpha / S_total
        probs = torch.nan_to_num(probs, nan=1.0 / float(self.C), posinf=1.0, neginf=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        vacuity = float(self.C) / S_total
        vacuity = torch.nan_to_num(vacuity, nan=1.0, posinf=1.0, neginf=1.0).clamp(
            0.0, 1.0
        )

        return {
            "alpha": alpha,
            "evidence": evidence,
            "probs": probs,
            "vacuity": vacuity,
            "S": S,
            "B_mat": B_mat,
            "logits_raw": logits_raw,
        }


def evidential_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    lambda_kl: float = 1.0,
    epoch: int = 0,
    T_anneal: int = 10,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha = torch.nan_to_num(alpha, nan=1.0, posinf=1e4, neginf=1.0)
    alpha = alpha.float().clamp(min=1.0 + eps)

    if target.dim() == 2 and target.shape[-1] == num_classes:
        y = target.to(device=alpha.device, dtype=torch.float32)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = y.clamp_min(0.0)

        y_sum = y.sum(dim=-1, keepdim=True).clamp_min(eps)
        y = y / y_sum
    else:
        target = target.reshape(-1)
        target = target.long().clamp(0, num_classes - 1)

        y = F.one_hot(target, num_classes=num_classes).to(
            device=alpha.device, dtype=torch.float32
        )

    S = alpha.sum(dim=-1, keepdim=True).clamp_min(eps)

    L_err = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)
    L_err = torch.nan_to_num(L_err, nan=0.0, posinf=1e4, neginf=0.0)

    alpha_tilde = y + (1.0 - y) * alpha
    alpha_tilde = alpha_tilde.clamp(min=1.0 + eps)

    alpha_tilde_sum = alpha_tilde.sum(dim=-1, keepdim=True).clamp_min(eps)

    L_KL = (
        torch.lgamma(alpha_tilde_sum)
        - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
        + (
            (alpha_tilde - 1.0)
            * (torch.digamma(alpha_tilde) - torch.digamma(alpha_tilde_sum))
        ).sum(dim=-1, keepdim=True)
        - torch.lgamma(
            torch.tensor(float(num_classes), device=alpha.device, dtype=torch.float32)
        )
    ).squeeze(-1)

    L_KL = torch.nan_to_num(L_KL, nan=0.0, posinf=1e4, neginf=0.0)
    L_KL = L_KL.clamp_min(0.0)

    if T_anneal > 0:
        lambda_t = max(0.0, min(1.0, float(epoch) / float(T_anneal)))
    else:
        lambda_t = 1.0

    total = (L_err + lambda_t * float(lambda_kl) * L_KL).mean()
    total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)

    return total, L_err.detach().mean(), L_KL.detach().mean()


class GeoSpecClassifier(nn.Module):
    """Dual-branch unified pipeline with DECA-THC classifier head.

    Both side-vit branches (b1 unmasked, b2 causal-frequency-masked) execute
    continuously across train()/eval()/inference per the Phase-Invariance
    Contract. Their [CLS] embeddings are fused via the Dynamic Evidential
    Clifford Algebra & Topo-Homological Classifier (DECA-THC), which replaces
    heuristic concatenation with a rigorous Clifford geometric algebra
    formulation.
    """

    SIDE_VIT_INPUT_CHANNELS: int = 3

    def __init__(self, side_vit_b1: nn.Module, side_vit_b2: nn.Module, cfg: CoreConfig):
        super().__init__()
        raw_trainable = getattr(cfg.network, "backbone_trainable_layers", []) or []
        for i in raw_trainable:
            assert 0 <= int(i) <= 4

        raw_branch_cfg = getattr(
            cfg.network,
            "vit_feature_branch"
        )
        self.feature_branch = sorted([int(i) for i in raw_branch_cfg])
        for idx in self.feature_branch:
            assert 0 <= idx <= 3
        assert (
            len(self.feature_branch) == 2
            and self.feature_branch[0] != self.feature_branch[1]
        )

        self.num_classes = cfg.dataset.num_classes
        image_channels = cfg.dataset.image_channels
        side_input_size = cfg.network.side_input_size
        assert side_input_size % 16 == 0

        self._image_channels = image_channels
        self._side_vit_ch = self.SIDE_VIT_INPUT_CHANNELS
        self._side_input_size = side_input_size
        self.fold_id = getattr(cfg.network, "fold_id", 0)
        input_size = getattr(
            cfg.dataset, "input_size"
        )

        self.cnn_backbone = MultiScaleMedMambaTBackbone(
            in_chans=image_channels, pretrained=True
        )

        feat_dims = self.cnn_backbone.channels
        proj_channels, side_vit_ch = 64, self.SIDE_VIT_INPUT_CHANNELS
        f_low, f_high = self.feature_branch

        self.fusion_module = UnifiedManifoldFusion(
            low_level_channels=feat_dims[f_low],
            high_level_channels=feat_dims[f_high],
            output_channels=proj_channels,
            tucker_rank=16,
            input_size=input_size,
            low_stage_idx=f_low,
            high_stage_idx=f_high,
        )
        self.cross_attn = LinearRFFCrossAttention(
            query_channels=side_vit_ch,
            context_channels=proj_channels,
            output_channels=side_vit_ch,
            num_features=64,
            fold_id=self.fold_id,
            pad_factor=16,
        )
        self.refine = self._make_refinement(side_vit_ch)
        self.drop_path = DropPath(getattr(cfg.network, "drop_path_rate", 0.0))
        self.sv_input_norm = AffineInstanceNorm(side_vit_ch)

        if self._image_channels != self._side_vit_ch and not (
            self._image_channels == 1 and self._side_vit_ch == 3
        ):
            self._context_proj = nn.Conv2d(
                self._image_channels, self._side_vit_ch, kernel_size=1, bias=False
            )

        self.side_vit_b1 = side_vit_b1
        self.side_vit_b2 = side_vit_b2
        side_vit_output_hidden_size = self.side_vit_b1.side_encoder.hidden_size

        # === DECA-THC: replaces heuristic ClassifierHead ===
        # Dynamic Evidential Clifford Algebra & Topo-Homological Classifier
        self.deca_thc = DECATHCHead(
            hidden_dim=side_vit_output_hidden_size,
            num_classes=self.num_classes,
            topo_k=16,
            bivec_bottleneck_dim=128,
            bivec_threshold=64,
        )

        self.domain_classifier = DomainClassifier(
            proj_channels, getattr(cfg.dataset, "num_domains", 4)
        )

        self.register_buffer(
            "class_prototypes",
            torch.zeros(self.num_classes, side_vit_output_hidden_size),
        )
        self.register_buffer(
            "prototypes_initialized", torch.zeros(self.num_classes, dtype=torch.bool)
        )

        self.prototype_momentum = getattr(cfg.network, "prototype_momentum", 0.99)
        self.tta_lr = getattr(cfg.network, "tta_lr", 1e-3)
        self.tta_lambda_conf = getattr(cfg.network, "tta_lambda_conf", 1.0)
        self._last_aux_losses = {}

        self._tta_init_params = None

    @staticmethod
    def _make_refinement(channels: int) -> nn.Module:
        refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        nn.init.kaiming_normal_(refine[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(refine[3].weight, mode="fan_out", nonlinearity="linear")
        nn.init.constant_(refine[-1].weight, 1e-4)
        nn.init.zeros_(refine[-1].bias)
        return refine

    def _expand_context(self, x):
        if self._image_channels == self._side_vit_ch:
            return x
        if self._image_channels == 1 and self._side_vit_ch == 3:
            return x.repeat(1, 3, 1, 1)
        return self._context_proj(x)

    def _prepare_context(self, x):
        context = self._expand_context(x)
        return F.interpolate(
            context,
            size=(self._side_input_size, self._side_input_size),
            mode="bilinear",
            align_corners=False,
        )

    def _compute_hyperbolic_distance(
        self, cls_embedding: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.prototypes_initialized.all():
            return None

        c = self.fusion_module.curvature.float()
        eps = float(self.fusion_module.eps)

        z_poinc = self.fusion_module._expmap0(cls_embedding.float(), c)
        proto_poinc = self.fusion_module._expmap0(self.class_prototypes.float(), c)

        z_exp = z_poinc.unsqueeze(1)
        P_exp = proto_poinc.unsqueeze(0)

        neg_z = -z_exp

        dot_np = (neg_z * P_exp).sum(dim=-1, keepdim=True)
        norm_n_sq = (neg_z * neg_z).sum(dim=-1, keepdim=True)
        norm_p_sq = (P_exp * P_exp).sum(dim=-1, keepdim=True)

        numerator = (1.0 + 2.0 * c * dot_np + c * norm_p_sq) * neg_z + (
            1.0 - c * norm_n_sq
        ) * P_exp

        denominator = (
            1.0 + 2.0 * c * dot_np + (c**2) * norm_n_sq * norm_p_sq
        ).clamp_min(eps)

        mobius_add = numerator / denominator
        mobius_norm = torch.sqrt((mobius_add * mobius_add).sum(dim=-1) + eps)

        dist = (2.0 / torch.sqrt(c)) * torch.atanh(
            torch.clamp(torch.sqrt(c) * mobius_norm, max=1.0 - eps)
        )

        return torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)

    def _pipeline(
        self,
        side_vit_classifier,
        features,
        context,
        key_states,
        value_states,
        fusion_out=None,
        wavelet_mask=None,
        return_full_metrics: bool = True,
        hyperbolic_cache=None,
    ):
        if fusion_out is None:
            F_low, F_high = (
                features[self.feature_branch[0]],
                features[self.feature_branch[1]],
            )

            (
                proc_feat,
                C_conf,
                G_bands,
                x_style,
                aux_losses,
                F_hyp,
                new_hyperbolic_cache,
            ) = self.fusion_module(
                F_low,
                F_high,
                wavelet_mask,
                hyperbolic_cache=hyperbolic_cache,
            )
        else:
            proc_feat, C_conf, G_bands, x_style, aux_losses, F_hyp = fusion_out
            new_hyperbolic_cache = hyperbolic_cache

        vit_input, grounding_attn = self.cross_attn(context, proc_feat)

        vit_input = vit_input + self.drop_path(self.refine(vit_input))

        C_conf_resized = F.interpolate(
            C_conf,
            size=vit_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        vit_input = self.sv_input_norm(vit_input, C_conf_resized)

        cls_embedding = side_vit_classifier(vit_input, key_states, value_states)

        hyp_dist_matrix = (
            self._compute_hyperbolic_distance(cls_embedding)
            if return_full_metrics
            else None
        )

        return {
            "domain_logits": None,
            "aux_losses": aux_losses,
            "G_bands": G_bands,
            "C_conf": C_conf,
            "cls_embedding": cls_embedding,
            "x_style": x_style,
            "grounding_attn": grounding_attn,
            "hyp_dist_matrix": hyp_dist_matrix,
            "F_hyp": F_hyp,
            "hyperbolic_cache": new_hyperbolic_cache,
            "proc_feat": proc_feat,
        }

    def forward(
        self,
        x,
        key_states,
        value_states,
        use_causal_mask=True,
        compute_equivariance=True,
        suppress_side_effects=False,
    ):
        features = self.cnn_backbone(x)
        context = self._prepare_context(x)
        F_low, F_high = (
            features[self.feature_branch[0]],
            features[self.feature_branch[1]],
        )
        fusion_out_full = self.fusion_module(F_low, F_high, None)
        fusion_out_nomask = fusion_out_full[:6]
        hyperbolic_cache_nomask = fusion_out_full[6]
        F_hyp_nomask = fusion_out_nomask[5]

        if self.training and compute_equivariance:
            x_flip = torch.flip(x, dims=[-1])
            with torch.no_grad():
                features_flip = self.cnn_backbone(x_flip)
            F_low_flip = features_flip[self.feature_branch[0]].detach()
            F_high_flip = features_flip[self.feature_branch[1]].detach()
            _, _, _, _, _, F_hyp_flip, _ = self.fusion_module(
                F_low_flip, F_high_flip, None
            )
            sym_loss = F.l1_loss(
                F_hyp_nomask.float(),
                torch.flip(F_hyp_flip, dims=[-1]).float(),
            )
            sym_loss = torch.nan_to_num(sym_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            sym_loss = torch.zeros((), device=x.device, dtype=torch.float32)

        B, L = x.shape[0], self.fusion_module.L

        wavelet_mask = None
        if use_causal_mask:
            wavelet_mask = torch.ones((B, 3, L), device=x.device, dtype=torch.float32)
            wavelet_mask[:, 0:2, :] = 0.5
            if self.training:
                keep_prob = 0.5
                stochastic_mask = torch.bernoulli(
                    torch.full(
                        (B, 2, L),
                        keep_prob,
                        device=x.device,
                        dtype=torch.float32,
                    )
                )
                # Unit-expectation stochastic multiplier over the structural
                # base mask. Expected train-time mask equals eval-time mask:
                # E[0.5 * Bernoulli(0.5) / 0.5] = 0.5.
                wavelet_mask[:, 0:2, :] = (
                    wavelet_mask[:, 0:2, :] * stochastic_mask / keep_prob
                )

        compute_side_metrics = not suppress_side_effects
        out_b1 = self._pipeline(
            self.side_vit_b1,
            features,
            context,
            key_states,
            value_states,
            fusion_out=fusion_out_nomask,
            wavelet_mask=None,
            return_full_metrics=compute_side_metrics,
            hyperbolic_cache=hyperbolic_cache_nomask,
        )
        out_b2 = self._pipeline(
            self.side_vit_b2,
            features,
            context,
            key_states,
            value_states,
            fusion_out=None,
            wavelet_mask=wavelet_mask,
            return_full_metrics=False,
            hyperbolic_cache=hyperbolic_cache_nomask,
        )

        z1 = out_b1["cls_embedding"]
        z2 = out_b2["cls_embedding"]
        deca_out = self.deca_thc(z1, z2)

        if compute_side_metrics:
            domain_logits_list = []
            c = self.fusion_module.curvature
            # Shared hyperbolic style representation.
            # Branch-2 style is cache-derived and detached, so we intentionally
            # use the shared/unmasked style branch for adversarial style alignment.
            x_style_tan = self.fusion_module._logmap0(out_b1["x_style"], c)
            style_gap = F.adaptive_avg_pool2d(x_style_tan.float(), 1).flatten(1)
            style_gap = torch.nan_to_num(style_gap, nan=0.0, posinf=0.0, neginf=0.0)
            domain_logits_list.append(self.domain_classifier(style_gap))

            proc_gap_b1 = F.adaptive_avg_pool2d(out_b1["proc_feat"].float(), 1).flatten(
                1
            )
            proc_gap_b1 = torch.nan_to_num(proc_gap_b1, nan=0.0, posinf=0.0, neginf=0.0)
            domain_logits_list.append(self.domain_classifier(proc_gap_b1))

            proc_gap_b2 = F.adaptive_avg_pool2d(out_b2["proc_feat"].float(), 1).flatten(
                1
            )
            proc_gap_b2 = torch.nan_to_num(proc_gap_b2, nan=0.0, posinf=0.0, neginf=0.0)
            domain_logits_list.append(self.domain_classifier(proc_gap_b2))

            domain_logits = torch.stack(
                [logits.float() for logits in domain_logits_list], dim=0
            ).mean(dim=0)
            domain_logits = torch.nan_to_num(
                domain_logits, nan=0.0, posinf=0.0, neginf=0.0
            )

            domain_logits_style = domain_logits_list[0]
            domain_logits_b1 = domain_logits_list[1]
            domain_logits_b2 = domain_logits_list[2]

            z1_f = torch.nan_to_num(z1.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            z2_f = torch.nan_to_num(z2.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            z1_n = F.normalize(z1_f, p=2, dim=-1, eps=1e-6)
            z2_n = F.normalize(z2_f, p=2, dim=-1, eps=1e-6)
            cos_sim = F.cosine_similarity(z1_n, z2_n, dim=-1, eps=1e-6)
            branch_diversity_loss = 0.01 * cos_sim.pow(2).mean()
            branch_diversity_loss = torch.nan_to_num(
                branch_diversity_loss, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            domain_logits = None
            domain_logits_style = None
            domain_logits_b1 = None
            domain_logits_b2 = None
            branch_diversity_loss = torch.zeros(
                (), device=x.device, dtype=torch.float32
            )

        result = dict(out_b1)
        result["domain_logits"] = domain_logits
        result["domain_logits_style"] = domain_logits_style
        result["domain_logits_b1"] = domain_logits_b1
        result["domain_logits_b2"] = domain_logits_b2

        aux_losses = dict(out_b1["aux_losses"])
        aux_losses["branch_diversity"] = branch_diversity_loss
        result["aux_losses"] = aux_losses

        result["alpha"] = deca_out["alpha"]
        result["evidence"] = deca_out["evidence"]
        result["probs"] = deca_out["probs"]
        result["vacuity"] = deca_out["vacuity"]
        result["log_probs"] = torch.log(deca_out["probs"].clamp_min(1e-8))
        result["log_probs"] = torch.nan_to_num(
            result["log_probs"], nan=0.0, posinf=0.0, neginf=-1e8
        )


        result["logits"] = torch.nan_to_num(
            deca_out["logits_raw"], nan=0.0, posinf=50.0, neginf=-50.0
        )
        result["logits_raw"] = deca_out["logits_raw"]
        result["masked_cls_embedding"] = z2

        # High-priority hygiene:
        # masked branch aux losses are shared-cache zeros; detach them to avoid
        # accidental backward usage in external trainers.
        masked_aux_losses = out_b2["aux_losses"]
        result["masked_aux_losses"] = {
            k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in masked_aux_losses.items()
        }

        result["sym_loss"] = sym_loss
        result["branch_diversity_loss"] = branch_diversity_loss

        if result.get("hyperbolic_cache") is not None:
            result["hyperbolic_cache"] = tuple(
                t.detach() if torch.is_tensor(t) else t
                for t in result["hyperbolic_cache"]
            )

        if not suppress_side_effects:
            self._last_aux_losses = {
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in result["aux_losses"].items()
            }
        else:
            # Explicitly clear the cache during suppressed passes (e.g., TTA)
            # to release any previously held graph references and prevent memory leaks.
            self._last_aux_losses = {}

        return result

    def reset_tta_state(self):
        self._tta_init_params = {
            n: p.data.clone() for n, p in self.sv_input_norm.named_parameters()
        }
        self._tta_optimizer = None

    def tta_step(self, x, key_states, value_states):
        if self._tta_init_params is None:
            self.reset_tta_state()

        req_grad_state = {n: p.requires_grad for n, p in self.named_parameters()}
        was_training = self.training

        try:
            for n, p in self.named_parameters():
                p.requires_grad = False

            tta_params = []
            for n, p in self.sv_input_norm.named_parameters():
                p.requires_grad = True
                tta_params.append(p)

            if getattr(self, "_tta_optimizer", None) is None:
                self._tta_optimizer = torch.optim.SGD(
                    tta_params, lr=self.tta_lr, momentum=0.9
                )
            else:
                self._tta_optimizer.param_groups[0]["params"] = tta_params

            self.eval()

            outputs = self.forward(
                x,
                key_states,
                value_states,
                use_causal_mask=False,
                compute_equivariance=False,
                suppress_side_effects=True,
            )

            probs = outputs["probs"].float()
            probs = torch.nan_to_num(
                probs, nan=1.0 / float(self.num_classes), posinf=1.0, neginf=0.0
            )
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            C_conf = outputs["C_conf"].float()
            C_conf = torch.nan_to_num(C_conf, nan=0.0, posinf=1.0, neginf=0.0).clamp(
                0.0, 1.0
            )

            vacuity = outputs["vacuity"].float()
            vacuity = torch.nan_to_num(vacuity, nan=1.0, posinf=1.0, neginf=1.0).clamp(
                0.0, 1.0
            )

            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
            entropy = torch.nan_to_num(
                entropy, nan=float("inf"), posinf=float("inf"), neginf=0.0
            )

            e0 = 0.4 * math.log(self.num_classes)

            conf_per_sample = C_conf.mean(dim=[1, 2, 3])
            conf_per_sample = torch.nan_to_num(
                conf_per_sample, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)

            vacuity_flat = vacuity.squeeze(-1)

            finite_mask = (
                torch.isfinite(entropy)
                & torch.isfinite(conf_per_sample)
                & torch.isfinite(vacuity_flat)
            )

            entropy_reliable = entropy < e0
            conf_reliable = conf_per_sample > 0.5
            vacuity_reliable = vacuity_flat < 0.5

            reliable_mask = (
                finite_mask & entropy_reliable & conf_reliable & vacuity_reliable
            )

            if reliable_mask.any():
                conf_weight = self.tta_lambda_conf * conf_per_sample[reliable_mask] + (
                    1.0 - self.tta_lambda_conf
                )

                ent_loss = (entropy[reliable_mask] * conf_weight).mean()

                reg_terms = [
                    (
                        (
                            param
                            - self._tta_init_params[n].to(
                                device=param.device, dtype=param.dtype
                            )
                        )
                        ** 2
                    ).sum()
                    for n, param in self.sv_input_norm.named_parameters()
                ]

                reg_loss = (
                    sum(reg_terms)
                    if reg_terms
                    else torch.zeros((), device=x.device, dtype=torch.float32)
                )

                loss = ent_loss + 0.1 * reg_loss

                if torch.isfinite(loss):
                    self._tta_optimizer.zero_grad()
                    loss.backward()
                    self._tta_optimizer.step()

        finally:
            for n, p in self.named_parameters():
                p.requires_grad = req_grad_state[n]

            if was_training:
                self.train()

    def update_prototypes(self, cls_embeddings, labels, predictions, momentum=None):
        if momentum is None:
            momentum = self.prototype_momentum

        momentum = float(momentum)
        if not math.isfinite(momentum):
            momentum = 0.99

        momentum = min(max(momentum, 0.0), 1.0)

        # ------------------------------------------------------------
        # Normalize predictions
        # ------------------------------------------------------------
        if predictions.dim() == 3:
            if predictions.shape[1] == 1:
                predictions = predictions.squeeze(1)
            elif predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            else:
                predictions = predictions[:, 0, :]

        if predictions.dim() == 2:
            if predictions.shape[-1] == self.num_classes:
                predictions = predictions.argmax(dim=-1)
            else:
                predictions = predictions.reshape(-1)

        predictions = predictions.reshape(-1).long()

        # ------------------------------------------------------------
        # Normalize labels
        # ------------------------------------------------------------
        if labels.dim() == 3:
            if labels.shape[1] == 1:
                labels = labels.squeeze(1)
            elif labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            else:
                labels = labels[:, 0, :]

        if labels.dim() == 2:
            if labels.shape[-1] == self.num_classes:
                labels = labels.argmax(dim=-1)
            else:
                labels = labels.reshape(-1)

        labels = labels.reshape(-1).long()

        # ------------------------------------------------------------
        # Normalize embeddings
        # ------------------------------------------------------------
        if cls_embeddings.dim() == 3:
            if cls_embeddings.shape[1] == 1:
                cls_embeddings = cls_embeddings.squeeze(1)
            elif cls_embeddings.shape[-1] == 1:
                cls_embeddings = cls_embeddings.squeeze(-1)
            else:
                cls_embeddings = cls_embeddings[:, 0, :]

        if cls_embeddings.dim() != 2:
            cls_embeddings = cls_embeddings.reshape(predictions.shape[0], -1)

        if cls_embeddings.shape[0] != predictions.shape[0]:
            raise ValueError(
                f"cls_embeddings batch size ({cls_embeddings.shape[0]}) "
                f"does not match predictions batch size ({predictions.shape[0]})."
            )

        if cls_embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"cls_embeddings batch size ({cls_embeddings.shape[0]}) "
                f"does not match labels batch size ({labels.shape[0]})."
            )

        if cls_embeddings.shape[-1] != self.class_prototypes.shape[-1]:
            raise ValueError(
                f"cls_embeddings feature dimension ({cls_embeddings.shape[-1]}) "
                f"does not match prototype dimension ({self.class_prototypes.shape[-1]})."
            )

        cls_embeddings = torch.nan_to_num(
            cls_embeddings.float(),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        ).clamp(-1e4, 1e4)

        cls_embeddings = cls_embeddings.to(
            device=self.class_prototypes.device,
            dtype=self.class_prototypes.dtype,
        )

        labels = labels.to(cls_embeddings.device)
        predictions = predictions.to(cls_embeddings.device)

        finite_rows = torch.isfinite(cls_embeddings).all(dim=-1)


        for cls in range(self.num_classes):
            gt_mask = finite_rows & (labels == cls)

            if not gt_mask.any():
                continue

            gt_embeddings = cls_embeddings[gt_mask]
            batch_proto = gt_embeddings.mean(dim=0)

            batch_proto = torch.nan_to_num(
                batch_proto,
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )

            if not torch.isfinite(batch_proto).all():
                continue

            if not self.prototypes_initialized[cls]:
                self.class_prototypes.data[cls] = batch_proto
                self.prototypes_initialized.data[cls] = True
            else:
                correct_mask = gt_mask & (predictions == labels)

                if correct_mask.any():
                    update_embeddings = cls_embeddings[correct_mask]
                else:
                    update_embeddings = gt_embeddings

                update_proto = update_embeddings.mean(dim=0)

                update_proto = torch.nan_to_num(
                    update_proto,
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )

                if not torch.isfinite(update_proto).all():
                    continue

                self.class_prototypes.data[cls] = (
                    momentum * self.class_prototypes.data[cls]
                    + (1.0 - momentum) * update_proto
                )

    def compute_hsm(self, x, target_class, key_states, value_states):
        was_training = self.training
        self.eval()
        try:
            if isinstance(target_class, int) or (
                isinstance(target_class, torch.Tensor) and target_class.ndim == 0
            ):
                cls_idx = int(target_class)
                if not self.prototypes_initialized[cls_idx]:
                    raise ValueError(
                        f"Prototype for class {cls_idx} has not been initialized."
                    )
                target_idx = torch.full(
                    (x.shape[0],), cls_idx, device=x.device, dtype=torch.long
                )
            elif isinstance(target_class, torch.Tensor) and target_class.ndim == 1:
                if target_class.shape[0] != x.shape[0]:
                    raise ValueError(
                        f"Target class batch size ({target_class.shape[0]}) "
                        f"must match input batch size ({x.shape[0]})."
                    )
                target_idx = target_class.to(x.device, dtype=torch.long)
                if not self.prototypes_initialized[target_idx].all():
                    raise ValueError(
                        "Not all target class prototypes have been initialized."
                    )
            else:
                raise TypeError(
                    "target_class must be an int, a 0D tensor, or a 1D batched tensor."
                )
            try:
                model_dtype = next(self.cnn_backbone.parameters()).dtype
            except StopIteration:
                model_dtype = self.fusion_module.W_low.weight.dtype

            if not model_dtype.is_floating_point:
                model_dtype = torch.float32

            def _cast_floating_tensors(obj, dtype):
                if torch.is_tensor(obj):
                    if obj.is_floating_point():
                        return obj.to(dtype=dtype)
                    return obj
                if isinstance(obj, list):
                    return [_cast_floating_tensors(v, dtype) for v in obj]
                if isinstance(obj, tuple):
                    casted = tuple(_cast_floating_tensors(v, dtype) for v in obj)
                    if hasattr(obj, "_make"):
                        return obj._make(casted)
                    return casted
                if isinstance(obj, dict):
                    return {k: _cast_floating_tensors(v, dtype) for k, v in obj.items()}
                return obj

            x = x.clone().detach().to(dtype=model_dtype).requires_grad_(True)

            # Enforce the same dtype boundary for injected FPT+ key/value
            # tensors without modifying black-box internals or tensor shapes.
            key_states = _cast_floating_tensors(key_states, model_dtype)
            value_states = _cast_floating_tensors(value_states, model_dtype)

            with torch.enable_grad():
                outputs = self.forward(
                    x,
                    key_states,
                    value_states,
                    use_causal_mask=False,
                    compute_equivariance=False,
                    suppress_side_effects=True,
                )
                target_evidence = (
                    outputs["alpha"].gather(1, target_idx.unsqueeze(1)).sum()
                )
                target_evidence = torch.nan_to_num(
                    target_evidence, nan=0.0, posinf=0.0, neginf=0.0
                )
                grad = torch.autograd.grad(target_evidence, x, create_graph=False)[0]

            grad = grad.detach().float()
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.sqrt((grad * grad).sum(dim=1) + 1e-8)
        finally:
            if was_training:
                self.train()
