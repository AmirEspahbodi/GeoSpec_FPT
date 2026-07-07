import math
from typing import Any, Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
#  DropPath (Stochastic Depth) Implementation
#
#  [Shared module — identical implementation in both Branch 1 and Branch 2.
#   Each branch instantiates its own DropPath for independent stochastic
#   depth masks.]
# ===========================================================================
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


# ===========================================================================
#  Identity-initialized per-channel affine with geometric confidence
#  modulation.
#
#  Affine_modulated(x) = x * (gamma + beta * C_conf) + delta
#
#  At init: gamma=1, beta=0, delta=0  =>  TRUE identity.
#  As training progresses, beta learns to scale the input based on the
#  geometric certainty of spatial regions—actively suppressing regions
#  identified as noisy by the hyperbolic manifold's conformal factor.
#
#  [From Branch 2 — backward compatible with Branch 1.
#   When conf=None (Branch 1 usage), output = x * gamma + delta,
#   which is mathematically identical to Branch 1's IdentityInitAffine.
#   The extra beta parameter is zero-initialized and dormant when
#   conf is not provided.]
# ===========================================================================
class IdentityInitAffine(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))  # gamma
        self.beta = nn.Parameter(torch.zeros(num_channels))  # beta (confidence)
        self.bias = nn.Parameter(torch.zeros(num_channels))  # delta

    def forward(
        self, x: torch.Tensor, conf: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        gamma = self.weight.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        delta = self.bias.view(1, -1, 1, 1)
        if conf is not None:
            # conf: (B, 1, H, W) — broadcasts with per-channel gamma/beta
            return x * (gamma + beta * conf) + delta
        return x * gamma + delta


# ===========================================================================
#  Gated Feature Fusion Head (VRAM Efficient Classifier) — Dual-Branch
#
#  Extended from the single-branch GatedFeatureFusionHead used in both
#  Branch 1 (parameter name: hidden_size) and Branch 2 (parameter name:
#  input_size). The dual-branch version accepts TWO independent side-ViT
#  embeddings and fuses them via:
#    1. Per-branch LayerNorm + self-gating (channel-wise feature selection)
#    2. Dynamic cross-branch weighting (sample-adaptive branch importance
#       via softmax over 2 scalar weights)
#    3. GELU activation + dropout + linear projection to num_classes
#
#  This design preserves the gated fusion philosophy of both original
#  heads while enabling the two branches to contribute adaptively based
#  on per-sample signal quality.
# ===========================================================================
class GatedFeatureFusionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        reduction: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Per-branch LayerNorm (applied independently to each branch embedding)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Per-branch self-gating (channel-wise feature selection)
        self.gate1 = nn.Sequential(
            nn.Linear(hidden_size, max(hidden_size // reduction, 1)),
            nn.GELU(),
            nn.Linear(max(hidden_size // reduction, 1), hidden_size),
            nn.Sigmoid(),
        )
        self.gate2 = nn.Sequential(
            nn.Linear(hidden_size, max(hidden_size // reduction, 1)),
            nn.GELU(),
            nn.Linear(max(hidden_size // reduction, 1), hidden_size),
            nn.Sigmoid(),
        )

        # Dynamic cross-branch weighting (sample-adaptive branch importance)
        self.branch_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, max(hidden_size // reduction, 1)),
            nn.GELU(),
            nn.Linear(max(hidden_size // reduction, 1), 2),
            nn.Softmax(dim=-1),
        )

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        for m in (
            list(self.gate1.modules())
            + list(self.gate2.modules())
            + list(self.branch_gate.modules())
        ):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for gate in [self.gate1, self.gate2]:
            final_linear = gate[-2]
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, 4.0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Normalize each branch embedding independently
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        # Per-branch self-gating
        g1 = self.gate1(x1)
        g2 = self.gate2(x2)
        x1 = x1 * g1
        x2 = x2 * g2

        # Dynamic cross-branch weighting (softmax ensures weights sum to 1)
        branch_weights = self.branch_gate(torch.cat([x1, x2], dim=-1))  # (B, 2)
        fused = (
            branch_weights[:, 0:1] * x1 + branch_weights[:, 1:2] * x2
        )  # (B, hidden_size)

        fused = self.act(fused)
        fused = self.drop(fused)
        logits = self.proj(fused)
        return logits


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
            out_indices=(0, 1, 2, 3),  # ConvNeXtV2 has 4 feature stages
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


# ===========================================================================
#  AWT-TF: Adaptive Wavelet-Tucker Fusion with Dual-Domain Spectral Gating
#
#  Replaces GatedAttentionModule with a paradigm that decouples feature
#  alignment into the frequency domain (Multi-Level Haar DWT) and semantic
#  interaction into the tensor algebra domain (Dynamic Tucker Decomposition).
#
#  Four synergistic mechanisms:
#    1. Multi-Level Adaptive Haar DWT — lossless resolution alignment.
#    2. Dynamic Tucker Decomposition — higher-order (rank-r²) cross-layer
#       interaction via a dual-pathway predicted core tensor.
#    3. Channel-Wise Dynamic Spectral Gating with Cross-Band Coherence —
#       efficient channel-wise gates modulated by a learnable 3×3
#       cross-band interaction matrix.
#    4. Identity-Anchored Multi-Scale Stabilization — zero-initialized
#       residual skip preventing distribution drift during early PEFT.
# ===========================================================================
class AWTTFModule(nn.Module):
    """Adaptive Wavelet-Tucker Fusion (AWT-TF) module.

    Parameters
    ----------
    low_level_channels : int
        Channel dimension of the shallow (high-resolution) feature map X_low.
    high_level_channels : int
        Channel dimension of the deep (low-resolution) feature map X_high.
    output_channels : int
        Output channel dimension C_out.
    tucker_rank : int, optional
        Rank r for the Tucker decomposition (default: 16). Controls the
        higher-order interaction capacity (rank r² cross-layer interactions).
    """

    def __init__(
        self,
        low_level_channels: int,
        high_level_channels: int,
        output_channels: int,
        tucker_rank: int = 16,
    ):
        super().__init__()
        self.low_level_channels = low_level_channels
        self.high_level_channels = high_level_channels
        self.output_channels = output_channels
        self.r = tucker_rank

        C_low = low_level_channels
        C_high = high_level_channels
        C_out = output_channels
        r = tucker_rank

        # ---- Step 1: Channel Alignment ----
        # Project deep features to match shallow channel dimension.
        self.high_align = nn.Conv2d(C_high, C_low, kernel_size=1, bias=False)

        # ---- Step 3: Dual-Pathway Dynamic Core Prediction ----
        # Spatial pathway: GAP(X_high_tilde) || GAP(LL_L) -> MLP -> [B, r, r, C_out]
        spatial_in_dim = 2 * C_low
        spatial_hidden = max(spatial_in_dim // 4, 32)
        spatial_out_dim = r * r * C_out
        self.mlp_spatial = nn.Sequential(
            nn.Linear(spatial_in_dim, spatial_hidden),
            nn.GELU(),
            nn.Linear(spatial_hidden, spatial_out_dim),
        )

        # Spectral pathway: GAP(Conv1x1(LL_L)) -> reshape -> [B, r, r, C_out]
        # The Conv1x1 has r*r*C_out output channels. After GAP and reshape,
        # it yields the spectral core factor.
        # NOTE: Since Conv1x1 is a linear pointwise operation, it commutes
        # with GAP: GAP(Conv1x1(X)) == Conv1x1(GAP(X)). We exploit this
        # mathematical equivalence in forward() to avoid materializing the
        # large (B, r²·C_out, H, W) intermediate tensor, saving substantial
        # VRAM with zero numerical difference.
        self.spectral_conv = nn.Conv2d(C_low, r * r * C_out, kernel_size=1, bias=False)

        # ---- Step 4: Static Factor Projection ----
        # Low-rank projections into Tucker factor space.
        self.U_high = nn.Conv2d(C_low, r, kernel_size=1, bias=False)
        self.U_LL = nn.Conv2d(C_low, r, kernel_size=1, bias=False)

        # ---- Step 6: Cross-Band Spectral Gating ----
        # Shared 1x1 projections for high-frequency bands (C_low -> C_out).
        # Separate per band-type (LH/HL/HH) but shared across wavelet levels
        # for parameter efficiency.
        self.band_proj_lh = nn.Conv2d(C_low, C_out, kernel_size=1, bias=False)
        self.band_proj_hl = nn.Conv2d(C_low, C_out, kernel_size=1, bias=False)
        self.band_proj_hh = nn.Conv2d(C_low, C_out, kernel_size=1, bias=False)

        # Shared MLP for band gate prediction (3*C_low -> 3*C_out).
        band_in_dim = 3 * C_low
        band_hidden = max(band_in_dim // 4, 32)
        band_out_dim = 3 * C_out
        self.mlp_bands = nn.Sequential(
            nn.Linear(band_in_dim, band_hidden),
            nn.GELU(),
            nn.Linear(band_hidden, band_out_dim),
        )

        # Learnable 3x3 cross-band coherence matrix.
        # Models physical co-occurrence of edge directions across
        # frequency sub-bands (e.g., horizontal edges in LH correlate
        # with vertical edges in HL at corner structures).
        # Initialized to zeros => softmax yields uniform 1/3 mixing.
        self.M_cross = nn.Parameter(torch.zeros(3, 3))

        # ---- Step 8: Final Projection and Identity-Anchored Skip ----
        self.final_proj = nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)
        self.final_norm = nn.GroupNorm(num_groups=1, num_channels=C_out)
        self.act = nn.GELU()

        # Skip projection from original shallow features.
        self.skip_proj = nn.Conv2d(C_low, C_out, kernel_size=1, bias=False)
        # Learnable scalar alpha, initialized to 0 for identity anchoring.
        self.alpha = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights for stable PEFT training.

        Key design decisions:
        - Xavier uniform for all projection convs (standard practice).
        - Truncated normal for MLP weights (transformer convention).
        - Zero-init final GroupNorm => main pathway outputs 0 at epoch 0.
        - alpha = 0 => skip contributes 0 at epoch 0.
        Together these guarantee Y = 0 at initialization, preventing any
        distribution drift into the downstream side-ViT. The cross-attention
        module's own gamma=0 then ensures the side-ViT receives the raw
        image context. As training progresses, alpha and GroupNorm weights
        grow autonomously, gradually introducing Tucker-fused semantics.
        """
        # Channel alignment
        nn.init.xavier_uniform_(self.high_align.weight)

        # Static factor projections
        nn.init.xavier_uniform_(self.U_high.weight)
        nn.init.xavier_uniform_(self.U_LL.weight)

        # Band projections
        for conv in [self.band_proj_lh, self.band_proj_hl, self.band_proj_hh]:
            nn.init.xavier_uniform_(conv.weight)

        # Spectral conv
        nn.init.xavier_uniform_(self.spectral_conv.weight)

        # MLPs (truncated normal, transformer convention)
        for m in self.mlp_spatial.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.mlp_bands.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Final projection
        nn.init.xavier_uniform_(self.final_proj.weight)
        # Zero-init GroupNorm: main pathway = 0 at init => identity anchoring
        nn.init.zeros_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

        # Skip projection (Xavier; alpha=0 controls contribution)
        nn.init.xavier_uniform_(self.skip_proj.weight)
        # alpha is already zero from __init__

    # -------------------------------------------------------------------
    #  Haar Wavelet Transform helpers (parameter-free, autograd-safe)
    # -------------------------------------------------------------------
    @staticmethod
    def _haar_dwt_2d(x: torch.Tensor):
        """Single-level 2D Haar Discrete Wavelet Transform.

        Decomposes x into four sub-bands (LL, LH, HL, HH), each at half
        spatial resolution. The Haar orthonormal normalization factor 1/2
        (from 1/√2 × 1/√2 in separable 1D application) is used.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, C, H, W) with H, W even.

        Returns
        -------
        (LL, LH, HL, HH) : each (B, C, H/2, W/2)
            LL: low-low (approximation)
            LH: low-high (horizontal detail)
            HL: high-low (vertical detail)
            HH: high-high (diagonal detail)
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, (
            f"Haar DWT requires even spatial dimensions, got ({H}, {W})"
        )
        # Extract 2x2 block quadrants via stride-2 slicing
        a = x[..., 0::2, 0::2]  # top-left
        b = x[..., 0::2, 1::2]  # top-right
        c = x[..., 1::2, 0::2]  # bottom-left
        d = x[..., 1::2, 1::2]  # bottom-right

        LL = (a + b + c + d) * 0.5
        LH = (a - b + c - d) * 0.5
        HL = (a + b - c - d) * 0.5
        HH = (a - b - c + d) * 0.5
        return LL, LH, HL, HH

    @staticmethod
    def _haar_idwt_2d(
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        """Single-level 2D Inverse Haar Discrete Wavelet Transform.

        Reconstructs the full-resolution tensor from four sub-bands.
        This is the exact mathematical inverse of _haar_dwt_2d.

        Implementation uses stack + reshape + permute + reshape (autograd-safe,
        no in-place operations on leaf tensors).

        Parameters
        ----------
        ll, lh, hl, hh : torch.Tensor
            Sub-bands each of shape (B, C, H/2, W/2).

        Returns
        -------
        torch.Tensor of shape (B, C, H, W)
        """
        B, C, H_half, W_half = ll.shape

        # Reconstruct the four quadrants of each 2x2 block
        a = (ll + lh + hl + hh) * 0.5  # top-left
        b = (ll - lh + hl - hh) * 0.5  # top-right
        c = (ll + lh - hl - hh) * 0.5  # bottom-left
        d = (ll - lh - hl + hh) * 0.5  # bottom-right

        # Interleave quadrants into full-resolution tensor.
        # Stack: (B, C, H/2, W/2, 4) where last dim indexes [a, b, c, d]
        out = torch.stack([a, b, c, d], dim=-1)
        # Reshape last dim: 4 -> (2, 2) representing (row_in_block, col_in_block)
        out = out.reshape(B, C, H_half, W_half, 2, 2)
        # Permute to interleave: (B, C, H/2, 2, W/2, 2)
        # so that H/2 and 2 combine into H, and W/2 and 2 combine into W
        out = out.permute(0, 1, 2, 4, 3, 5)
        # Final reshape to (B, C, H, W)
        out = out.reshape(B, C, H_half * 2, W_half * 2)
        return out

    @staticmethod
    def _compute_dwt_levels(low_size: int, high_size: int) -> Tuple[int, bool]:
        """Compute the number of DWT levels L to align spatial dimensions.
        Returns (L, is_feasible). Falls back to 0 if dimensions are incompatible.
        """
        if low_size < high_size:
            return 0, False

        ratio = low_size / high_size
        # Check if ratio is an integer
        if abs(ratio - round(ratio)) > 1e-6:
            return 0, False

        ratio_int = int(round(ratio))
        L = 0
        r = ratio_int
        while r > 1:
            if r % 2 != 0:
                return 0, False
            r //= 2
            L += 1

        # Check if all intermediate spatial dimensions are even
        temp_h = low_size
        for _ in range(L):
            if temp_h % 2 != 0:
                return 0, False
            temp_h //= 2

        return L, True

    def forward(
        self,
        low_level_feat: torch.Tensor,
        high_level_feat: torch.Tensor,
    ) -> torch.Tensor:
        B, C_low, H_low, W_low = low_level_feat.shape
        _, C_high, H_high, W_high = high_level_feat.shape

        # Determine the number of DWT levels needed for spatial alignment
        L_h, feasible_h = self._compute_dwt_levels(H_low, H_high)
        L_w, feasible_w = self._compute_dwt_levels(W_low, W_high)

        is_feasible = feasible_h and feasible_w and (L_h == L_w)
        L_actual = L_h if is_feasible else 0

        # Step 1: Channel Alignment
        X_high_tilde = self.high_align(high_level_feat)

        # Step 2: Multi-Level Haar Wavelet Decomposition (with fallback)
        if L_actual > 0:
            LL = low_level_feat
            high_freq_bands: List = []
            for _ in range(L_actual):
                LL, LH, HL, HH = self._haar_dwt_2d(LL)
                high_freq_bands.append((LH, HL, HH))
            LL_L = LL
        else:
            # Fallback for incompatible spatial dimensions
            LL_L = F.interpolate(
                low_level_feat,
                size=X_high_tilde.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            high_freq_bands = []

        # Step 3: Dual-Pathway Dynamic Core Prediction
        g_spatial = torch.cat(
            [
                F.adaptive_avg_pool2d(X_high_tilde, 1).flatten(1),
                F.adaptive_avg_pool2d(LL_L, 1).flatten(1),
            ],
            dim=1,
        )
        G_spatial = self.mlp_spatial(g_spatial)
        G_spatial = G_spatial.view(B, self.r, self.r, self.output_channels)

        g_LL = F.adaptive_avg_pool2d(LL_L, 1)
        spectral_feat = self.spectral_conv(g_LL)
        g_spectral = spectral_feat.flatten(1)
        G_spectral = g_spectral.view(B, self.r, self.r, self.output_channels)

        G = G_spatial * G_spectral

        # Step 4: Static Factor Projection
        P_high = self.U_high(X_high_tilde)
        P_LL = self.U_LL(LL_L)

        # Step 5: Higher-Order Tucker Reconstruction
        LL_fused = torch.einsum("bijk,bihw,bjhw->bkhw", G, P_high, P_LL)

        # Step 6: Channel-Wise Dynamic Spectral Gating
        M_cross_softmax = F.softmax(self.M_cross, dim=-1)

        modulated_bands: List = []
        for l in range(L_actual):
            LH_l, HL_l, HH_l = high_freq_bands[l]
            stacked = torch.cat([LH_l, HL_l, HH_l], dim=1)
            g_bands = F.adaptive_avg_pool2d(stacked, 1).flatten(1)
            G_bands = self.mlp_bands(g_bands)
            G_bands = G_bands.view(B, 3, self.output_channels)
            G_bands = torch.sigmoid(G_bands)
            G_hat = torch.einsum("ij,bjk->bik", M_cross_softmax, G_bands)

            LH_proj = self.band_proj_lh(LH_l)
            HL_proj = self.band_proj_hl(HL_l)
            HH_proj = self.band_proj_hh(HH_l)

            gate_lh = G_hat[:, 0, :].unsqueeze(-1).unsqueeze(-1)
            gate_hl = G_hat[:, 1, :].unsqueeze(-1).unsqueeze(-1)
            gate_hh = G_hat[:, 2, :].unsqueeze(-1).unsqueeze(-1)

            tilde_LH = LH_proj * gate_lh
            tilde_HL = HL_proj * gate_hl
            tilde_HH = HH_proj * gate_hh

            modulated_bands.append((tilde_LH, tilde_HL, tilde_HH))

        # Step 7: Multi-Level Wavelet Reconstruction
        current_LL = LL_fused
        for l in range(L_actual - 1, -1, -1):
            tilde_LH, tilde_HL, tilde_HH = modulated_bands[l]
            current_LL = self._haar_idwt_2d(current_LL, tilde_LH, tilde_HL, tilde_HH)

        if L_actual == 0:
            current_LL = F.interpolate(
                current_LL, size=(H_low, W_low), mode="bilinear", align_corners=False
            )
        X_fused = current_LL

        # Step 8: Identity-Anchored Output
        out = self.final_proj(X_fused)
        out = self.final_norm(out)
        out = self.act(out)

        skip = self.skip_proj(low_level_feat)
        out = out + self.alpha * skip

        return out


# ===========================================================================
#  Dense Linearized Cross-Attention (LinearRFF)
#  Approximates softmax attention via Random Fourier Features for O(N) scaling.
#  Provides dense, continuous pixel-level global context.
# ===========================================================================
class LinearRFFCrossAttention(nn.Module):
    def __init__(
        self,
        query_channels: int,
        context_channels: int,
        output_channels: int,
        num_features: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        # FIX 3: Enforce channel equality for residual addition
        assert query_channels == output_channels, (
            f"query_channels ({query_channels}) must equal output_channels ({output_channels}) "
            f"for residual addition in LinearRFFCrossAttention."
        )
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

        self.register_buffer("W", torch.randn(num_features, inter_channels))
        self.num_features = num_features

        # d^{-1/4} so that (q·scale)(k·scale) = q·k / √d (standard attention)
        self.scale = inter_channels**-0.25
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

        # FIX: Initialize to a small non-zero value (1e-2) instead of 0.0
        # This breaks the dead-gradient cycle while preserving identity init.
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
        orig_dtype = x.dtype
        x = x.float()
        x_proj = x @ self.W.float().t()
        x_norm_sq = (x**2).sum(dim=-1, keepdim=True) / 2.0
        log_phi = x_proj - x_norm_sq
        log_phi = torch.clamp(log_phi, max=0.0)  # Prevent exp overflow
        return torch.exp(log_phi).to(orig_dtype)

    def forward(
        self, query_feat: torch.Tensor, context_feat: torch.Tensor
    ) -> torch.Tensor:
        B, C_q, H, W = query_feat.shape

        q = self.query_conv(query_feat).flatten(2).transpose(1, 2)
        k = self.key_conv(context_feat).flatten(2).transpose(1, 2)
        v = self.value_conv(context_feat).flatten(2).transpose(1, 2)

        q = q * self.scale
        k = k * self.scale

        phi_q = self._phi(q)
        phi_k = self._phi(k)

        k_context = torch.bmm(phi_k.transpose(1, 2), v)
        k_norm = phi_k.sum(dim=1, keepdim=True).transpose(1, 2)

        num = torch.bmm(phi_q, k_context)
        den = torch.bmm(phi_q, k_norm).clamp_min(1e-6)

        out = num / den
        out = self.attn_drop(out)

        out = out.transpose(1, 2).reshape(B, C_q, H, W)
        out = self.local_bias(out)

        enhancement = self.proj_conv(out)
        enhancement = self.proj_norm(enhancement)
        enhancement = self.proj_drop(enhancement)

        fused = query_feat + self.gamma * enhancement
        return fused


# ===========================================================================
#  LC-HPHF v2: Learnable-Curvature Dual-Path Hyperbolic Poincaré
#  Hierarchical Fusion
# ===========================================================================
class LCHPHFv2(nn.Module):
    def __init__(
        self,
        low_level_channels: int,
        high_level_channels: int,
        output_channels: int,
        c_min: float = 0.1,
        c_max: float = 10.0,
        curv_reg_weight: float = 1e-3,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.c_min = c_min
        self.c_max = c_max
        self.curv_reg_weight = curv_reg_weight
        self.eps = eps

        init_theta_c = math.log(math.expm1(1.0))
        self.theta_c = nn.Parameter(torch.tensor(init_theta_c, dtype=torch.float32))

        self.W_low = nn.Conv2d(
            low_level_channels, output_channels, kernel_size=1, bias=False
        )
        self.W_high = nn.Conv2d(
            high_level_channels, output_channels, kernel_size=1, bias=False
        )

        self.tau_low = nn.Parameter(torch.ones(1))
        self.tau_high = nn.Parameter(torch.ones(1))

        self.g_low = nn.Conv2d(output_channels, 1, kernel_size=1, bias=True)
        self.g_high = nn.Conv2d(output_channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.g_low.weight)
        nn.init.zeros_(self.g_low.bias)
        nn.init.zeros_(self.g_high.weight)
        nn.init.zeros_(self.g_high.bias)

        self.W_euc = nn.Conv2d(
            output_channels, output_channels, kernel_size=1, bias=False
        )
        self.W_o_hyp = nn.Conv2d(
            output_channels, output_channels, kernel_size=1, bias=False
        )
        self.W_o_euc = nn.Conv2d(
            output_channels, output_channels, kernel_size=1, bias=False
        )

        self.gn_hyp = nn.GroupNorm(1, output_channels)
        self.gn_euc = nn.GroupNorm(1, output_channels)

        # --- BUG FIX #2: Add explicit zero-init final GroupNorm ---
        # This ensures strict identity-anchoring at init regardless of
        # gn_hyp/gn_euc initialization, mirroring AWTTFModule's robust design.
        self.final_norm = nn.GroupNorm(1, output_channels)

        init_phi = math.log(0.3 / 0.7)
        self.phi = nn.Parameter(torch.tensor(init_phi, dtype=torch.float32))

        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for conv in [self.W_low, self.W_high, self.W_euc, self.W_o_hyp, self.W_o_euc]:
            nn.init.xavier_uniform_(conv.weight)

        nn.init.zeros_(self.gn_hyp.weight)
        nn.init.zeros_(self.gn_hyp.bias)
        nn.init.zeros_(self.gn_euc.weight)
        nn.init.zeros_(self.gn_euc.bias)

        # Zero-init the new final_norm to guarantee proc_feat_b2 = 0 at epoch 0
        nn.init.zeros_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.theta_c) + self.eps

    def curvature_reg_loss(self) -> torch.Tensor:
        c = self.curvature
        return self.curv_reg_weight * (
            torch.relu(c - self.c_max).pow(2) + torch.relu(self.c_min - c).pow(2)
        )

    def _expmap0(self, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        v_norm = v.norm(dim=1, keepdim=True)
        sqrt_c = torch.sqrt(c)
        tanh_arg = torch.clamp(sqrt_c * v_norm, max=15.0)
        tanh_val = torch.tanh(tanh_arg)
        denom = sqrt_c * (v_norm + self.eps)
        x = tanh_val * v / denom
        return x

    def _logmap0(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_norm = x.norm(dim=1, keepdim=True)
        sqrt_c = torch.sqrt(c)
        artanh_arg = torch.clamp(sqrt_c * x_norm, min=0.0, max=1.0 - self.eps)
        artanh_val = torch.atanh(artanh_arg)
        denom = sqrt_c * (x_norm + self.eps)
        u = artanh_val * x / denom
        return u

    def forward(
        self,
        low_level_feat: torch.Tensor,
        high_level_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        high_level_up = F.interpolate(
            high_level_feat,
            size=low_level_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        c = self.curvature

        v_low = self.tau_low * self.W_low(low_level_feat)
        v_high = self.tau_high * self.W_high(high_level_up)

        x_low = self._expmap0(v_low, c)
        x_high = self._expmap0(v_high, c)

        x_low_norm_sq = (x_low * x_low).sum(dim=1, keepdim=True)
        x_high_norm_sq = (x_high * x_high).sum(dim=1, keepdim=True)

        conformal_low = 1.0 - c * x_low_norm_sq
        conformal_high = 1.0 - c * x_high_norm_sq

        g_low_val = torch.sigmoid(self.g_low(x_low))
        g_high_val = torch.sigmoid(self.g_high(x_high))

        # ---- Step 6: Einstein Mid-point Fusion (Exact Closed-Form) ----
        # --- BUG FIX #1: Exact Einstein Midpoint via Hyperboloid Projection ---
        # The previous code applied the Klein model formula directly to Poincaré
        # coordinates. We correctly convert Poincaré to Klein, compute the
        # hyperboloid centroid (Einstein midpoint) using Lorentz factors, and
        # project it back to the Poincaré ball via stereographic projection.

        # 1. Poincaré -> Klein coordinates: x_K = 2 * x_P / (1 + c * ||x_P||^2)
        xK_low = 2.0 * x_low / (1.0 + c * x_low_norm_sq).clamp_min(self.eps)
        xK_high = 2.0 * x_high / (1.0 + c * x_high_norm_sq).clamp_min(self.eps)

        # 2. Lorentz factors for Klein coordinates: gamma = 1 / sqrt(1 - c * ||x_K||^2)
        gamma_K_low = 1.0 / torch.sqrt(
            (1.0 - c * (xK_low * xK_low).sum(dim=1, keepdim=True)).clamp_min(self.eps)
        )
        gamma_K_high = 1.0 / torch.sqrt(
            (1.0 - c * (xK_high * xK_high).sum(dim=1, keepdim=True)).clamp_min(self.eps)
        )

        # 3. Weighted sum of hyperboloid embeddings
        w_low_K = g_low_val * gamma_K_low
        w_high_K = g_high_val * gamma_K_high

        sum_gamma_xK = w_low_K * xK_low + w_high_K * xK_high
        sum_gamma = w_low_K + w_high_K

        # 4. Lorentz norm of the sum
        norm_sq = sum_gamma * sum_gamma - c * (sum_gamma_xK * sum_gamma_xK).sum(
            dim=1, keepdim=True
        )
        norm = torch.sqrt(norm_sq.clamp_min(self.eps))

        # 5. Project back to Poincaré ball: x_P = spatial / (time + norm)
        m_H = sum_gamma_xK / (sum_gamma + norm + self.eps)

        # ---- Step 7: Logarithmic Map back to Euclidean Space ----
        u_hyp = self._logmap0(m_H, c)

        # ---- Step 8: Dual-Path Euclidean Residual ----
        u_euc = self.W_euc(v_low + v_high)

        # ---- Step 9: Dual-Path Coupling & Output ----
        F_hyp = self.gn_hyp(self.W_o_hyp(u_hyp))
        F_euc = self.gn_euc(self.W_o_euc(u_euc))

        phi_clamped = torch.clamp(self.phi, min=-3.0, max=3.0)
        alpha = torch.sigmoid(phi_clamped)

        # --- BUG FIX #2: Apply final_norm to ensure zero output at init ---
        # This decouples identity-anchoring from gn_hyp/gn_euc initialization
        # and ensures strict identity at epoch 0.
        F_out = self.act(self.final_norm(alpha * F_hyp + (1.0 - alpha) * F_euc))

        # ---- Step 10: Auxiliary Conformal Confidence Map Export ----
        C_conf = conformal_low.clamp(0.0, 1.0)

        return F_out, C_conf


# ===========================================================================
#  Sparse Exact Patch-Attention (Query / PixelShuffle)
#  Patchifies query to exact tokens, computes exact softmax, reconstructs
#  spatial.  Provides discrete, exact macro-semantic global context.
# ===========================================================================
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
        # FIX 3: Enforce channel equality for residual addition
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

        # FIX 2: Initialize to 1e-2 to prevent one-step gradient freeze
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


# ===========================================================================
#  Query-Enhanced Side-ViT Classifier (Dual-Branch Combined Edition)
#
#  Combines two independently developed branches into a single unified model:
#
#    Branch 1 (AWT-TF + LinearRFF):
#      - AWTTFModule: Adaptive Wavelet-Tucker Fusion for multi-scale
#        feature alignment in the frequency domain with higher-order
#        tensor decomposition.
#      - LinearRFFCrossAttention: Dense linearized cross-attention via
#        Random Fourier Features for O(N) continuous global context.
#      - IdentityInitAffine: Simple per-channel affine (no conf modulation).
#      - Config: cfg.network.vit1_feature_branch
#
#    Branch 2 (LC-HPHF v2 + Spatial Cross-Attention):
#      - LCHPHFv2: Learnable-Curvature Dual-Path Hyperbolic Poincaré
#        Hierarchical Fusion with conformal confidence map export.
#      - SpatialCrossAttention: Sparse exact patch-attention with
#        PixelShuffle reconstruction for discrete macro-semantic context.
#      - IdentityInitAffine: Geometric confidence-modulated affine.
#      - Config: cfg.network.vit2_feature_branch
#
#  System-Level Architecture:
#    ONE frozen ConvNeXtV2 backbone (shared feature extractor)
#      ├── Branch 1 pipeline (fully independent):
#      │     AWT-TF → LinearRFF Cross-Attn → Refine → Norm → Side-ViT
#      │     → vit_out_1 (B, hidden_size)
#      └── Branch 2 pipeline (fully independent):
#            LC-HPHF v2 → Spatial Cross-Attn → Refine → Norm(conf) → Side-ViT
#            → vit_out_2 (B, hidden_size)
#
#    ONE shared Side-ViT (called once per branch, processes each branch's
#    vit_input independently with the same frozen K/V memory banks)
#
#    ONE GatedFeatureFusionHead (dual-input):
#      Fuses vit_out_1 and vit_out_2 via per-branch self-gating and
#      dynamic cross-branch softmax weighting → final classification logits
#
#  The two branches are FULLY INDEPENDENT from backbone feature extraction
#  through the Side-ViT output. They only merge at the classifier_head,
#  which adaptively combines their complementary representations:
#    - Branch 1: frequency-domain wavelet-tucker features + dense RFF context
#    - Branch 2: hyperbolic geometric features + sparse exact patch context
# ===========================================================================
class GeoSpecClassifier(nn.Module):
    SIDE_VIT_INPUT_CHANNELS: int = 3

    def __init__(self, side_vit_b1: nn.Module, side_vit_b2: nn.Module, cfg: Any):
        super().__init__()

        # ================================================================
        # Configuration validation
        # ================================================================
        raw_trainable = getattr(cfg.network, "backbone_trainable_layers", []) or []
        for i in raw_trainable:
            assert 0 <= int(i) <= 4, (
                f"backbone_trainable_layers values must be in [0, 4] "
                f"(0=stem, 1-4=stages.0-3), got {i}."
            )
        backbone_trainable_layers = [int(i) for i in raw_trainable]

        # FIX 4: Removed first redundant validation block for vit1_feature_branch

        # ---- Branch 1 feature stage selection ----
        self.vit1_feature_branch = sorted(
            [int(i) for i in cfg.network.vit1_feature_branch]
        )
        for idx in self.vit1_feature_branch:
            assert 0 <= idx <= 3, (
                f"Branch 1 index must be in [0, 3] (got {idx}). "
                f"ConvNeXtV2-Tiny has 4 feature stages (0-indexed 0-3)."
            )
        assert len(self.vit1_feature_branch) == 2, (
            f"vit1_feature_branch must have exactly 2 features, "
            f"got {len(self.vit1_feature_branch)}."
        )
        assert self.vit1_feature_branch[0] != self.vit1_feature_branch[1], (
            f"vit1_feature_branch has duplicate indices "
            f"{self.vit1_feature_branch}. Two-element branches must use "
            f"distinct stages."
        )

        # FIX 4: Removed first redundant validation block for vit2_feature_branch

        # ---- Branch 2 feature stage selection ----
        raw_branch2_cfg = getattr(
            cfg.network,
            "vit2_feature_branch",
            getattr(cfg.network, "vit_feature_branch", []),
        )
        self.vit2_feature_branch = sorted([int(i) for i in raw_branch2_cfg])
        for idx in self.vit2_feature_branch:
            assert 0 <= idx <= 3, (
                f"Branch 2 index must be in [0, 3] (got {idx}). "
                f"ConvNeXtV2-Tiny has 4 feature stages (0-indexed 0-3)."
            )
        assert len(self.vit2_feature_branch) == 2, (
            f"vit2_feature_branch must have exactly 2 features, "
            f"got {len(self.vit2_feature_branch)}."
        )
        assert self.vit2_feature_branch[0] != self.vit2_feature_branch[1], (
            f"vit2_feature_branch has duplicate indices "
            f"{self.vit2_feature_branch}. Two-element branches must use "
            f"distinct stages."
        )

        self.cfg = cfg
        self.num_classes = cfg.dataset.num_classes
        image_channels = cfg.dataset.image_channel_num
        side_input_size = cfg.network.side_input_size

        assert side_input_size % 16 == 0, (
            f"side_input_size must be divisible by 16 (ViT patch size), "
            f"got {side_input_size}."
        )

        self._image_channels = image_channels
        self._side_vit_ch = self.SIDE_VIT_INPUT_CHANNELS
        self._side_input_size = side_input_size

        # ================================================================
        # ONE shared CNN backbone (frozen ConvNeXtV2)
        # ================================================================
        self.cnn_backbone = MultiScaleConvNeXtV2Backbone(
            model_name="convnextv2_tiny",
            pretrained=True,
            in_chans=image_channels,
            backbone_trainable_layers=backbone_trainable_layers,
        )

        feat_dims = self.cnn_backbone.channels  # Outputs [96, 192, 384, 768]
        proj_channels = 64
        side_vit_ch = self.SIDE_VIT_INPUT_CHANNELS

        # ================================================================
        # FIX 1: Intermediate setup code (Restored missing instantiations)
        # ================================================================
        f1, f2 = self.vit1_feature_branch
        f3, f4 = self.vit2_feature_branch

        # Branch 1 Geometric/Spectral Fusion
        self.gate_b1 = AWTTFModule(
            low_level_channels=feat_dims[f1],
            high_level_channels=feat_dims[f2],
            output_channels=proj_channels,
        )
        # Branch 2 Hyperbolic Fusion
        self.gate_b2 = LCHPHFv2(
            low_level_channels=feat_dims[f3],
            high_level_channels=feat_dims[f4],
            output_channels=proj_channels,
        )

        # Cross-Attention Modules (query_channels must equal output_channels)
        self.spatial_fusion_b1 = LinearRFFCrossAttention(
            query_channels=side_vit_ch,
            context_channels=proj_channels,
            output_channels=side_vit_ch,
        )
        self.spatial_fusion_b2 = SpatialCrossAttention(
            query_channels=side_vit_ch,
            context_channels=proj_channels,
            output_channels=side_vit_ch,
        )

        # Refinement & DropPath
        self.refine_b1 = self._make_refinement(side_vit_ch)
        self.refine_b2 = self._make_refinement(side_vit_ch)

        drop_path_rate = getattr(cfg.network, "drop_path_rate", 0.0)
        self.drop_path_b1 = DropPath(drop_path_rate)
        self.drop_path_b2 = DropPath(drop_path_rate)

        # Side-ViT Input Normalization
        self.sv_input_norm_b1 = IdentityInitAffine(side_vit_ch)
        self.sv_input_norm_b2 = IdentityInitAffine(side_vit_ch)

        # Context Projection (handles arbitrary input channels safely)
        if self._image_channels != self._side_vit_ch and not (
            self._image_channels == 1 and self._side_vit_ch == 3
        ):
            self._context_proj = nn.Conv2d(
                self._image_channels, self._side_vit_ch, kernel_size=1, bias=False
            )

        # ================================================================
        # FIX 6: Corrected documentation comment
        # TWO independent Side-ViTs (BLACK BOX — FPT+ core)
        # ================================================================
        self.side_vit_b1 = side_vit_b1
        self.side_vit_b2 = side_vit_b2

        assert (
            self.side_vit_b1.side_encoder.hidden_size
            == self.side_vit_b2.side_encoder.hidden_size
        ), (
            f"Both side-ViTs must have the same hidden_size, but got "
            f"{self.side_vit_b1.side_encoder.hidden_size} and "
            f"{self.side_vit_b2.side_encoder.hidden_size}."
        )

        side_vit_output_hidden_size = self.side_vit_b1.side_encoder.hidden_size

        # ================================================================
        # ONE classifier head (GatedFeatureFusionHead — Dual-Branch)
        # ================================================================
        self.classifier_head = GatedFeatureFusionHead(
            hidden_size=side_vit_output_hidden_size,
            num_classes=self.num_classes,
            reduction=4,
            dropout=0.1,
        )

    @staticmethod
    def _init_conv_weights(module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _make_refinement(channels: int) -> nn.Module:
        """Two-conv refinement block with zero-init residual.
        Conv1 -> GroupNorm -> ReLU -> Conv2 -> GroupNorm(zero-init)
        """
        refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        nn.init.kaiming_normal_(refine[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(refine[3].weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(refine[-1].weight)
        nn.init.zeros_(refine[-1].bias)
        return refine

    def _expand_context(self, x: torch.Tensor) -> torch.Tensor:
        if self._image_channels == self._side_vit_ch:
            return x
        if self._image_channels == 1 and self._side_vit_ch == 3:
            return x.repeat(1, 3, 1, 1)
        return self._context_proj(x)

    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """
        Return auxiliary regularization losses for the training loop.

        Currently includes:
        - ``curvature_reg``: Soft log-barrier regularization on the
          LC-HPHF v2 learnable curvature parameter (Branch 2's gate),
          keeping c within [c_min, c_max].

        Usage in training loop::

            losses = model.get_auxiliary_losses()
            total_loss = main_loss + sum(losses.values())
        """
        losses: Dict[str, torch.Tensor] = {}
        if isinstance(self.gate_b2, LCHPHFv2):
            losses["curvature_reg"] = self.gate_b2.curvature_reg_loss()
        return losses

    def forward(
        self,
        x: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        # ============================================================
        # Step 1: ONE shared CNN backbone at NATIVE input resolution
        # Both branches consume the same feature maps.
        # ============================================================
        features = self.cnn_backbone(x)

        # ============================================================
        # Step 2: Shared context preparation (computed ONCE)
        # Resize the IMAGE to side_input_size BEFORE cross-attention.
        # Both branches use the same resized context.
        # ============================================================
        target_size = (self._side_input_size, self._side_input_size)
        context = self._expand_context(x)
        context = F.interpolate(
            context, size=target_size, mode="bilinear", align_corners=False
        )

        # ============================================================
        # Branch 1: AWT-TF + LinearRFF Cross-Attention
        # (Fully independent pipeline)
        # ============================================================

        # ---- Step B1-A: AWT-TF feature fusion ----
        # AWTTFModule returns a single fused tensor (no confidence map).
        proc_feat_b1 = self.gate_b1(*[features[f] for f in self.vit1_feature_branch])

        # ---- Step B1-B: Dense Linearized RFF Cross-Attention ----
        vit_input_b1 = self.spatial_fusion_b1(context, proc_feat_b1)

        # ---- Step B1-C: Residual refinement (zero-init, at side_input_size) ----
        vit_input_b1 = vit_input_b1 + self.drop_path_b1(self.refine_b1(vit_input_b1))

        # ---- Step B1-D: Side-ViT input stabilization (no conf) ----
        # IdentityInitAffine without conf => simple per-channel affine.
        # At init: gamma=1, delta=0 => true identity.
        vit_input_b1 = self.sv_input_norm_b1(vit_input_b1)

        # ---- Step B1-E: Side-ViT forward pass ----
        # Queries the pre-loaded frozen K/V memory banks.
        vit_out_b1 = self.side_vit_b1(vit_input_b1, key_states, value_states)

        # ============================================================
        # Branch 2: LC-HPHF v2 + Spatial Cross-Attention
        # (Fully independent pipeline)
        # ============================================================

        # ---- Step B2-A: LC-HPHF v2 geometric fusion ----
        # Returns (F_out, C_conf) — fused features + conformal confidence map.
        proc_feat_b2, conf_map = self.gate_b2(
            *[features[f] for f in self.vit2_feature_branch]
        )

        # ---- Step B2-B: Sparse Exact Patch Attention Fusion ----
        vit_input_b2 = self.spatial_fusion_b2(context, proc_feat_b2)

        # ---- Step B2-C: Residual refinement (zero-init, at side_input_size) ----
        vit_input_b2 = vit_input_b2 + self.drop_path_b2(self.refine_b2(vit_input_b2))

        # ---- Step B2-D: Geometric Nervous System — IdentityInitAffine ----
        # The exported conformal confidence map C_conf spatially modulates
        # the affine layer:
        #   Affine(x) = x * (gamma + beta * C_conf) + delta
        # At init: beta=0 => identity. As training progresses, beta
        # learns to suppress noisy regions identified by the hyperbolic
        # manifold.
        if conf_map is not None:
            # Resize confidence map to match vit_input spatial dimensions
            conf_map_resized = F.interpolate(
                conf_map,
                size=vit_input_b2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            vit_input_b2 = self.sv_input_norm_b2(vit_input_b2, conf_map_resized)
        else:
            vit_input_b2 = self.sv_input_norm_b2(vit_input_b2)

        # ---- Step B2-E: Side-ViT forward pass ----
        # Same shared side_vit, different vit_input.
        # Queries the same pre-loaded frozen K/V memory banks.
        vit_out_b2 = self.side_vit_b2(vit_input_b2, key_states, value_states)

        # ============================================================
        # Step 3: ONE Gated Feature Fusion + Classification
        # The two branches' side-ViT embeddings are fused via:
        #   1. Per-branch LayerNorm + self-gating (channel selection)
        #   2. Dynamic cross-branch softmax weighting (branch importance)
        #   3. GELU + dropout + linear projection → logits
        # ============================================================
        logits = self.classifier_head(vit_out_b1, vit_out_b2)
        return logits
