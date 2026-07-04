from typing import Any, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
#  DropPath (Stochastic Depth) Implementation
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
        random_tensor = torch.floor(random_tensor + keep_prob)  # Binarize
        return x.div(keep_prob) * random_tensor


# ===========================================================================
#  Identity-initialized per-channel affine.
#  y = gamma * x + beta, with gamma=1, beta=0 at init => TRUE identity.
#  Used specifically for Branch 1 (LinearRFF) to stabilize Side-ViT input.
# ===========================================================================
class IdentityInitAffine(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


# ===========================================================================
#  Gated Feature Fusion Head (VRAM Efficient Classifier)
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
        self.norm = nn.LayerNorm(hidden_size * 2)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, max((hidden_size * 2) // reduction, 1)),
            nn.GELU(),
            nn.Linear(max((hidden_size * 2) // reduction, 1), hidden_size * 2),
            nn.Sigmoid(),
        )

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        g = self.gate(x)
        x = x * g
        x = self.act(x)
        x = self.drop(x)
        x = self.proj(x)
        return x


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
            out_indices=(
                0,
                1,
                2,
                3,
            ),  # ConvNeXtV2 has 4 feature stages (no stem feature)
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
        """Find the stage module index from parameter name parts."""
        for i in range(len(parts) - 1):
            if parts[i] in self._STAGE_PREFIXES:
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    def _is_stem_param(self, parts: List[str]) -> bool:
        """Check if a parameter belongs to the stem module (feature 0)."""
        for part in parts:
            if part in self._STAGE_PREFIXES:
                return False
            if part in self._STEM_NAMES:
                return True
        return False

    def _is_trainable_param(self, name: str) -> bool:
        """Determine if a parameter should be trainable."""
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
        """Set frozen modules to eval, keep trainable modules in train mode."""
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
#  Gated Attention Module (multi-scale feature fusion)
# ===========================================================================
class GatedAttentionModule(nn.Module):
    def __init__(
        self,
        low_level_channels: int,
        high_level_channels: int,
        output_channels: int,
    ):
        super().__init__()
        self.high_norm = nn.GroupNorm(1, high_level_channels)
        self.attn_conv = nn.Conv2d(
            high_level_channels, low_level_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.proj_conv = nn.Conv2d(
            low_level_channels, output_channels, kernel_size=1, bias=False
        )
        self.bn = nn.GroupNorm(1, output_channels)
        self.act = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.attn_conv.weight)
        nn.init.kaiming_normal_(
            self.proj_conv.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(
        self, low_level_feat: torch.Tensor, high_level_feat: torch.Tensor
    ) -> torch.Tensor:
        high_level_feat = self.high_norm(high_level_feat)
        high_level_up = F.interpolate(
            high_level_feat,
            size=low_level_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attention_map = self.sigmoid(self.attn_conv(high_level_up))
        attended_feat = low_level_feat * attention_map
        output = self.proj_conv(attended_feat)
        output = self.bn(output)
        output = self.act(output)
        return output


# ===========================================================================
#  Branch 1: Linear Random Fourier Features (RFF) Cross-Attention
#  Provides global, linear-time context aggregation via kernel approximation.
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

        # d^{-1/4} so that (q·scale)(k·scale) = q·k / √d  (standard attn)
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

        self.gamma = nn.Parameter(torch.zeros(1))
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
        log_phi = torch.clamp(log_phi, max=0.0)
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
#  Branch 2: Parallel Axial Factorized Cross-Attention
#  Preserves structural spatial boundaries via 1D axis-factorized attention.
# ===========================================================================
class AxialFactorizedCrossAttention(nn.Module):
    def __init__(
        self,
        query_channels: int,
        context_channels: int,
        output_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
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

        self.scale = inter_channels**-0.5
        self.attn_drop = nn.Dropout(dropout)

        self.proj_conv = nn.Conv2d(
            query_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.proj_norm = nn.GroupNorm(1, output_channels)
        self.proj_drop = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for conv in [self.query_conv, self.key_conv, self.value_conv, self.proj_conv]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="linear")

    def forward(
        self, query_feat: torch.Tensor, context_feat: torch.Tensor
    ) -> torch.Tensor:
        B, C_q, H_q, W_q = query_feat.shape
        _, C_k, H_k, W_k = context_feat.shape

        q = self.query_conv(query_feat)
        k = self.key_conv(context_feat)
        v = self.value_conv(context_feat)

        # --------------------------------------------------------------
        # 1. Height-wise Cross-Attention
        # --------------------------------------------------------------
        k_h = F.interpolate(k, size=(H_k, W_q), mode="bilinear", align_corners=False)
        v_h = F.interpolate(v, size=(H_k, W_q), mode="bilinear", align_corners=False)

        q_h = q.permute(0, 3, 2, 1).reshape(B * W_q, H_q, -1).contiguous()
        k_h = k_h.permute(0, 3, 1, 2).reshape(B * W_q, -1, H_k).contiguous()
        v_h = v_h.permute(0, 3, 2, 1).reshape(B * W_q, H_k, -1).contiguous()

        attn_h = torch.bmm(q_h, k_h) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        attn_h = self.attn_drop(attn_h)

        out_h = torch.bmm(attn_h, v_h)
        out_h = out_h.reshape(B, W_q, H_q, C_q).permute(0, 3, 2, 1).contiguous()

        # --------------------------------------------------------------
        # 2. Width-wise Cross-Attention
        # --------------------------------------------------------------
        k_w = F.interpolate(k, size=(H_q, W_k), mode="bilinear", align_corners=False)
        v_w = F.interpolate(v, size=(H_q, W_k), mode="bilinear", align_corners=False)

        q_w = q.permute(0, 2, 3, 1).reshape(B * H_q, W_q, -1).contiguous()
        k_w = k_w.permute(0, 2, 1, 3).reshape(B * H_q, -1, W_k).contiguous()
        v_w = v_w.permute(0, 2, 3, 1).reshape(B * H_q, W_k, -1).contiguous()

        attn_w = torch.bmm(q_w, k_w) * self.scale
        attn_w = attn_w.softmax(dim=-1)
        attn_w = self.attn_drop(attn_w)

        out_w = torch.bmm(attn_w, v_w)
        out_w = out_w.reshape(B, H_q, W_q, C_q).permute(0, 3, 1, 2).contiguous()

        # --------------------------------------------------------------
        # 3. Aggregate parallel axial attentions
        # --------------------------------------------------------------
        out = out_h + out_w

        enhancement = self.proj_conv(out)
        enhancement = self.proj_norm(enhancement)
        enhancement = self.proj_drop(enhancement)

        fused = query_feat + self.gamma * enhancement
        return fused


# ===========================================================================
#  Hybrid Side-ViT Classifier
#  (Branch 1: LinearRFF + Branch 2: Axial)
# ===========================================================================
class QueryEnhancedSideViTClassifier_rff_axial(nn.Module):
    SIDE_VIT_INPUT_CHANNELS: int = 3

    def __init__(self, side_vit1: nn.Module, side_vit2: nn.Module, cfg: Any):
        super().__init__()

        raw_trainable = getattr(cfg.network, "backbone_trainable_layers", []) or []
        for i in raw_trainable:
            assert 0 <= int(i) <= 4, (
                f"backbone_trainable_layers values must be in [0, 4] "
                f"(0=stem, 1-4=stages.0-3), got {i}."
            )
        backbone_trainable_layers = [int(i) for i in raw_trainable]

        self.vit1_feature_branch = sorted(
            [int(i) for i in cfg.network.vit1_feature_branch]
        )
        self.vit2_feature_branch = sorted(
            [int(i) for i in cfg.network.vit2_feature_branch]
        )

        for idx in self.vit1_feature_branch + self.vit2_feature_branch:
            assert 0 <= idx <= 3, (
                f"Branch index must be in [0, 3] (got {idx}). "
                f"ConvNeXtV2-Tiny has 4 feature stages (0-indexed 0-3)."
            )
        for label, branch in [
            ("vit1_feature_branch", self.vit1_feature_branch),
            ("vit2_feature_branch", self.vit2_feature_branch),
        ]:
            assert 1 <= len(branch) <= 2, (
                f"{label} must have 1 or 2 features, got {len(branch)}."
            )
            if len(branch) == 2:
                assert branch[0] != branch[1], (
                    f"{label} has duplicate indices {branch}. "
                    f"Two-element branches must use distinct stages."
                )

        self.cfg = cfg
        self.num_classes = cfg.dataset.num_classes
        image_channels = cfg.dataset.image_channel_num
        side_input_size = cfg.network.side_input_size

        assert side_input_size % 16 == 0, (
            f"side_input_size must be divisible by 16 (ViT patch size), got {side_input_size}."
        )

        self._image_channels = image_channels
        self._side_vit_ch = self.SIDE_VIT_INPUT_CHANNELS
        self._side_input_size = side_input_size

        # ---------------- CNN backbone ----------------
        self.cnn_backbone = MultiScaleConvNeXtV2Backbone(
            model_name="convnextv2_tiny",
            pretrained=True,
            in_chans=image_channels,
            backbone_trainable_layers=backbone_trainable_layers,
        )

        feat_dims = self.cnn_backbone.channels  # Outputs [96, 192, 384, 768]
        proj_channels = 64
        side_vit_ch = self.SIDE_VIT_INPUT_CHANNELS

        branch1_dim = [feat_dims[i] for i in self.vit1_feature_branch]
        branch2_dim = [feat_dims[i] for i in self.vit2_feature_branch]

        # ---------------- Branch 1 Projections ----------------
        if len(self.vit1_feature_branch) == 2:
            self.gate1 = GatedAttentionModule(*branch1_dim, proj_channels)
        else:
            self.proj_sv1 = nn.Sequential(
                nn.Conv2d(branch1_dim[0], proj_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, proj_channels),
                nn.ReLU(inplace=True),
            )
            self._init_conv_weights(self.proj_sv1)

        # ---------------- Branch 2 Projections ----------------
        if len(self.vit2_feature_branch) == 2:
            self.gate2 = GatedAttentionModule(*branch2_dim, proj_channels)
        else:
            self.proj_sv2 = nn.Sequential(
                nn.Conv2d(branch2_dim[0], proj_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, proj_channels),
                nn.ReLU(inplace=True),
            )
            self._init_conv_weights(self.proj_sv2)

        # ---------------- Cross-attention fusions ----------------
        # Branch 1: Linear RFF Cross-Attention
        self.spatial_fusion1 = LinearRFFCrossAttention(
            query_channels=side_vit_ch,
            context_channels=proj_channels,
            output_channels=side_vit_ch,
            num_features=64,
            dropout=0.0,
        )
        # Branch 2: Axial Factorized Cross-Attention
        self.spatial_fusion2 = AxialFactorizedCrossAttention(
            query_channels=side_vit_ch,
            context_channels=proj_channels,
            output_channels=side_vit_ch,
            dropout=0.0,
        )

        # ---------------- Refinement (zero-init residual) ----------------
        self.refine1 = self._make_refinement(side_vit_ch)
        self.refine2 = self._make_refinement(side_vit_ch)

        drop_path_rate = getattr(cfg.network, "drop_path_rate", 0.1)
        if drop_path_rate > 0.0:
            self.drop_path1 = DropPath(drop_path_rate)
            self.drop_path2 = DropPath(drop_path_rate)
        else:
            self.drop_path1 = nn.Identity()
            self.drop_path2 = nn.Identity()

        # ---------------- Side-ViT input stabilization (Branch 1 Only) ----------------
        # Identity-initialized affine to protect Side-ViT 1 from Linear RFF drift
        self.sv_input_norm1 = IdentityInitAffine(side_vit_ch)

        # ---------------- Channel adapter (image -> side_vit channels) ----------------
        if image_channels != side_vit_ch and not (
            image_channels == 1 and side_vit_ch == 3
        ):
            self._context_proj = nn.Conv2d(
                image_channels, side_vit_ch, kernel_size=1, bias=False
            )
            nn.init.kaiming_normal_(
                self._context_proj.weight, mode="fan_out", nonlinearity="linear"
            )
        else:
            self._context_proj = None

        # ---------------- Side-ViTs (BLACK BOX) ----------------
        self.side_vit1 = side_vit1
        self.side_vit2 = side_vit2
        side_vit_output_hidden_size = self.side_vit1.side_encoder.hidden_size

        # ---------------- Classifier head ----------------
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
        refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        for m in refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(refine[-1].weight)
        nn.init.zeros_(refine[-1].bias)
        return refine

    def _expand_context(self, x: torch.Tensor) -> torch.Tensor:
        if self._image_channels == self._side_vit_ch:
            return x
        if self._image_channels == 1 and self._side_vit_ch == 3:
            return x.repeat(1, 3, 1, 1)
        return self._context_proj(x)

    def forward(
        self,
        x: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. CNN backbone at NATIVE input resolution
        features = self.cnn_backbone(x)

        # 2. Branch 1: process CNN features
        if len(self.vit1_feature_branch) == 2:
            proc_feat1 = self.gate1(*[features[f] for f in self.vit1_feature_branch])
        else:
            proc_feat1 = self.proj_sv1(features[self.vit1_feature_branch[0]])

        # 3. Branch 2: process CNN features
        if len(self.vit2_feature_branch) == 2:
            proc_feat2 = self.gate2(*[features[f] for f in self.vit2_feature_branch])
        else:
            proc_feat2 = self.proj_sv2(features[self.vit2_feature_branch[0]])

        # 4. Resize the IMAGE to side_input_size BEFORE cross-attention.
        target_size = (self._side_input_size, self._side_input_size)
        context = self._expand_context(x)
        context = F.interpolate(
            context, size=target_size, mode="bilinear", align_corners=False
        )

        # 5. Spatial Cross-Attention
        # Branch 1: Linear RFF Attention
        vit_input1 = self.spatial_fusion1(context, proc_feat1)
        # Branch 2: Axial Factorized Attention
        vit_input2 = self.spatial_fusion2(context, proc_feat2)

        # 6. Residual refinement
        vit_input1 = vit_input1 + self.drop_path1(self.refine1(vit_input1))
        vit_input2 = vit_input2 + self.drop_path2(self.refine2(vit_input2))

        # 6b. Apply input stabilization specifically to Branch 1
        vit_input1 = self.sv_input_norm1(vit_input1)

        # 7. Side-ViTs (BLACK BOX - FPT+ core, untouched)
        vit_out1 = self.side_vit1(vit_input1, key_states, value_states)
        vit_out2 = self.side_vit2(vit_input2, key_states, value_states)

        # 8. Gated feature fusion + classification
        combined = torch.cat([vit_out1, vit_out2], dim=1)
        logits = self.classifier_head(combined)
        return logits
