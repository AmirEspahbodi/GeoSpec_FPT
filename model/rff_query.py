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
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor


# ===========================================================================
#  Identity-initialized per-channel affine.
#  y = gamma * x + beta, with gamma=1, beta=0 at init => TRUE identity.
#  Used to stabilize side-ViT input and prevent K/V distribution drift.
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
#  Gated Attention Module (multi-scale feature fusion)
# ===========================================================================
class GatedAttentionModule(nn.Module):
    def __init__(
        self, low_level_channels: int, high_level_channels: int, output_channels: int
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
#  Branch 1: Dense Linearized Cross-Attention (LinearRFF)
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
#  Branch 2: Sparse Exact Patch-Attention (Query / PixelShuffle)
#  Patchifies query to exact tokens, computes exact softmax, reconstructs spatial.
#  Provides discrete, exact macro-semantic global context.
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
            query_channels, query_channels * (patch_size**2), kernel_size=1, bias=False
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=patch_size)

        # 4. Refinement
        self.proj_conv = nn.Conv2d(
            query_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.proj_norm = nn.GroupNorm(1, output_channels)
        self.proj_drop = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.zeros(1))
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
        recon = self.reconstruct_proj(out)  # (B, C_q * patch_size^2, q_h, q_w)
        recon = self.pixel_shuffle(recon)  # (B, C_q, H, W)

        # (6) Zero-init residual refinement
        enhancement = self.proj_conv(recon)
        enhancement = self.proj_norm(enhancement)
        enhancement = self.proj_drop(enhancement)

        fused = query_feat + self.gamma * enhancement
        return fused


# ===========================================================================
#  Multi-Fidelity Query-Enhanced Side-ViT Classifier
#  Combines Branch 1 (LinearRFF) and Branch 2 (SpatialCrossAttention)
# ===========================================================================
class QueryEnhancedSideViTClassifier_rff_query(nn.Module):
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

        # ---------------- Branch 1 Prep ----------------
        if len(self.vit1_feature_branch) == 2:
            self.gate1 = GatedAttentionModule(*branch1_dim, proj_channels)
        else:
            self.proj_sv1 = nn.Sequential(
                nn.Conv2d(branch1_dim[0], proj_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, proj_channels),
                nn.ReLU(inplace=True),
            )
            self._init_conv_weights(self.proj_sv1)

        # ---------------- Branch 2 Prep ----------------
        if len(self.vit2_feature_branch) == 2:
            self.gate2 = GatedAttentionModule(*branch2_dim, proj_channels)
        else:
            self.proj_sv2 = nn.Sequential(
                nn.Conv2d(branch2_dim[0], proj_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, proj_channels),
                nn.ReLU(inplace=True),
            )
            self._init_conv_weights(self.proj_sv2)

        # ---------------- Multi-Fidelity Cross-Attention Fusions ---------
        # Branch 1: Dense Linearized RFF Attention
        self.spatial_fusion1 = LinearRFFCrossAttention(
            query_channels=side_vit_ch,
            context_channels=proj_channels,
            output_channels=side_vit_ch,
            num_features=64,
            dropout=0.0,
        )
        # Branch 2: Sparse Exact Patch Attention
        self.spatial_fusion2 = SpatialCrossAttention(
            query_channels=side_vit_ch,
            context_channels=proj_channels,
            output_channels=side_vit_ch,
            patch_size=16,
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

        # ---------------- Side-ViT input stabilization ----------------
        # Applied uniformly to both branches to prevent K/V drift
        self.sv_input_norm1 = IdentityInitAffine(side_vit_ch)
        self.sv_input_norm2 = IdentityInitAffine(side_vit_ch)

        # ---------------- Channel adapter (image -> side_vit channels) ----
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
        # Conv1: followed by ReLU -> "relu" gain is correct
        nn.init.kaiming_normal_(refine[0].weight, mode="fan_out", nonlinearity="relu")
        # Conv2: followed by GroupNorm (linear path) -> "linear" gain is correct
        nn.init.kaiming_normal_(refine[3].weight, mode="fan_out", nonlinearity="linear")
        # Zero-init last norm -> refine(x) == 0 at init -> identity residual.
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
        # 1. CNN backbone at NATIVE input resolution (preserves feature quality)
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

        # 5. Multi-Fidelity Cross-Attention
        # Branch 1: Dense Linearized RFF Attention
        vit_input1 = self.spatial_fusion1(context, proc_feat1)
        # Branch 2: Sparse Exact Patch Attention
        vit_input2 = self.spatial_fusion2(context, proc_feat2)

        # 6. Residual refinement (operates at side_input_size)
        vit_input1 = vit_input1 + self.drop_path1(self.refine1(vit_input1))
        vit_input2 = vit_input2 + self.drop_path2(self.refine2(vit_input2))

        # 6b. Apply input stabilization (identity at init, learns to compensate
        #     for cross-attention/refinement drift to keep side-ViT input stable).
        vit_input1 = self.sv_input_norm1(vit_input1)
        vit_input2 = self.sv_input_norm2(vit_input2)

        # 7. Side-ViTs (BLACK BOX - FPT+ core, untouched)
        vit_out1 = self.side_vit1(vit_input1, key_states, value_states)
        vit_out2 = self.side_vit2(vit_input2, key_states, value_states)

        # 8. Gated feature fusion + classification
        combined = torch.cat([vit_out1, vit_out2], dim=1)
        logits = self.classifier_head(combined)
        return logits
