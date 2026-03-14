"""
maxxvit_enhanced.py

Enhanced DTF-AT architecture built on top of maxxvit.py.
Adds 3 architectural modifications for audio event classification:

1. HybridPreStem        - Residual CNN block placed BEFORE the TF-decoupled Stem
                          Extracts low-level acoustic features (edges, harmonics)
                          before specialized time-frequency processing

2. MultiHeadAttnPool    - Learnable multi-head attention pooling replaces avg pool
                          Multiple heads simultaneously attend to different
                          discriminative time-frequency regions — critical for
                          multi-label audio with overlapping events

3. MultiScaleFusionHead - Hierarchical cross-attention fusion across all 4 encoder stages
                          Captures both low-level acoustic patterns (early stages)
                          and high-level semantic features (later stages)

Variants supported:
    attn_pool    : Enhancement 2 only
    hybrid_stem  : Enhancement 1 only
    multiscale   : Enhancement 3 only
    all_combined : All 3 together
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, List, Optional, Union

from audio_model_timm.models.maxxvit import (
    MaxxVit,
    MaxxVitCfg,
    MaxxVitConvCfg,
    MaxxVitTransformerCfg,
    checkpoint_filter_fn,
)
from audio_model_timm.layers import (
    get_act_layer,
    get_norm_layer,
    get_norm_act_layer,
    NormMlpClassifierHead,
    ClassifierHead,
)

__all__ = [
    'MaxxVitEnhanced',
    'MaxxVitCfg',
    'MaxxVitConvCfg',
    'MaxxVitTransformerCfg',
    'checkpoint_filter_fn',
]


# ==============================================================================
# Enhancement 1: Hybrid Pre-Stem
# ==============================================================================

class HybridPreStem(nn.Module):
    """
    Residual CNN block placed BEFORE the existing TF-decoupled Stem.

    Motivation (Proposal §4):
        The current stem directly applies time-frequency decoupled convolutions.
        Standard CNNs are excellent at extracting local low-level features
        (spectral edges, harmonic patterns, onset textures) from spectrograms.
        Adding a residual block first lets the model build richer input
        representations before the specialized TF processing begins.

    Architecture:
        x → Conv(in_chs→hidden, 3x3) → BN+GELU
          → Conv(hidden→in_chs, 3x3) → BN
          → x + block(x)   [residual keeps original spectrogram info]

    Input / Output shape: [B, in_chs, H, W]  (channel count preserved)
    The existing Stem receives the same interface as before.
    """

    def __init__(
        self,
        in_chs: int,
        hidden_chs: int = 16,
        norm_layer: str = 'batchnorm2d',
        act_layer: str = 'gelu_tanh',
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        norm_act = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)

        # Two conv layers in the residual path
        self.conv1 = nn.Conv2d(in_chs, hidden_chs, kernel_size=3, padding=1, bias=False)
        self.norm_act1 = norm_act(hidden_chs)                  # BN + GELU

        self.conv2 = nn.Conv2d(hidden_chs, in_chs, kernel_size=3, padding=1, bias=False)
        self.norm2 = get_norm_layer(norm_layer)(in_chs)        # BN only (no activation)
                                                                # → cleaner residual addition

        # Initialize conservatively so training starts close to identity
        nn.init.zeros_(self.conv2.weight)

        print(f"  HybridPreStem: {in_chs}→{hidden_chs}→{in_chs} (residual, 3x3 convs)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm_act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual        # residual preserves raw spectrogram info


# ==============================================================================
# Enhancement 2: Multi-Head Attention Pooling
# ==============================================================================

class MultiHeadAttentionPool(nn.Module):
    """
    Replaces simple adaptive average pooling with learnable multi-head attention.

    Motivation (Proposal §2):
        Average pooling treats all spatial (time-frequency) positions equally.
        AudioSet clips often contain 2–3 co-occurring sound events at different
        time-frequency regions. Multi-head attention lets each head independently
        focus on a different discriminative region, naturally handling
        overlapping events in multi-label classification.

    Mechanism:
        - Flatten spatial tokens: [B, C, H, W] → [B, N, C]  (N = H×W)
        - Learnable query [num_heads, head_dim] attends over all N tokens
        - Each head specialises on a different time-frequency region
        - Merge heads → [B, C]

    Input:  [B, C, H, W]   (NCHW, output of final encoder stage)
    Output: [B, C]
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # One learnable query per head
        self.query = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.key_proj  = nn.Linear(dim, dim)
        self.val_proj  = nn.Linear(dim, dim)
        self.out_proj  = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        # Pre-attention layer norm (channels-last)
        self.norm = nn.LayerNorm(dim)

        print(f"  MultiHeadAttentionPool: dim={dim}, num_heads={num_heads}, "
              f"head_dim={self.head_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # [B, C, H, W] → [B, N, C]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        x_flat = self.norm(x_flat)

        # Keys and values: [B, N, C]
        k = self.key_proj(x_flat)
        v = self.val_proj(x_flat)

        # Reshape to multi-head: [B, num_heads, N, head_dim]
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Expand query to batch: [B, num_heads, 1, head_dim]
        q = self.query.expand(B, -1, -1, -1)

        # Scaled dot-product attention: [B, num_heads, 1, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Attended output: [B, num_heads, 1, head_dim] → [B, C]
        out = (attn @ v)                             # [B, num_heads, 1, head_dim]
        out = out.squeeze(2)                         # [B, num_heads, head_dim]
        out = out.transpose(1, 2).reshape(B, C)      # [B, C]
        out = self.out_proj(out)                     # [B, C]

        return out


# ==============================================================================
# Enhancement 3: Multi-Scale Feature Fusion Head
# ==============================================================================

class MultiScaleFusionHead(nn.Module):
    """
    Hierarchical cross-attention fusion across all 4 encoder stages.

    Motivation (Proposal §1):
        The baseline only uses final-layer (stage-4) features for classification.
        Audio events have multi-scale structure: early stages capture fine-grained
        temporal/spectral detail, later stages capture semantic patterns.
        This head aggregates all 4 stages via cross-attention, letting the
        model decide which stages are most informative per prediction.

    Architecture:
        For each stage i  (dims: 96, 192, 384, 768):
            pool_i(stage_i) → project_i → [B, fusion_dim]     (per-stage token)

        Stack 4 tokens: [B, 4, fusion_dim]

        Cross-attention (self-attention over stage tokens):
            All stages attend to each other
            → deeper stages can integrate shallower stage context
            → residual connection

        Learnable weighted fusion → [B, fusion_dim]
        → MLP head → [B, num_classes]

    When use_attn_pool=True (all_combined variant):
        MultiHeadAttentionPool replaces AdaptiveAvgPool2d for each stage
        → richer per-stage summaries

    Args:
        stage_dims      : channel dims per stage, e.g. (96, 192, 384, 768)
        fusion_dim      : common projection dimension
        num_heads       : heads for cross-attention
        num_classes     : output classes (527)
        drop_rate       : dropout
        head_hidden_size: hidden dim in classifier MLP (0 = linear only)
        use_attn_pool   : use MultiHeadAttentionPool instead of avg pool per stage
    """

    def __init__(
        self,
        stage_dims: Tuple[int, ...] = (96, 192, 384, 768),
        fusion_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 527,
        drop_rate: float = 0.0,
        head_hidden_size: int = 768,
        use_attn_pool: bool = False,
    ):
        super().__init__()
        self.num_stages    = len(stage_dims)
        self.fusion_dim    = fusion_dim
        self.use_attn_pool = use_attn_pool

        # ---- Per-stage pooling & projection ----
        if use_attn_pool:
            # Attention-based pooling per stage
            self.stage_pools = nn.ModuleList([
                MultiHeadAttentionPool(dim=d, num_heads=8, dropout=drop_rate)
                for d in stage_dims
            ])
            # Projection after pool: [B, d] → [B, fusion_dim]
            self.stage_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.GELU(),
                )
                for d in stage_dims
            ])
        else:
            # Standard avg pool: [B, d, H, W] → [B, d] → [B, fusion_dim]
            self.stage_pools = None
            self.stage_projections = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(d, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.GELU(),
                )
                for d in stage_dims
            ])

        # ---- Cross-attention over stage tokens ----
        # All stages attend to each other (self-attention over 4 tokens)
        self.cross_attn_norm = nn.LayerNorm(fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=drop_rate,
            batch_first=True,
        )

        # ---- Learnable stage importance weights ----
        self.fusion_weights = nn.Parameter(
            torch.ones(self.num_stages) / self.num_stages
        )

        # ---- Classifier head ----
        self.head_norm = nn.LayerNorm(fusion_dim)
        self.head_drop = nn.Dropout(drop_rate)

        if head_hidden_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(fusion_dim, head_hidden_size),
                nn.Tanh(),
            )
            self.classifier = nn.Linear(head_hidden_size, num_classes)
        else:
            self.pre_logits = nn.Identity()
            self.classifier = nn.Linear(fusion_dim, num_classes)

        print(f"  MultiScaleFusionHead: stage_dims={stage_dims}, "
              f"fusion_dim={fusion_dim}, use_attn_pool={use_attn_pool}")

    def forward(self, stage_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            stage_features: list of 4 tensors [B, C_i, H_i, W_i]
        Returns:
            logits: [B, num_classes]
        """
        # Pool + project each stage → [B, fusion_dim]
        if self.use_attn_pool:
            projected = [
                proj(pool(feat))
                for proj, pool, feat in zip(
                    self.stage_projections, self.stage_pools, stage_features
                )
            ]
        else:
            projected = [
                proj(feat)
                for proj, feat in zip(self.stage_projections, stage_features)
            ]

        # Stack: [B, num_stages, fusion_dim]
        stacked = torch.stack(projected, dim=1)

        # Cross-attention: all stage tokens attend to each other
        # This lets deeper stages selectively pull context from shallower ones
        normed   = self.cross_attn_norm(stacked)
        attn_out, _ = self.cross_attn(normed, normed, normed)
        stacked  = stacked + attn_out                  # residual

        # Softmax-weighted fusion → [B, fusion_dim]
        weights  = torch.softmax(self.fusion_weights, dim=0)          # [num_stages]
        fused    = (stacked * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # [B, fusion_dim]

        # Classify
        fused    = self.head_norm(fused)
        fused    = self.head_drop(fused)
        fused    = self.pre_logits(fused)
        logits   = self.classifier(fused)

        return logits


# ==============================================================================
# MaxxVitEnhanced — drop-in replacement for MaxxVit
# ==============================================================================

class MaxxVitEnhanced(MaxxVit):
    """
    Enhanced DTF-AT model.

    Subclasses MaxxVit and adds 3 optional enhancements controlled by flags:
        use_hybrid_stem  : prepend HybridPreStem before TF Stem
        use_attn_pool    : replace avg pool with MultiHeadAttentionPool in head
        use_multiscale   : replace head with MultiScaleFusionHead (all 4 stages)

    The base MaxxVit is built first via super().__init__() so that pretrained
    ImageNet weights can still be loaded for all shared components. New components
    (HybridPreStem, AttentionPool, MultiScaleFusionHead) initialise from scratch.
    """

    def __init__(
        self,
        cfg: MaxxVitCfg,
        img_size: Union[int, Tuple[int, int]] = 224,
        in_chans: int = 1,
        num_classes: int = 527,
        global_pool: str = 'avg',
        drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        window_size_list=None,
        window_size_time_freq=None,
        feat_map_size_list=None,
        use_attn_pool: bool = False,
        use_hybrid_stem: bool = False,
        use_multiscale: bool = False,
        **kwargs,
    ):
        # Build the full base model first (enables pretrained weight loading)
        super().__init__(
            cfg=cfg,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            window_size_list=window_size_list,
            window_size_time_freq=window_size_time_freq,
            feat_map_size_list=feat_map_size_list,
            **kwargs,
        )

        self.use_attn_pool   = use_attn_pool
        self.use_hybrid_stem = use_hybrid_stem
        self.use_multiscale  = use_multiscale

        # ------------------------------------------------------------------
        # Enhancement 1: Hybrid Pre-Stem
        # ------------------------------------------------------------------
        if use_hybrid_stem:
            self.hybrid_pre_stem = HybridPreStem(
                in_chs=in_chans,
                hidden_chs=16,
                norm_layer=cfg.conv_cfg.norm_layer,
                act_layer=cfg.conv_cfg.act_layer,
                norm_eps=cfg.conv_cfg.norm_eps if cfg.conv_cfg.norm_eps else 1e-5,
            )

        # ------------------------------------------------------------------
        # Enhancement 2: Multi-Head Attention Pooling
        # Only used when multiscale is NOT active (multiscale has its own pooling)
        # When all_combined: attn pool is used INSIDE MultiScaleFusionHead
        # ------------------------------------------------------------------
        if use_attn_pool and not use_multiscale:
            self.attn_pool = MultiHeadAttentionPool(
                dim=self.num_features,   # 768
                num_heads=8,
                dropout=drop_rate,
            )
            # Build a custom classification head that uses attn_pool instead of avg pool
            # (replaces NormMlpClassifierHead's internal avg pool)
            head_hidden_size = cfg.head_hidden_size  # 768
            self.attn_head_norm = nn.LayerNorm(self.num_features)
            if head_hidden_size:
                self.attn_head_mlp = nn.Sequential(
                    nn.Linear(self.num_features, head_hidden_size),
                    nn.Tanh(),
                )
                self.attn_head_fc = nn.Linear(head_hidden_size, num_classes)
            else:
                self.attn_head_mlp = nn.Identity()
                self.attn_head_fc  = nn.Linear(self.num_features, num_classes)
            # Disable original head (free memory, avoid unused parameters)
            self.head = nn.Identity()
            self.norm = nn.Identity()

        # ------------------------------------------------------------------
        # Enhancement 3: Multi-Scale Feature Fusion Head
        # Replaces the standard single-stage head with cross-stage fusion
        # When all_combined: also uses attn_pool inside each stage projection
        # ------------------------------------------------------------------
        if use_multiscale:
            stage_dims = cfg.embed_dim          # (96, 192, 384, 768)
            head_hidden_size = cfg.head_hidden_size or 0
            self.multiscale_head = MultiScaleFusionHead(
                stage_dims=stage_dims,
                fusion_dim=256,
                num_heads=8,
                num_classes=num_classes,
                drop_rate=drop_rate,
                head_hidden_size=head_hidden_size,
                use_attn_pool=use_attn_pool,  # all_combined: True → richer stage summaries
            )
            # Disable standard head
            self.head = nn.Identity()
            self.norm = nn.Identity()

    # ------------------------------------------------------------------
    # Forward: features
    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor):
        # Enhancement 1: apply residual pre-stem before TF stem
        if self.use_hybrid_stem:
            x = self.hybrid_pre_stem(x)

        x = self.stem(x)

        if self.use_multiscale:
            # Collect output from every stage for multi-scale fusion
            stage_outputs = []
            for stage in self.stages:
                x = stage(x)
                stage_outputs.append(x)
            x = self.norm(x)        # Identity when head_hidden_size is set
            return x, stage_outputs
        else:
            for stage in self.stages:
                x = stage(x)
            x = self.norm(x)
            return x

    # ------------------------------------------------------------------
    # Forward: full model
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Path 1: Multi-Scale Fusion (Enhancement 3, ± Enhancement 1)
        if self.use_multiscale:
            x, stage_outputs = self.forward_features(x)
            return self.multiscale_head(stage_outputs)

        # Path 2: Attention Pooling (Enhancement 2, ± Enhancement 1)
        x = self.forward_features(x)
        if self.use_attn_pool:
            x = self.attn_pool(x)            # [B, C, H, W] → [B, C]
            x = self.attn_head_norm(x)
            x = self.attn_head_mlp(x)
            x = self.attn_head_fc(x)
            return x

        # Path 3: Baseline (Enhancement 1 only, or no enhancement)
        return self.forward_head(x)