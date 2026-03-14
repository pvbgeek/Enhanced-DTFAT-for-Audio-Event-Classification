"""
the_new_audio_model_enhanced.py

Enhanced model builder for DTF-AT experiments.
Supports the following variants:
  - attn_pool      : Multi-Head Attention Pooling (replaces avg pool)
  - hybrid_stem    : Residual CNN block before TF-decoupled stem
  - multiscale     : Hierarchical Multi-Scale Feature Fusion Head
  - all_combined   : All three enhancements together

Base architecture: DTF-AT (MaxxVit adapted for audio)
Pretrained weights: timm/maxvit_small_tf_384.in1k (ImageNet-1K)
"""

import os
import torch
import torch.nn as nn
from audio_model_timm.models.maxxvit import MaxxVit
from audio_model_timm.models.maxxvit_enhanced import (
    MaxxVitEnhanced,
    MaxxVitCfg,
    MaxxVitConvCfg,
    MaxxVitTransformerCfg,
    checkpoint_filter_fn,
)
from audio_model_timm.models import load_pretrained_apr22

# ---- Cache path setup (same as original) ----
MNT_PATH = "absolute-path//pytorch_home/"
VOL_PATH = '/vol/research/fmodel_av/tony/pytorch_home/'

to_set_var = ["TORCH_HOME", "HF_HOME", "PIP_CACHE_DIR"]
SET_PATH = None
if os.path.isdir(MNT_PATH):
    SET_PATH = MNT_PATH
elif os.path.isdir(VOL_PATH):
    SET_PATH = VOL_PATH
if SET_PATH is not None:
    print(f"SET_PATH {SET_PATH}")
    for v in to_set_var:
        os.environ[v] = SET_PATH
else:
    print(f"Both {MNT_PATH} and {VOL_PATH} not present. Using default.")
# ---------------------------------------------


def get_enhanced_model(n_classes, imgnet=True, variant='attn_pool'):
    """
    Build and return the enhanced DTF-AT model for the specified variant.

    Args:
        n_classes  : Number of output classes (527 for AudioSet)
        imgnet     : Whether to load ImageNet pretrained weights
        variant    : One of ['attn_pool', 'hybrid_stem', 'multiscale', 'all_combined']

    Returns:
        Trained-ready PyTorch model
    """

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---- Shared model config ----
    img_size = (1024, 128)
    in_channel = 1
    window_size_l = [(8, 8), (8, 8), (8, 8), (8, 4)]
    use_nchw = False
    window_size_time_freq = [
        [(6, 3), (3, 6)],
        [(6, 3), (3, 6)],
        [(6, 3), (3, 6)],
        [(4, 3), (3, 4)],
    ]
    feat_map_size_list = [(512, 64), (256, 32), (128, 16), (64, 8)]
    drop_path_rate = 0.3

    def make_cfg():
        """Returns a fresh MaxxVitCfg — called for both base and enhanced model."""
        return MaxxVitCfg(
            embed_dim=(96, 192, 384, 768),
            depths=(2, 2, 5, 2),
            block_type=('M', 'M', 'M', 'M'),
            stem_width=64,
            stem_bias=True,
            conv_cfg=MaxxVitConvCfg(
                block_type='mbconv',
                expand_ratio=4.0,
                expand_output=True,
                kernel_size=3,
                group_size=1,
                pre_norm_act=False,
                output_bias=True,
                stride_mode='dw',
                pool_type='avg2',
                downsample_pool_type='avg2',
                padding='same',
                attn_early=False,
                attn_layer='se',
                attn_act_layer='silu',
                attn_ratio=0.25,
                init_values=1e-06,
                act_layer='gelu_tanh',
                norm_layer='batchnorm2d',
                norm_layer_cl='',
                norm_eps=0.001,
            ),
            transformer_cfg=MaxxVitTransformerCfg(
                dim_head=32,
                head_first=False,
                expand_ratio=4.0,
                expand_first=True,
                shortcut_bias=True,
                attn_bias=True,
                attn_drop=0.0,
                proj_drop=0.0,
                pool_type='avg2',
                rel_pos_type='bias_tf',
                rel_pos_dim=512,
                partition_ratio=32,
                window_size=None,
                grid_size=None,
                no_block_attn=False,
                use_nchw_attn=use_nchw,
                init_values=None,
                act_layer='gelu_tanh',
                norm_layer='layernorm2d',
                norm_layer_cl='layernorm',
                norm_eps=1e-05,
            ),
            head_hidden_size=768,
            weight_init='vit_eff',
        )

    def load_pretrained_maxvit_small(m):
        """Load ImageNet weights from HuggingFace hub into model m."""
        pretrained_cfg = {
            'url': '',
            'hf_hub_id': 'timm/maxvit_small_tf_384.in1k',
            'architecture': 'maxvit_small_tf_224',
            'tag': 'in1k',
            'custom_load': False,
            'input_size': (3, 224, 224),
            'fixed_input_size': True,
            'interpolation': 'bicubic',
            'crop_pct': 0.95,
            'crop_mode': 'center',
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': 1000,
            'pool_size': (7, 7),
            'first_conv': 'stem.conv1',
            'classifier': 'head.fc',
        }
        load_pretrained_apr22(
            m,
            pretrained_cfg=pretrained_cfg,
            num_classes=n_classes,
            in_chans=3,
            filter_fn=checkpoint_filter_fn,
            strict=False,
        )
        return m

    # ---- Variant flags ----
    use_attn_pool   = variant in ('attn_pool',   'all_combined')
    use_hybrid_stem = variant in ('hybrid_stem', 'all_combined')
    use_multiscale  = variant in ('multiscale',  'all_combined')

    print(f"\n=== Building Enhanced Model: variant={variant} ===")
    print(f"    use_attn_pool   = {use_attn_pool}")
    print(f"    use_hybrid_stem = {use_hybrid_stem}")
    print(f"    use_multiscale  = {use_multiscale}")

    # ---- Build enhanced model ----
    model = MaxxVitEnhanced(
        cfg=make_cfg(),
        num_classes=n_classes,
        in_chans=in_channel,
        img_size=img_size,
        window_size_list=window_size_l,
        drop_path_rate=drop_path_rate,
        window_size_time_freq=window_size_time_freq,
        feat_map_size_list=feat_map_size_list,
        use_attn_pool=use_attn_pool,
        use_hybrid_stem=use_hybrid_stem,
        use_multiscale=use_multiscale,
    )

    model.train(False)

    if imgnet:
        print("Loading ImageNet pretrained weights via clean base model...")

        # ----------------------------------------------------------------
        # WHY: load_pretrained_apr22 checks for unexpected keys and will
        # error on attn_pool / hybrid_stem / multiscale_head components.
        # FIX: load weights into a clean base MaxxVit first, then copy
        # shared weights into the enhanced model using strict=False.
        # New components (attn_pool etc.) keep their random init.
        # ----------------------------------------------------------------
        base_model = MaxxVit(
            cfg=make_cfg(),
            num_classes=n_classes,
            in_chans=in_channel,
            img_size=img_size,
            window_size_list=window_size_l,
            drop_path_rate=drop_path_rate,
            window_size_time_freq=window_size_time_freq,
            feat_map_size_list=feat_map_size_list,
        )
        base_model.train(False)
        base_model = load_pretrained_maxvit_small(base_model)

        # Copy shared weights — strict=False skips new components
        missing, unexpected = model.load_state_dict(
            base_model.state_dict(), strict=False
        )
        print(f"Pretrained weights loaded successfully.")
        print(f"  New components (randomly initialized): {len(missing)} keys")
        print(f"  Unexpected keys ignored:               {len(unexpected)} keys")
        if missing:
            new_modules = set(k.split('.')[0] for k in missing)
            print(f"  New modules initialized from scratch: {new_modules}")

        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.train(True)

    # ---- Sanity forward pass ----
    x = torch.randn(1, in_channel, img_size[0], img_size[1])
    with torch.no_grad():
        out = model(x)
    print(f"Enhanced model forward pass output shape: {out.shape}")
    print(f"Total trainable parameters: {count_parameters(model)/1e6:.3f} million")

    return model


if __name__ == "__main__":
    for variant in ['attn_pool', 'hybrid_stem', 'multiscale', 'all_combined']:
        print(f"\n{'='*60}")
        m = get_enhanced_model(n_classes=527, imgnet=False, variant=variant)
        x = torch.randn(2, 1, 1024, 128)
        with torch.no_grad():
            out = m(x)
        print(f"variant={variant} | output={out.shape}")