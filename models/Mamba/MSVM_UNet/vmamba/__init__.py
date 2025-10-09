from __future__ import annotations
import os
import re
import copy
import torch
from loguru import logger
from functools import partial
from typing import Optional, Any
from .vmamba import VSSM

__all__ = ["VSSM", "ENCODERS", "build_model"]

DEFAULT_CONFIG = {
    "PATCH_SIZE": 4,
    "IN_CHANS": 3,
    "DEPTHS": [2, 2, 9, 2],
    "EMBED_DIM": 96,
    "SSM_D_STATE": 16,
    "SSM_RATIO": 2.0,
    "SSM_RANK_RATIO": 2.0,
    "SSM_DT_RANK": "auto",
    "SSM_ACT_LAYER": "silu",
    "SSM_CONV": 3,
    "SSM_CONV_BIAS": True,
    "SSM_DROP_RATE": 0.0,
    "SSM_INIT": "v0",
    "SSM_FORWARDTYPE": "v2",
    "MLP_RATIO": 4.0,
    "MLP_ACT_LAYER": "gelu",
    "MLP_DROP_RATE": 0.0,
    "PATCH_NORM": True,
    "NORM_LAYER": "ln",
    "DOWNSAMPLE": "v2",
    "PATCHEMBED": "v2",
    "POSEMBED": False,
    "GMLP": False,

    "DROP_PATH_RATE": 0.1,
    "USE_CHECKPOINT": False,
    "IMG_SIZE": 224,
}

def get_config(config: dict[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(DEFAULT_CONFIG)
    target.update(config)
    return target

def load_pretrained_ckpt(model: VSSM, ckpt: str) -> VSSM:
    logger.info(f"Loading weights from: {ckpt}")
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias"]

    t_device = next(model.parameters()).device
    model = model.cpu()
    ckpt = torch.load(ckpt, map_location="cpu")
    model_dict = model.state_dict()
    loaded_key_set = set()
    for kr, v in ckpt["model"].items():
        if kr in skip_params:
            logger.info(f"Skipping weights: {kr}")
            continue
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        if "ln_1" in kr:
            kr = kr.replace("ln_1", "norm")
        if "self_attention" in kr:
            kr = kr.replace("self_attention", "op")
        if kr in model_dict.keys():
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
            model_dict[kr] = v
            loaded_key_set.add(kr)
            logger.info(f"Loaded weights: {kr}")
        else:
            logger.info(f"Passing weights: {kr}")

    model.load_state_dict(model_dict)
    return model.to(t_device)

def build_model(config: dict[str, Any], ckpt: Optional[str] = None, **kwargs: Any) -> VSSM:
    config = get_config(config)
    model = VSSM(
        patch_size=config["PATCH_SIZE"],
        in_chans=config["IN_CHANS"],
        depths=config["DEPTHS"],
        dims=config["EMBED_DIM"],
        ssm_d_state=config["SSM_D_STATE"],
        ssm_ratio=config["SSM_RATIO"],
        ssm_rank_ratio=config["SSM_RANK_RATIO"],
        ssm_dt_rank=("auto" if config["SSM_DT_RANK"] == "auto" else int(config["SSM_DT_RANK"])),
        ssm_act_layer=config["SSM_ACT_LAYER"],
        ssm_conv=config["SSM_CONV"],
        ssm_conv_bias=config["SSM_CONV_BIAS"],
        ssm_drop_rate=config["SSM_DROP_RATE"],
        ssm_init=config["SSM_INIT"],
        forward_type=config["SSM_FORWARDTYPE"],
        mlp_ratio=config["MLP_RATIO"],
        mlp_act_layer=config["MLP_ACT_LAYER"],
        mlp_drop_rate=config["MLP_DROP_RATE"],
        drop_path_rate=config["DROP_PATH_RATE"],
        patch_norm=config["PATCH_NORM"],
        norm_layer=config["NORM_LAYER"],
        downsample_version=config["DOWNSAMPLE"],
        patchembed_version=config["PATCHEMBED"],
        gmlp=config["GMLP"],
        use_checkpoint=config["USE_CHECKPOINT"],
        posembed=config["POSEMBED"],
        imgsize=config["IMG_SIZE"],
        **kwargs
    )

    print(ckpt)
    if ckpt and os.path.exists(ckpt):
        model = load_pretrained_ckpt(model=model, ckpt=ckpt)
    return model

def build_tiny_0230s(**kwargs: Any) -> VSSM:
    return build_model({
        "IN_CHANS": kwargs.pop("in_channels", 3),
        "PATCH_SIZE": kwargs.pop("patch_size", 4),

        "EMBED_DIM": 96,
        "DEPTHS": [2, 2, 8, 2],
        "SSM_D_STATE": 1,
        "SSM_DT_RANK": "auto",
        "SSM_RATIO": 1.0,
        "SSM_CONV": 3,
        "SSM_CONV_BIAS": False,
        "SSM_FORWARDTYPE": "v05_noz",
        "MLP_RATIO": 4.0,
        "DOWNSAMPLE": "v3",
        "PATCHEMBED": "v2",
        "NORM_LAYER": "ln2d",

        "DROP_PATH_RATE": 0.2,
    }, **kwargs)

def build_small_0229s(**kwargs: Any) -> VSSM:
    patch_size = kwargs.pop("patch_size", 4)
    return build_model({
        "IN_CHANS": kwargs.pop("in_channels", 3),
        "PATCH_SIZE": patch_size,

        "EMBED_DIM": 96,
        "DEPTHS": [2, 2, 20, 2],
        "SSM_D_STATE": 1,
        "SSM_DT_RANK": "auto",
        "SSM_RATIO": 1.0,
        "SSM_CONV": 3,
        "SSM_CONV_BIAS": False,
        "SSM_FORWARDTYPE": "v05_noz",
        "MLP_RATIO": 4.0,
        "DOWNSAMPLE": "v3",
        "PATCHEMBED": "v2",
        "NORM_LAYER": "ln2d",

        "DROP_PATH_RATE": 0.3,
    }, **kwargs)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENCODERS = {
    "tiny_0230s": partial(
        build_tiny_0230s,
        ckpt=os.path.join(root, "pretrain/vssm1_tiny_0230s_ckpt_epoch_264.pth"),
    ),
    "small_0229s": partial(
        build_small_0229s,
        ckpt=os.path.join(root, "pretrain/vssm1_small_0229s_ckpt_epoch_240.pth"),
    ),
}
