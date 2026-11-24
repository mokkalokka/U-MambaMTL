"""Build the umamba_mtl_single model for inference.

This file contains only the parts needed for inference from
``experiments/picai/umamba_mtl_single`` (trainer.py and config.yaml); the required
configuration values are set as defaults below.
"""

from functools import partial
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from munch import Munch

from umamba_mtl.module_selector import get_network

# Network and inference settings of the pretrained umamba_mtl_single model
# (defaults from experiments/picai/umamba_mtl_single/config.yaml).
DEFAULT_NETWORK_CONFIG = dict(
    name="UMambaBot",
    in_channels=3,
    num_classes=5,
    n_stages=7,
    features_per_stage=[32, 64, 128, 256, 256, 384, 512],
    kernel_sizes=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],
    n_conv_per_stage=[2, 2, 2, 2, 2, 2, 2],
    strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]],
    deep_supervision=False,
)
DEFAULT_INFERENCE_CONFIG = dict(
    spatial_size=(256, 256, 20),  # sliding-window roi size
    sw_batch_size=4,
    infer_overlap=0.25,
)


def split_umamba_mtl_single_output(logits):
    """Split the 5-channel model output into per-task probabilities.

    - channel 1: PCa detection logits (sigmoid -> probability)
    - channels 2-4: zonal anatomy logits (softmax over [background, PZ, TZ])
    """
    pca_prob = torch.sigmoid(logits[:, 1:2])
    zonal_prob = torch.softmax(logits[:, 2:], dim=1)
    return pca_prob, zonal_prob


def build_umamba_mtl_single(checkpoint_path, device="cpu", network_config=None, inference_config=None):
    """Build the umamba_mtl_single network and load the weights from a Lightning checkpoint.

    Returns ``(model, inferer)``, where ``inferer(x)`` runs sliding-window inference
    with the training roi size, batch size and overlap.
    """
    network_cfg = Munch(network=Munch(**{**DEFAULT_NETWORK_CONFIG, **(network_config or {})}))
    inference_cfg = {**DEFAULT_INFERENCE_CONFIG, **(inference_config or {})}

    model = get_network(network_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state)
    model.eval().to(device)

    inferer = partial(
        sliding_window_inference,
        roi_size=inference_cfg["spatial_size"],
        sw_batch_size=inference_cfg["sw_batch_size"],
        predictor=model,
        overlap=inference_cfg["infer_overlap"],
    )
    return model, inferer


def build_umamba_mtl_single_ensemble(weights_dir, folds=(0, 1, 2, 3, 4), device="cpu"):
    """Build the cross-validation ensemble; returns a list of ``(model, inferer)`` pairs."""
    weights_dir = Path(weights_dir)
    return [build_umamba_mtl_single(weights_dir / f"f{fold}.ckpt", device=device) for fold in folds]