"""Download pretrained model weights from Google Drive.

All models are downloaded into ``weights/<model_name>/`` in the repository
root, where ``<model_name>`` matches the experiment names used throughout
the repository (e.g. ``weights/umamba_mtl_single/``).

By default only the umamba_mtl_single weights are downloaded. Toggle the
flags below to fetch the other models as well.
"""

from pathlib import Path

import gdown

download_umamba_mtl_single_weights = True
download_umamba_mtl_dual_weights = False
download_umamba_prostate_weights = False
download_swin_unetr_weights = False
download_umamba_weights = False

# Resolve the repository root from this file's location, so the script can be
# run from anywhere (e.g. `uv run python utils/download_weights.py`).
repo_root = Path(__file__).resolve().parent.parent
weights_root = repo_root / "weights"

# Google Drive folder for each model, together with its download flag.
models = {
    "umamba_mtl_single": {
        "url": "https://drive.google.com/drive/folders/1-6nRI5nDcxs5mrIcjbtdm6UwTztPc6ao",
        "download": download_umamba_mtl_single_weights,
    },
    "umamba_mtl_dual": {
        "url": "https://drive.google.com/drive/folders/1kupUYIwAu9JHrIKs0UjS1_KqzivmwjAR",
        "download": download_umamba_mtl_dual_weights,
    },
    "umamba_prostate_seg": {
        "url": "https://drive.google.com/drive/folders/1H0HpP1BWFPmKCq2m38zNzt15lhyWID71",
        "download": download_umamba_prostate_weights,
    },
    "swin_unetr": {
        "url": "https://drive.google.com/drive/folders/1z-Vz9UOkzEvCYFDcmWHLwWa7oBH4vs1w",
        "download": download_swin_unetr_weights,
    },
    "umamba": {
        "url": "https://drive.google.com/drive/folders/1Wh94mEDAjl8flyOHQNxQpMmQqNfoGF-d",
        "download": download_umamba_weights,
    },
}

for model_name, model in models.items():
    if not model["download"]:
        continue
    target_folder = weights_root / model_name
    print(f"Downloading {model_name} weights to {target_folder} ...")
    # NOTE: `output` must not end with a path separator, otherwise gdown
    # nests the Google Drive root folder inside the target folder.
    gdown.download_folder(url=model["url"], output=str(target_folder), resume=True)

print("Done!")