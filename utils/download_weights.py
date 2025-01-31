import gdown
import shutil
import os

download_umamba_prostate_weights = True
download_swin_unetr_weights = False
download_umamba_weights = False
download_umamba_mtl_weights = True

gc_model_base_path = "../gc_algorithms/base_container/models/"
prostate_weights_path = "tmp/weights/prostate_weights.ckpt"


if download_umamba_prostate_weights:
    url = "https://drive.google.com/drive/folders/1H0HpP1BWFPmKCq2m38zNzt15lhyWID71"
    gdown.download_folder(url, output="./tmp/", resume=True)
    
if download_swin_unetr_weights:
    url = "https://drive.google.com/drive/folders/1z-Vz9UOkzEvCYFDcmWHLwWa7oBH4vs1w"
    gdown.download_folder(url, output= gc_model_base_path + "swin_unetr/", resume=True)

if download_umamba_weights:
    url = "https://drive.google.com/drive/folders/1Wh94mEDAjl8flyOHQNxQpMmQqNfoGF-d"
    gdown.download_folder(url, output="../gc_algorithms/base_container/models/umamba/", resume=True)
    
if download_umamba_mtl_weights:
    url = "https://drive.google.com/drive/folders/1kupUYIwAu9JHrIKs0UjS1_KqzivmwjAR"
    gdown.download_folder(url, output="../gc_algorithms/base_container/models/umamba_mtl/", resume=True)
     
# Distribute prostate weights into each gc_model folder:
if os.path.isfile(prostate_weights_path):    
    for download, model_name in zip([download_swin_unetr_weights, download_umamba_weights, download_umamba_mtl_weights], ["swin_unetr", "umamba", "umamba_mtl"]):
        dst_path_prostate = f"../gc_algorithms/base_container/models/{model_name}/weights/"
        if (not os.path.isfile(dst_path_prostate)) and download:
            shutil.copy(prostate_weights_path, dst_path_prostate)
    # Removes the temporary prostate_weights
    shutil.rmtree("tmp")

print("Done!")


